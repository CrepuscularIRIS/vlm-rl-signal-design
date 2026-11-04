"""DTPO outcome-only loss ablation on unchanged multi-turn Caption v5.1.
No crop judge, new reward, SAPO readout tokens, dynamic sampling or extra rollouts.
Official loss control_alpha=.5; outcome group normalization uses official sample std.
All non-final assistant segments form the tool/intermediate term; final segment the answer term.
A truncated final tool segment is still a final segment, not a verified answer.
"""
from __future__ import annotations
import inspect,json,os,sys,textwrap,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.distributed as dist
from dtpo_official import compute_DTPO_advantage


def turn_mask(response_mask):
    m=response_mask.bool();previous=torch.cat([torch.zeros_like(m[:,:1]),m[:,:-1]],dim=1)
    return (m & ~previous).long().cumsum(-1)*m.long()


def partitions(turns,response_mask):
    last=turns.max(-1,keepdim=True).values;m=response_mask.bool()
    return (turns>0)&(turns<last)&m,(turns>0)&(turns==last)&m


def token_loss(old,lp,adv,low=.2,high=.24):
    ratio=(lp-old).exp()
    return torch.maximum(-adv*ratio,-adv*ratio.clamp(1-low,1+high))


def rewrite(cls,name,edits):
    fn=getattr(cls,name);src=textwrap.dedent(inspect.getsource(fn))
    for old,new in edits:
        assert src.count(old)==1,(name,old,src.count(old));src=src.replace(old,new)
    ns=dict(vars(sys.modules[fn.__module__]));exec(compile(src,'<dtpo '+name+'>','exec'),ns);setattr(cls,name,ns[name])

_CALL=0
@torch.no_grad()
def compute_advantage(data,adv_estimator=None,gamma=1.,lam=1.):
    global _CALL
    b=data.batch;m=b['response_mask'];tm=turn_mask(m);tool,answer=partitions(tm,m)
    y=b['token_level_rewards'].sum(-1);assert ((y==0)|(y==1)).all()
    # Official function only supports labels 1/2. The outcome-only tool stream is zero;
    # map every intermediate segment to 1 and the final to 2, preserving all action tokens.
    mapped=torch.where(tool,1,torch.where(answer,2,0));uid=data.non_tensor_batch['uid']
    a,ret,at,ao=compute_DTPO_advantage(torch.zeros_like(b['token_level_rewards']),b['token_level_rewards'],mapped,m,uid,lambda_coeff=.3)
    assert torch.isfinite(a).all() and not at.any()
    assert not a[~m.bool()].any()
    b['advantages']=a;b['returns']=ret
    _CALL+=1;out=Path(os.environ['DTPO_OUTPUT']);out.mkdir(exist_ok=True)
    mm=data.non_tensor_batch['multi_modal_data'];groups=defaultdict(list)
    for u,v in zip(uid,y.tolist()):groups[str(u)].append(v)
    assert all(len(v)==8 for v in groups.values()),'Expected n8 intact groups'
    rec={'time':time.time(),'prepared_call':_CALL,'trajectories':len(y),'groups':len(groups),'mixed_groups':sum(0<sum(v)<len(v) for v in groups.values()),'Y':float(y.mean()),'tool_tokens':int(tool.sum()),'answer_tokens':int(answer.sum()),'assistant_turns':int(tm.max(-1).values.sum()),'adv_positive_trajectories':int((ao>0).sum()),'adv_negative_trajectories':int((ao<0).sum()),'adv_zero_trajectories':int((ao==0).sum()),'tool_reward_nonzero':0,'generated_policy_tokens':int(m.sum()),'processed_tokens':sum(int(x.get('caption_proc_tokens',0)) for x in mm),'crops':sum(int(x.get('caption_crops',0)) for x in mm)}
    with (out/'dtpo_targets.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    if _CALL==1:
        torch.save({k:v.detach().cpu() for k,v in b.items()},out/'first_batch.pt')
        (out/'first_metadata.json').write_text(json.dumps(mm.tolist(),default=str));(out/'first_uids.json').write_text(json.dumps(uid.tolist(),default=str))
    print('[dtpo prepared] '+json.dumps(rec),flush=True);return data


def update_policy(self,data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch
    from verl.utils.py_functional import append_to_dict
    self.actor_module.train();temp=data.meta_info['temperature'];metrics=defaultdict(list)
    assert self.config.ppo_epochs==1 and self.config.kl_coef==0 and 'ref_log_probs' not in data.batch
    assert self.config.clip_ratio_low==.2 and self.config.clip_ratio_high==.24
    selected=data.select(['input_ids','attention_mask','position_ids','responses','response_mask','old_log_probs','advantages'],['multi_modal_inputs'])
    for mini in selected.split(self.config.global_batch_size_per_device):
        tool,answer=partitions(turn_mask(mini.batch['response_mask']),mini.batch['response_mask'])
        nt=tool.sum().float();na=answer.sum().float();dist.all_reduce(nt);dist.all_reduce(na);assert na>0
        if self.config.dynamic_batching:
            micros,_=prepare_dynamic_batch(mini,max_token_len=self.config.micro_batch_size_per_device_for_update*mini.batch['input_ids'].size(-1))
        else:micros=mini.split(self.config.micro_batch_size_per_device_for_update)
        self.actor_optimizer.zero_grad();tool_sum=0.;answer_sum=0.;diff_max=0.;clip_sum=0.
        for micro in micros:
            x={**micro.batch,**micro.non_tensor_batch};m=x['response_mask'];it,ia=partitions(turn_mask(m),m)
            lp=self._forward_micro_batch(x,temp);assert torch.isfinite(lp).all()
            tl=token_loss(x['old_log_probs'],lp,x['advantages'])
            t=(tl*it).sum()/nt.clamp_min(1)*self.world_size
            a=(tl*ia).sum()/na*self.world_size;loss=.5*t+a
            assert torch.isfinite(loss);loss.backward()
            tool_sum+=float(t.detach());answer_sum+=float(a.detach())
            diff_max=max(diff_max,float((lp.detach()-x['old_log_probs']).abs()[m.bool()].max()))
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        norm=self.actor_module.clip_grad_norm_(self.config.max_grad_norm) if isinstance(self.actor_module,FSDP) else torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(),self.config.max_grad_norm)
        assert torch.isfinite(norm),'Non-finite actual VLM gradient'
        probes=[(p,p.detach().reshape(-1)[:4096].clone()) for p in self.actor_module.parameters() if p.numel()] if os.environ.get('DTPO_PHASE')=='smoke' else []
        lr=self.actor_optimizer.param_groups[0]['lr'];self.actor_optimizer.step();self.actor_optimizer.zero_grad()
        changed=sum(int((p.detach().reshape(-1)[:4096]!=before).sum()) for p,before in probes)
        event={'time':time.time(),'rank':self.rank,'tool_loss':tool_sum,'answer_loss':answer_sum,'pg_loss':.5*tool_sum+answer_sum,'grad_norm':float(norm),'global_tool_tokens':float(nt),'global_answer_tokens':float(na),'old_logprob_max_diff':diff_max,'lr':lr,'changed_parameter_probe_elements':changed if probes else None}
        with (Path(os.environ['DTPO_OUTPUT'])/f'updates_rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        append_to_dict(metrics,{'actor/'+k:v for k,v in event.items() if k not in ['time','rank'] and v is not None})
    return metrics


def install():
    from verl.workers.actor.dp_actor import DataParallelPPOActor as A
    import verl.trainer.ray_trainer as RT
    if getattr(A,'_dtpo_installed',False):return
    A.update_policy=update_policy;RT.compute_advantage=compute_advantage
    if os.environ.get('DTPO_PHASE')=='smoke':
        rewrite(RT.RayPPOTrainer,'fit',[('    # perform validation after training\n    if self.val_reward_fn is not None:','    # perform validation after training\n    if self.val_reward_fn is not None and False: # bounded smoke skips final dev')])
    A._dtpo_installed=True
    print('[dtpo] outcome-only multi-turn loss ablation; .5 intermediate + final; no tool judge/reward/filter',flush=True)
