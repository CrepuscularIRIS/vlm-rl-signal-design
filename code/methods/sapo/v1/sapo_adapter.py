"""SAPO-v4 shared-logit V/Q adaptation to the frozen Caption v5.1 EasyR1 runner.
One full causal visual sequence per trajectory; read all turn boundaries in one pass.
No sigmoid, new value head, group filter, branch rollout, or observation policy loss.
"""
from __future__ import annotations
import inspect,json,os,sys,textwrap,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sapo_official import compute_arq_gae_targets,compute_arq_value_loss
TOKENS=(151659,151661) # existing FIM prefix/suffix, never image/chat boundaries
MAX_TURNS=9
EXTRA=['sapo_vpos','sapo_qpos','sapo_turn_mask']
TARGETS=['sapo_old_v','sapo_old_q','sapo_v_target','sapo_q_target']


def add_boundaries(batch):
    mask=batch['response_mask'].bool();B,R=mask.shape;P=batch['input_ids'].shape[1]-R
    v=torch.full((B,MAX_TURNS),P-1,dtype=torch.long,device=mask.device);q=v.clone();valid=torch.zeros_like(v,dtype=torch.bool)
    for i,row in enumerate(mask):
        starts=torch.nonzero(row & ~torch.cat([torch.zeros(1,dtype=torch.bool,device=row.device),row[:-1]])).flatten()
        ends=torch.nonzero(row & ~torch.cat([row[1:],torch.zeros(1,dtype=torch.bool,device=row.device)])).flatten()
        assert len(starts)==len(ends)<=MAX_TURNS
        for j,(a,b) in enumerate(zip(starts,ends)):
            v[i,j]=P+a-1;q[i,j]=P+b;valid[i,j]=True
    batch['sapo_vpos']=v;batch['sapo_qpos']=q;batch['sapo_turn_mask']=valid
    assert torch.all(batch['attention_mask'].gather(1,v).bool())
    assert torch.all(batch['attention_mask'].gather(1,q).bool())
    assert not torch.isin(batch['input_ids'][batch['attention_mask'].bool()],torch.tensor(TOKENS,device=mask.device)).any(), 'Reserved FIM token in actual context'


def mask_logits(logits):
    logits[...,list(TOKENS)]=torch.finfo(logits.dtype).min
    return logits


def rewrite(cls,name,edits,extra=None):
    fn=getattr(cls,name);src=textwrap.dedent(inspect.getsource(fn))
    for old,new in edits:
        assert src.count(old)==1,(name,old,src.count(old))
        src=src.replace(old,new)
    ns=dict(vars(sys.modules[fn.__module__]));ns.update(extra or {})
    exec(compile(src,'<sapo '+name+'>','exec'),ns)
    setattr(cls,name,ns[name])


def capture_forward(original):
    def forward(self,micro_batch,temperature):
        assert self.config.ulysses_size==1
        att=micro_batch['attention_mask'];B,S=att.shape
        pos=torch.stack([micro_batch['sapo_vpos'],micro_batch['sapo_qpos']],-1)
        root=getattr(self.actor_module,'_fsdp_wrapped_module',self.actor_module)
        head=root.get_output_embeddings();head=getattr(head,'_fsdp_wrapped_module',head)
        if self.config.padding_free:
            flat=torch.nonzero(att.reshape(-1)).flatten()
            absolute=pos+torch.arange(B,device=pos.device)[:,None,None]*S
            indices=torch.searchsorted(flat,absolute)
            assert torch.equal(flat[indices],absolute)
        else:
            indices=pos
        captured=[]
        def hook(module,args):
            hidden=args[0]
            if self.config.padding_free:
                selected=hidden.reshape(-1,hidden.shape[-1])[indices]
            else:
                selected=hidden[torch.arange(B,device=hidden.device)[:,None,None],indices]
            ids=torch.tensor(TOKENS,device=module.weight.device)
            z=F.linear(selected,module.weight.index_select(0,ids),None if module.bias is None else module.bias.index_select(0,ids))
            captured.append((z[...,0]-z[...,1]).float()) # [B,T,V/Q], tau=1, Rmax=1
        handle=head.register_forward_pre_hook(hook)
        try:lp=original(self,micro_batch,temperature)
        finally:handle.remove()
        assert len(captured)==1,'Expected one causal LM-head pass'
        self._sapo_last=captured[0]
        assert torch.isfinite(self._sapo_last).all()
        return lp
    return forward


@torch.no_grad()
def compute_log_prob(self,data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch,restore_dynamic_batch
    self.actor_module.eval();temp=data.meta_info['temperature']
    data=data.select(['input_ids','attention_mask','position_ids','responses']+EXTRA,['multi_modal_inputs'])
    if self.config.dynamic_batching:
        micros,order=prepare_dynamic_batch(data,max_token_len=self.config.micro_batch_size_per_device_for_experience*data.batch['input_ids'].size(-1))
    else:micros=data.split(self.config.micro_batch_size_per_device_for_experience);order=None
    lps=[];values=[]
    for micro in micros:
        lp=self._forward_micro_batch({**micro.batch,**micro.non_tensor_batch},temp)
        lps.append(lp);values.append(self._sapo_last)
    lp=torch.cat(lps);vq=torch.cat(values)
    if order is not None:
        lp=restore_dynamic_batch(lp,order);vq=restore_dynamic_batch(vq,order)
    self._sapo_old={'sapo_old_v':vq[:,:,0],'sapo_old_q':vq[:,:,1]}
    return lp


_CALL=0
@torch.no_grad()
def compute_advantage(data,adv_estimator=None,gamma=.95,lam=.95):
    global _CALL
    b=data.batch;m=b['sapo_turn_mask'].bool();rows,turns=torch.where(m);R=b['responses'].shape[-1];P=b['input_ids'].shape[-1]-R
    rewards=b['token_level_rewards'].sum(-1);assert ((rewards==0)|(rewards==1)).all(), 'Expected unchanged binary outcome'
    count=m.sum(-1);r=torch.where(turns==count[rows]-1,rewards[rows],0.)
    dones=turns==count[rows]-1
    result=compute_arq_gae_targets(r,dones,b['sapo_old_v'][m],b['sapo_old_q'][m],torch.ones((len(rows),1)),rows.tolist(),turns,float(gamma),float(lam),normalize=True,q_target_mode='sarsa_td0',advantage_mode='gae',invalid_action_penalty_coef=0.)
    advantages=torch.zeros_like(b['response_mask'],dtype=torch.float32);returns=advantages.clone()
    for k,(i,t) in enumerate(zip(rows.tolist(),turns.tolist())):
        lo=int(b['sapo_vpos'][i,t])-P+1;hi=int(b['sapo_qpos'][i,t])-P+1
        assert b['response_mask'][i,lo:hi].all()
        advantages[i,lo:hi]=result['turn_advantages'][k];returns[i,lo:hi]=result['v_targets'][k]
    b['advantages']=advantages;b['returns']=returns
    for kind in ['v','q']:
        target=torch.zeros_like(b['sapo_old_'+kind]);target[m]=result[kind+'_targets'].clamp(-1,1) # current official ray_trainer
        b['sapo_'+kind+'_target']=target
    assert not advantages[~b['response_mask'].bool()].any()
    _CALL+=1
    rec={'prepared_call':_CALL,'time':time.time(),'n_trajectories':len(rewards),'n_turns':int(m.sum()),'Y':float(rewards.mean()),'turn_positive':int((result['turn_advantages']>0).sum()),'turn_negative':int((result['turn_advantages']<0).sum()),'turn_zero':int((result['turn_advantages']==0).sum()),'raw_adv_min':float(result['raw_turn_advantages'].min()),'raw_adv_max':float(result['raw_turn_advantages'].max()),'v_target_clipped':int((result['v_targets'].abs()>1).sum()),'q_target_clipped':int((result['q_targets'].abs()>1).sum()),'qid':[mm.get('caption_qid') for mm in data.non_tensor_batch['multi_modal_data']]}
    out=Path(os.environ['SAPO_OUTPUT']);out.mkdir(exist_ok=True)
    with (out/'sapo_targets.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    if _CALL==1:
        torch.save({k:v.detach().cpu() for k,v in b.items()},out/'first_batch.pt')
        (out/'first_metadata.json').write_text(json.dumps(data.non_tensor_batch['multi_modal_data'].tolist(),default=str))
    print('[sapo prepared] '+json.dumps({k:v for k,v in rec.items() if k!='qid'}),flush=True)
    return data


def update_policy(self,data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch
    from verl.trainer.core_algos import compute_policy_loss
    from verl.utils.py_functional import append_to_dict
    self.actor_module.train();temp=data.meta_info['temperature'];metrics=defaultdict(list)
    assert self.config.ppo_epochs==1 and self.config.loss_avg_mode=='token'
    assert self.config.kl_coef==0 and 'ref_log_probs' not in data.batch
    selected=data.select(['input_ids','attention_mask','position_ids','responses','response_mask','old_log_probs','advantages']+EXTRA+TARGETS,['multi_modal_inputs'])
    for mini in selected.split(self.config.global_batch_size_per_device):
        nt=mini.batch['response_mask'].sum().float();nv=mini.batch['sapo_turn_mask'].sum().float()
        dist.all_reduce(nt);dist.all_reduce(nv);assert nt>0 and nv>0
        if self.config.dynamic_batching:
            micros,_=prepare_dynamic_batch(mini,max_token_len=self.config.micro_batch_size_per_device_for_update*mini.batch['input_ids'].size(-1))
        else:micros=mini.split(self.config.micro_batch_size_per_device_for_update)
        self.actor_optimizer.zero_grad();pg_sum=0.;v_sum=0.;q_sum=0.;diff_max=0.;dv=0.;dq=0.
        for micro in micros:
            x={**micro.batch,**micro.non_tensor_batch};mask=x['response_mask'];vm=x['sapo_turn_mask'].bool()
            lp=self._forward_micro_batch(x,temp);v,q=self._sapo_last.unbind(-1)
            assert torch.isfinite(lp).all()
            pg,_=compute_policy_loss(x['old_log_probs'],lp,x['advantages'],mask,.2,.2,3.,1.,1.,'default','token')
            vl,_=compute_arq_value_loss(v,x['sapo_old_v'],x['sapo_v_target'],.2,vm)
            ql,_=compute_arq_value_loss(q,x['sapo_old_q'],x['sapo_q_target'],.2,vm)
            loss=pg*mask.sum()*self.world_size/nt+.1*(vl+ql)*vm.sum()*self.world_size/nv
            assert torch.isfinite(loss)
            # Branch derivatives at the shared boundary outputs, not claimed as isolated backbone norms.
            d1=torch.autograd.grad(vl,v,retain_graph=True)[0];d2=torch.autograd.grad(ql,q,retain_graph=True)[0]
            dv+=float(d1.norm().detach());dq+=float(d2.norm().detach())
            loss.backward()
            pg_sum+=float(pg.detach()*mask.sum()/nt*self.world_size)
            v_sum+=float(vl.detach()*vm.sum()/nv*self.world_size);q_sum+=float(ql.detach()*vm.sum()/nv*self.world_size)
            if mask.any():diff_max=max(diff_max,float((lp.detach()-x['old_log_probs']).abs()[mask.bool()].max()))
        # Fail loudly instead of quietly skipping a non-finite update.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        norm=self.actor_module.clip_grad_norm_(self.config.max_grad_norm) if isinstance(self.actor_module,FSDP) else torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(),self.config.max_grad_norm)
        assert torch.isfinite(norm), 'Non-finite actual VLM combined gradient'
        probes=[(p,p.detach().reshape(-1)[:4096].clone()) for p in self.actor_module.parameters() if p.numel()] if os.environ.get('SAPO_PHASE')=='smoke' else []
        lr=self.actor_optimizer.param_groups[0]['lr'];self.actor_optimizer.step();self.actor_optimizer.zero_grad()
        changed=sum(int((p.detach().reshape(-1)[:4096]!=before).sum()) for p,before in probes)
        event={'time':time.time(),'rank':self.rank,'pg_loss':pg_sum,'v_loss':v_sum,'q_loss':q_sum,'grad_norm':float(norm),'v_boundary_grad_norm_sum':dv,'q_boundary_grad_norm_sum':dq,'old_logprob_max_diff':diff_max,'global_tokens':float(nt),'global_turns':float(nv),'lr':lr,'changed_parameter_probe_elements':changed if probes else None}
        with (Path(os.environ['SAPO_OUTPUT'])/f'updates_rank{self.rank}.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        append_to_dict(metrics,{'actor/'+k:v for k,v in event.items() if k not in ['time','rank'] and v is not None})
    return metrics


def install():
    from verl.workers.actor.dp_actor import DataParallelPPOActor as A
    from verl.workers.fsdp_workers import FSDPWorker as W
    import verl.trainer.ray_trainer as RT
    if getattr(A,'_sapo_installed',False):return
    # Called after legacy source patches but before lazy multimodal/prefetch wrappers.
    rewrite(A,'_forward_micro_batch',[
        ('logits_rmpad.div_(temperature)','_sapo_mask(logits_rmpad)\n        logits_rmpad.div_(temperature)'),
        ('logits.div_(temperature)','_sapo_mask(logits)\n        logits.div_(temperature)')],{'_sapo_mask':mask_logits})
    A._forward_micro_batch=capture_forward(A._forward_micro_batch)
    A.compute_log_prob=compute_log_prob;A.update_policy=update_policy
    rewrite(W,'compute_log_probs',[( 'tensors={"old_log_probs": output}', 'tensors={"old_log_probs": output, **self.actor._sapo_old}')])
    RT.compute_advantage=compute_advantage
    rewrite(RT.RayPPOTrainer,'__init__', [(
        '    if (\n        config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)\n        and config.worker.rollout.n == 1\n    ):\n        raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")',
        '    assert config.worker.rollout.n == 1, "SAPO requires exactly one rollout per task"\n    assert not config.algorithm.online_filtering and config.algorithm.disable_kl\n    assert config.algorithm.kl_coef == 0 and not self.use_critic\n    assert config.worker.actor.ppo_epochs == 1'
    )])
    if os.environ.get('SAPO_PHASE')=='smoke':
        rewrite(RT.RayPPOTrainer,'fit', [('    # perform validation after training\n    if self.val_reward_fn is not None:', '    # perform validation after training\n    if self.val_reward_fn is not None and False: # bounded smoke skips final dev')])
    A._sapo_installed=True
    print('[sapo] shared raw logit gap; FIM readouts; GAE/SARSA; current-official target clamp; no critic',flush=True)
