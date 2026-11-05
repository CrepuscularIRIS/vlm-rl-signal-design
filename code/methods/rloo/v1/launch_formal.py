"""Execute the frozen existing v53 pair after a recorded full-step integration review."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time
P=Path('$PROJECT');E=P/'evidence/method_v53_20261029';C=P/'grpo_v53_dl'
D=Path('$ROOT/datasets/grpo_caption_v51_4k')
GIB=1024**3
RESERVATION_GIB=200
def allocation(arm,phase):
 return ([0,1] if arm=='control' else [2,3],
         f'k8_{arm}_20261029' if phase=='formal' else ('pr2c1029' if arm=='control' else 'pr2d1029'),
         arm if phase=='formal' else f'resource_{arm}_2',32 if phase=='formal' else 1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['control','decision'])
 ap.add_argument('--phase',choices=['resource','formal'],default='formal');a=ap.parse_args()
 gate=json.loads((E/'REVIEW_GATE.json').read_text())
 if gate.get('verdict')!='pass' or gate.get('formal_protocol')!='4k8_v53_existing_design':raise RuntimeError('Signal gate not passed')
 signal=json.loads((E/'signal.json').read_text())
 if signal['state']!='awaiting_signal_review' or signal['exit_code']!=0:raise RuntimeError('Signal run not complete')
 gpus,tag,state_name,target=allocation(a.arm,a.phase)
 lock=(E/f'gpu_{gpus[0]}_{gpus[1]}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 admission=(E/'admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=E/f'{state_name}.json';out=P/f'outputs/v53_{tag}'
 if status.exists() or out.exists():raise RuntimeError('Existing arm; do not overwrite or restart blindly')
 if a.phase=='formal':
  resource=json.loads((E/'RESOURCE_GATE.json').read_text())
  if resource.get('verdict')!='pass':raise RuntimeError('Parallel resource gate not passed')
  for arm in ['control','decision']:
   rs=json.loads((E/f'resource_{arm}_2.json').read_text())
   if rs.get('state')!='completed' or rs.get('exit_code')!=0:raise RuntimeError('Parallel resource run incomplete')
   if rs.get('source_hashes')!=signal['source_hashes']:raise RuntimeError('Resource run used different source')
   if rs.get('cgroup_peak_bytes',440*GIB)>410*GIB:raise RuntimeError('Resource run lacks host headroom')
 for name,h in signal['source_hashes'].items():
  if sha(C/name)!=h:raise RuntimeError('Frozen source changed: '+name)
 if sha(D/'train.parquet')!=signal['train_sha256'] or sha(D/'val_nosft.parquet')!=signal['val_sha256']:raise RuntimeError('Frozen data changed')
 used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in used.splitlines()}
 if any(memory[i]>1024 for i in gpus):raise RuntimeError('Assigned GPUs occupied')
 current=int(Path('/sys/fs/cgroup/memory.current').read_text())
 limit_text=Path('/sys/fs/cgroup/memory.max').read_text().strip()
 if limit_text=='max':raise RuntimeError('Expected bounded shared cgroup')
 limit=int(limit_text)
 active=[]
 for name in ['resource_control','resource_decision','resource_control_2','resource_decision_2','control','decision']:
  sp=E/f'{name}.json'
  if sp.exists():
   st=json.loads(sp.read_text())
   if st.get('state') in ['starting','running']:active.append(st)
 if any(st.get('phase')!=a.phase for st in active):raise RuntimeError('Another experiment phase still active')
 # Reserve both workers' future peaks, rather than using current RSS alone.
 if (len(active)+1)*RESERVATION_GIB*GIB+30*GIB>limit:raise RuntimeError('Shared reservations exceed cgroup headroom')
 if current+RESERVATION_GIB*GIB>limit-20*GIB:raise RuntimeError('Insufficient measured host headroom')
 if os.statvfs(P).f_bavail*os.statvfs(P).f_frsize<200*1024**3:raise RuntimeError('Insufficient disk headroom')
 beta='0' if a.arm=='control' else '0.00390625'
 env=os.environ.copy();env.update(LAMBDA='0.01',BETA=beta,K='4',CAPTION_LAZY_MM='1',PYTHONUNBUFFERED='1',RAY_DEDUP_LOGS='0',RAY_ADDRESS='local')
 cmd=['bash',str(C/'grpo_v53_dl.sh'),tag,','.join(map(str,gpus)),
      f'data.train_files={D}/train.parquet',f'data.val_files={D}/val_nosft.parquet',
      'worker.rollout.n=8','worker.rollout.seed=42','worker.actor.ppo_epochs=1',
      'worker.actor.optim.lr_warmup_steps=1',f'trainer.max_steps={target}','trainer.total_epochs=1',
      f'trainer.val_before_train={"true" if a.phase=="formal" else "false"}','trainer.val_freq=25',f'trainer.save_freq={8 if a.phase=="formal" else 1}',
      'trainer.save_model_only=false','trainer.save_limit=2']
 state=dict(state='starting',arm=a.arm,phase=a.phase,command=cmd,output=str(out),gpus=gpus,target_steps=target,
   reserved_host_GiB=RESERVATION_GIB,ray_address='local',
   beta=float(beta),started=time.strftime('%F %T %z'),supervisor_pid=os.getpid(),
   source_hashes=signal['source_hashes'],train_sha256=signal['train_sha256'],val_sha256=signal['val_sha256'],
   origin='original_2epoch_A_SFT_no_RL_warm_start',signal_gate_sha256=sha(E/'REVIEW_GATE.json'))
 write(status,state)
 with (E/f'{state_name}.log').open('x') as log,(E/f'{state_name}_memory.jsonl').open('x') as mem:
  p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(state='running',child_pid=p.pid);write(status,state);peak=0
  fcntl.flock(admission,fcntl.LOCK_UN);admission.close()
  while p.poll() is None:
   n=int(Path('/sys/fs/cgroup/memory.current').read_text());peak=max(peak,n)
   mem.write(json.dumps({'time':time.time(),'cgroup_bytes':n})+'\n');mem.flush();time.sleep(5)
  state.update(exit_code=p.returncode,finished=time.strftime('%F %T %z'),cgroup_peak_bytes=peak)
 expected=[out/f'ckpt/global_step_{target}/dataloader.pt']+[out/f'ckpt/global_step_{target}/actor'/f'{k}_world_size_2_rank_{r}.pt' for k in ['model','optim','extra_state'] for r in [0,1]]
 state['missing_checkpoint_files']=[str(p) for p in expected if not p.is_file() or not p.stat().st_size]
 tracker=out/'ckpt/checkpoint_tracker.json'
 state['checkpoint_tracker']=json.loads(tracker.read_text()) if tracker.exists() else {}
 state['state']='completed' if p.returncode==0 and not state['missing_checkpoint_files'] and state['checkpoint_tracker'].get('last_global_step')==target else 'failed'
 write(status,state)
if __name__=='__main__':main()
