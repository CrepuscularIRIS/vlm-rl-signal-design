"""One-shot supervisor for the authorized DTPO work package; no retry/cleanup."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time,signal
P=Path('$PROJECT');C=P/'dtpo_vlm_20261101';E=P/'evidence/dtpo_vlm_20261101';G=2**30

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def stop_owned(root,tag):
 p=Path(f'/proc/{root}/cmdline')
 if not p.exists() or tag.encode() not in p.read_bytes():raise RuntimeError('Owned process identity mismatch; no signal sent')
 parents={}
 for q in Path('/proc').iterdir():
  if q.name.isdigit():
   try:parents[int(q.name)]=int((q/'stat').read_text().rsplit(')',1)[1].split()[1])
   except (OSError,ValueError):pass
 kids={root}
 while True:
  new={pid for pid,ppid in parents.items() if ppid in kids}-kids
  if not new:break
  kids|=new
 ids={}
 for pid in kids:
  try:ids[pid]=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
  except OSError:pass
 write(E/f'{tag}_guard_processes.json',ids)
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid,ident in ids.items():
   try:
    if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==ident:os.kill(pid,sig)
   except OSError:pass
  if sig==signal.SIGTERM:time.sleep(5)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['smoke','formal']);ap.add_argument('--attempt',type=int,default=1);args=ap.parse_args();phase=args.phase;key=f'{phase}_{args.attempt}'
 tag=f'{"s" if phase=="smoke" else "f"}1102a{args.attempt}';gpus=[2,3];out=P/'outputs'/('dtpo_'+tag);sp=E/(key+'.json')
 lock=(E/'gpu23.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if sp.exists() or out.exists():raise RuntimeError('Existing state/output: inspect rather than repeat')
 gate=json.loads((E/'CPU_GATE.json').read_text());sources=json.loads((E/'SOURCE.json').read_text());inputs=json.loads((E/'INPUT_HASHES.json').read_text())
 assert gate['pass'] and gate['source']==sources
 assert all(sha(C/n)==h for n,h in sources.items()),'Source changed'
 assert all(sha(Path(n))==h for n,h in inputs.items()),'Input/runtime changed'
 if phase=='formal':
  gate2=json.loads((E/'GPU_GATE.json').read_text());assert gate2['pass'] and gate2['source']==sources
  assert not list(E.glob('formal_*.json')),'Formal attempt already exists; review explicitly before any resume'
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows={int(l.split(',')[0]):[int(v) for v in l.split(',')[1:]] for l in gpu.splitlines()}
 if any(rows[g][0]>1024 or rows[g][1]>5 for g in gpus):raise RuntimeError('GPU occupied; do not touch owner')
 mem=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
 if mem+170*G>min(390*G,limit*.9):raise RuntimeError('Insufficient shared memory admission')
 free=os.statvfs(P).f_bavail*os.statvfs(P).f_frsize
 sapo_ck=P/'outputs/sapo_f1101a1/ckpt'
 present=sum(q.stat().st_size for q in sapo_ck.rglob('*') if q.is_file())
 sapo_reserve=max(0,103*G-present)
 required=(115 if phase=='formal' else 29)*G+sapo_reserve
 if free<required:raise RuntimeError(f'Insufficient disk: {free/G:.2f}GiB free, {required/G:.2f}GiB required including SAPO pending saves; no deletion')
 cmd=['bash',str(C/'run.sh'),tag,phase,','.join(map(str,gpus))]
 state=dict(state='starting',phase=phase,attempt=args.attempt,tag=tag,output=str(out),target_steps=1 if phase=='smoke' else 32,gpus=gpus,started=time.time(),supervisor_pid=os.getpid(),source_hashes=sources,command=cmd,admission_memory=mem,admission_free_disk=free)
 write(sp,state)
 with (E/f'{key}.log').open('x') as log,(E/f'{key}_memory.jsonl').open('x') as mf:
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(state='running',pid=child.pid);write(sp,state);peak=0;press=0
  while child.poll() is None:
   mem=int(Path('/sys/fs/cgroup/memory.current').read_text());peak=max(peak,mem);mf.write(json.dumps({'time':time.time(),'memory':mem})+'\n');mf.flush();press=press+1 if mem>390*G else 0
   if press>=3 or mem>400*G:
    state['memory_guard']={'at':time.time(),'bytes':mem};stop_owned(child.pid,tag);break
   time.sleep(5)
  child.wait(timeout=45);state.update(state='failed',exit_code=child.returncode,finished=time.time(),peak_shared_bytes=peak)
 if child.returncode==0:
  step=state['target_steps'];ck=out/f'ckpt/global_step_{step}'
  parts=['model'] if phase=='smoke' else ['model','optim','extra_state']
  files=[ck/'actor'/f'{part}_world_size_2_rank_{rank}.pt' for part in parts for rank in [0,1]]
  if phase=='formal':files.append(ck/'dataloader.pt')
  state['missing']=[str(x) for x in files if not x.is_file() or not x.stat().st_size]
  tracker=out/'ckpt/checkpoint_tracker.json';state['tracker']=json.loads(tracker.read_text()) if tracker.exists() else {}
  state['state']='completed' if not state['missing'] and state['tracker'].get('last_global_step')==step else 'failed_checkpoint'
 write(sp,state)
if __name__=='__main__':main()
