from pathlib import Path
import json,hashlib,subprocess,os,time
P=Path('$PROJECT');C=P/'dtpo_vlm_20261101';E=P/'evidence/dtpo_vlm_20261101';E.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
files=list(C.glob('*.py'))+[C/'run.sh',C/'OFFICIAL_LICENSE'];source={p.name:sha(p) for p in files}
old=json.loads((P/'evidence/dapo_loo_20261031/attempt2/INPUT_HASHES.json').read_text())['files']
inputs={p:sha(Path(p)) for p in old};assert inputs==old, 'Frozen inherited input/runtime drift'
for p in ['$EASYR1_DIR/verl/workers/actor/dp_actor.py','$EASYR1_DIR/verl/workers/fsdp_workers.py','$EASYR1_DIR/verl/__init__.py']:
 inputs[p]=sha(Path(p))
import pyarrow.parquet as pq
counts={}
for path in ['$ROOT/datasets/grpo_caption_v51_4k/train.parquet','$ROOT/datasets/grpo_caption_v51_4k/val_nosft.parquet']:
 rows=pq.read_table(path).to_pylist();counts[path]=len(rows)
assert list(counts.values())==[4096,422]
env=os.environ.copy();env.pop('CAPTION_GRPO_CODE',None)
r=subprocess.run([os.sys.executable,str(C/'test_adapter.py')],capture_output=True,text=True,env=env);(E/'cpu_tests.log').write_text(r.stdout+'\n'+r.stderr);assert r.returncode==0
for phase in ['smoke','formal']:
 env.update(CAPTION_GRPO_CODE=str(C),CAPTION_LAZY_MM='1',CAPTION_MM_PREFETCH='2',DTPO_PHASE=phase)
 r=subprocess.run([os.sys.executable,'-c','import verl; from verl.workers.actor.dp_actor import DataParallelPPOActor as A; from verl.trainer.config import PPOConfig; from verl.trainer.ray_trainer import RayPPOTrainer; c=PPOConfig(); c.algorithm.adv_estimator="grpo"; c.algorithm.disable_kl=True; c.algorithm.kl_coef=0; c.worker.rollout.n=8; c.data.rollout_batch_size=32; c.worker.actor.global_batch_size=32; c.worker.actor.ppo_epochs=1; c.trainer.max_steps=1; t=RayPPOTrainer(c,None,None,[],[],{},None); assert A._dtpo_installed and not t.use_critic and not t.use_reference_policy and t.training_steps==1; print("INSTALL_AND_CONSTRUCTOR_PASS")'],cwd='$EASYR1_DIR',env=env,capture_output=True,text=True)
 (E/f'install_{phase}.log').write_text(r.stdout+'\n'+r.stderr);assert r.returncode==0,(phase,r.stderr[-3000:])
(E/'SOURCE.json').write_text(json.dumps(source,indent=2));(E/'INPUT_HASHES.json').write_text(json.dumps(inputs,indent=2))
receipt={'pass':True,'cpu_tests':5,'time':time.time(),'official_commit':'1b53728f745396693903fde5a6e4b98c149448ea','source':source,'inputs':inputs,'pool_counts':counts,'GPU_verified':False}
(E/'CPU_GATE.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
