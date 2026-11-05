import os,tempfile,unittest,types,torch
from tensordict import TensorDict
from sapo_adapter import add_boundaries,capture_forward,mask_logits,compute_advantage,TOKENS
from sapo_official import compute_arq_gae_targets,compute_arq_value_loss

class Tests(unittest.TestCase):
 def batch(self):
  # Different left padding; two actions separated by a 3-token observation.
  mask=torch.tensor([[1,1,0,0,0,1,1,1,0],[1,1,1,0,0,0,0,0,0]])
  att=torch.tensor([[0,1,1,1]+[1]*8+[0],[1,1,1,1]+[1]*3+[0]*6])
  b=TensorDict({'input_ids':torch.arange(26).reshape(2,13),'responses':torch.arange(18).reshape(2,9),'response_mask':mask,'attention_mask':att},batch_size=[2]);add_boundaries(b);return b
 def test_boundaries(self):
  b=self.batch();self.assertEqual(b['sapo_vpos'][0,:2].tolist(),[3,8]);self.assertEqual(b['sapo_qpos'][0,:2].tolist(),[5,11]);self.assertEqual(b['sapo_turn_mask'].sum(-1).tolist(),[2,1])
 def test_gae_sarsa_manual(self):
  r=torch.tensor([0.,1.,0.]);v=torch.tensor([.2,.5,.1]);q=torch.tensor([.3,.6,.4]);x=compute_arq_gae_targets(r,torch.tensor([False,True,True]),v,q,torch.ones(3,1),['a','a','b'],torch.tensor([0,1,0]),.95,.95,normalize=False,q_target_mode='sarsa_td0')
  self.assertTrue(torch.allclose(x['gae_advantages'],torch.tensor([.275+.95*.95*.5,.5,-.1])))
  self.assertTrue(torch.allclose(x['q_targets'],torch.tensor([.57,1.,0.])))
 def test_turn_normalization(self):
  x=compute_arq_gae_targets(torch.tensor([0.,1.,0.]),torch.ones(3,dtype=torch.bool),torch.zeros(3),torch.zeros(3),torch.tensor([[1,0,0],[1,1,1],[1,1,0]]),['a','b','c'],torch.zeros(3,dtype=torch.long),.95,.95,q_target_mode='sarsa_td0')
  self.assertAlmostEqual(float(x['turn_advantages'].mean()),0.,places=6)
  self.assertEqual(float(x['advantages'][0,1]),0.)
 def test_clipped_value_and_gradient(self):
  v=torch.tensor([.5,2.],requires_grad=True);loss,_=compute_arq_value_loss(v,torch.tensor([0.,0.]),torch.tensor([1.,0.]),.2)
  self.assertAlmostEqual(float(loss),(.64+4)/4,places=6);loss.backward();self.assertTrue(torch.isfinite(v.grad).all())
 def test_zero_signal(self):
  x=compute_arq_gae_targets(torch.zeros(3),torch.ones(3,dtype=torch.bool),torch.zeros(3),torch.zeros(3),torch.ones(3,2),['a','b','c'],torch.zeros(3,dtype=torch.long),.95,.95,q_target_mode='sarsa_td0')
  self.assertTrue(torch.equal(x['advantages'],torch.zeros(3,2)))
 def test_mask_and_shared_head_gradients(self):
  # Small hidden dim, true vocabulary coordinates. Policy excludes both; V/Q uses the same LM head.
  head=torch.nn.Linear(4,151670,bias=False);h=torch.randn(2,4,requires_grad=True);z=head(h);values=z[:,TOKENS[0]]-z[:,TOKENS[1]];mask_logits(z)
  pg=-z.log_softmax(-1)[:,10].mean();gp=torch.autograd.grad(pg,head.weight,retain_graph=True)[0]
  self.assertEqual(float(gp[list(TOKENS)].abs().sum()),0.)
  value_loss=(values-1).square().mean();gv=torch.autograd.grad(value_loss,head.weight,retain_graph=True)[0]
  self.assertGreater(float(gv[list(TOKENS)].abs().sum()),0.);(pg+.1*value_loss).backward();self.assertTrue(torch.isfinite(h.grad).all())
 def test_capture_padded_unpadded_equal(self):
  b=self.batch();B,S=b['input_ids'].shape;h=torch.randn(B,S,4);head=torch.nn.Linear(4,151670,bias=False)
  class Root:
   def get_output_embeddings(self):return head
  for free in [False,True]:
   actor=types.SimpleNamespace(config=types.SimpleNamespace(ulysses_size=1,padding_free=free),actor_module=Root())
   def orig(self,m,temp):
    z=head(h[m['attention_mask'].bool()].unsqueeze(0) if free else h);return z.sum()*0
   capture_forward(orig)(actor,b,1.)
   if not free:expected=actor._sapo_last.detach()
   else:self.assertTrue(torch.allclose(actor._sapo_last,expected,atol=1e-6))
 def test_loss_aggregation(self):
  # Unequal token/turn counts across ranks, FSDP averages gradients across ranks.
  token_sums=[3.,8.];turn_sums=[2.,7.];nt=[5,10];nv=[1,4]
  distributed=sum(2*(token_sums[i]/sum(nt)+.1*turn_sums[i]/sum(nv)) for i in range(2))/2
  self.assertAlmostEqual(distributed,sum(token_sums)/sum(nt)+.1*sum(turn_sums)/sum(nv))
 def test_driver_targets_observation_mask(self):
  import numpy as np
  b=self.batch();m=b['sapo_turn_mask'];b['sapo_old_v']=torch.zeros(2,9);b['sapo_old_q']=torch.zeros(2,9);b['token_level_rewards']=torch.zeros(2,9);b['token_level_rewards'][0,7]=1
  data=types.SimpleNamespace(batch=b,non_tensor_batch={'multi_modal_data':np.array([{'caption_qid':'a'},{'caption_qid':'b'}],dtype=object)})
  with tempfile.TemporaryDirectory() as p:
   os.environ['SAPO_OUTPUT']=p;compute_advantage(data,gamma=.95,lam=.95)
  self.assertEqual(float(b['advantages'][~b['response_mask'].bool()].abs().sum()),0.)
  self.assertFalse(b['sapo_v_target'].requires_grad)
if __name__=='__main__':unittest.main(verbosity=2)
