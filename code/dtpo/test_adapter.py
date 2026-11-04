import unittest,tempfile,os,types
import torch,numpy as np
from tensordict import TensorDict
from dtpo_adapter import turn_mask,partitions,token_loss,compute_advantage
from dtpo_official import compute_policy_loss

class Tests(unittest.TestCase):
 def test_multiturn_and_receipt(self):
  m=torch.tensor([[1,1,0,0,1,0,1,1,0],[1,1,1,0,0,0,0,0,0]])
  t=turn_mask(m);self.assertEqual(t.tolist(),[[1,1,0,0,2,0,3,3,0],[1,1,1,0,0,0,0,0,0]])
  a,b=partitions(t,m);self.assertEqual(int(a.sum()),3);self.assertEqual(int(b.sum()),5);self.assertTrue(torch.equal(a|b,m.bool()));self.assertFalse((a&b).any())
 def test_global_micro_aggregation(self):
  # Partition per-rank sums and denominators; gradient averaging must match unsplit official loss.
  m=torch.tensor([[1.,1,0,1,1],[1,0,1,1,1],[1,1,1,1,1]])
  tm=turn_mask(m);a,b=partitions(tm,m);adv=torch.tensor([[1.,2,0,3,4],[-1,0,-2,-3,-4],[2,2,2,2,2]])
  lp=torch.zeros_like(m,requires_grad=True);old=torch.zeros_like(m)
  actual=compute_policy_loss(old,lp,adv,m,cliprange_low=.2,cliprange_high=.24,turn_mask=tm,control_level='tool_level',control_alpha=.5)[0]
  tl=token_loss(old,lp,adv);micro=sum((.5*(tl[i]*a[i]).sum()/a.sum().clamp_min(1)+(tl[i]*b[i]).sum()/b.sum()) for i in range(3))
  torch.testing.assert_close(actual,micro);g1=torch.autograd.grad(actual,lp,retain_graph=True)[0];g2=torch.autograd.grad(micro,lp)[0];torch.testing.assert_close(g1,g2)
 def test_reward_and_whole_groups(self):
  with tempfile.TemporaryDirectory() as p:
   os.environ['DTPO_OUTPUT']=p
   mask=torch.tensor([[1,1,0,1,0,1]]*8);r=torch.zeros(8,6);r[:3,-1]=1
   b=TensorDict({'response_mask':mask,'token_level_rewards':r},batch_size=[8])
   d=types.SimpleNamespace(batch=b,non_tensor_batch={'uid':np.array(['q']*8,dtype=object),'multi_modal_data':np.array([{} for _ in range(8)],dtype=object)})
   compute_advantage(d);a=b['advantages'];ys=r.sum(-1);z=(ys-ys.mean())/(ys.std()+1e-6)
   torch.testing.assert_close(a,z[:,None]*mask);self.assertTrue(torch.equal(r,b['token_level_rewards']))
 def test_zero_tool_gradient(self):
  m=torch.ones(2,4);t=turn_mask(m);it,ia=partitions(t,m);lp=torch.zeros(2,4,requires_grad=True)
  loss=(token_loss(lp.detach(),lp,m)*ia).sum()/ia.sum();loss.backward();self.assertTrue(torch.isfinite(lp.grad).all());self.assertEqual(int(it.sum()),0)
 def test_all_wrong_no_signal(self):
  from dtpo_official import compute_DTPO_advantage
  z=torch.zeros(8,4);m=torch.ones_like(z);a,*_=compute_DTPO_advantage(z,z,m,m,['x']*8,lambda_coeff=.3);self.assertEqual(float(a.abs().sum()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
