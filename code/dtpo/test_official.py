import unittest
import torch
from dtpo_official import compute_DTPO_advantage, compute_policy_loss

class OfficialTests(unittest.TestCase):
    def adv(self, turns=None, mask=None, tool=None, outcome=None):
        turns=torch.tensor([[1,1,2,2],[1,1,2,2]]) if turns is None else turns
        mask=torch.ones(2,4) if mask is None else mask
        tool=torch.tensor([[0.,0,0,1],[0,0,0,0]]) if tool is None else tool
        outcome=torch.tensor([[0.,0,0,0],[0,0,0,1]]) if outcome is None else outcome
        return compute_DTPO_advantage(tool,outcome,turns,mask,['q','q'],lambda_coeff=.3)
    def test_separate_advantage(self):
        a,_,t,o=self.adv();torch.testing.assert_close(a[:,:2],(o+.3*t)[:,None].expand(-1,2));torch.testing.assert_close(a[:,2:],o[:,None].expand(-1,2))
    def test_sample_std(self):
        _,_,t,o=self.adv();z=1/(2**.5+2e-6)
        torch.testing.assert_close(t,torch.tensor([z,-z]));torch.testing.assert_close(o,-t)
    def test_mask_zero(self):
        a,*_=self.adv(mask=torch.tensor([[1.,0,1,0],[1,0,1,0]]));self.assertEqual(a[:,[1,3]].abs().sum(),0)
    def test_uniform_rewards(self):
        a,*_=self.adv(tool=torch.zeros(2,4),outcome=torch.zeros(2,4));self.assertEqual(a.abs().sum(),0)
    def test_turn_three_not_supported(self):
        a,*_=self.adv(turns=torch.tensor([[1,2,3,3],[1,2,3,3]]));self.assertEqual(a[:,2:].abs().sum(),0)
    def test_direct_has_combined_adv(self):
        a,_,t,o=self.adv(turns=torch.ones(2,4));torch.testing.assert_close(a,(o+.3*t)[:,None].expand(-1,4))
    def test_split_loss_and_tool_receipt_gradient(self):
        lp=torch.zeros(2,5,requires_grad=True);old=torch.zeros_like(lp);adv=torch.tensor([[1.,9,2,2,2],[-1,9,3,3,3]])
        mask=torch.tensor([[1.,0,1,1,1],[1,0,1,1,1]]);turn=torch.tensor([[1,0,2,2,2],[1,0,2,2,2]])
        loss,*_=compute_policy_loss(old,lp,adv,mask,cliprange_low=.2,cliprange_high=.24,turn_mask=turn,control_level='tool_level',control_alpha=.5)
        expected=-.5*(adv[:,0].sum()/2)-adv[:,2:].sum()/6
        torch.testing.assert_close(loss,expected);loss.backward();self.assertEqual(lp.grad[:,1].abs().sum(),0)
        torch.testing.assert_close(lp.grad[:,0],-.5*adv[:,0]/2);torch.testing.assert_close(lp.grad[:,2:],-adv[:,2:]/6)
    def test_all_direct_no_tool_denominator(self):
        z=torch.zeros(2,3,requires_grad=True);mask=torch.ones_like(z)
        loss,*_=compute_policy_loss(z.detach(),z,mask,mask,cliprange=.2,turn_mask=mask,control_level='tool_level',control_alpha=.5)
        torch.testing.assert_close(loss,torch.tensor(-1.));loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_clipping(self):
        lp=torch.tensor([[1.,-1.]],requires_grad=True);old=torch.zeros_like(lp);adv=torch.tensor([[1.,-1.]])
        loss,*_=compute_policy_loss(old,lp,adv,torch.ones_like(lp),cliprange_low=.2,cliprange_high=.24)
        torch.testing.assert_close(loss,torch.tensor((-.24-.2)/2));loss.backward();self.assertEqual(lp.grad.abs().sum(),0)

if __name__=='__main__':unittest.main(verbosity=2)
