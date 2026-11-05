"""CPU regressions for the 10-27 review. GPU/FSDP equivalence is checked separately by smoke runs."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[2] / 'publications/vlm-caption-trajectories/code/reference'))
import decision_loss as DL
import caption_contracts as CC
import signal_check as SC


def test_executed_crop_without_admitted_receipt(tmp_path):
    from PIL import Image
    from caption_env import CropEnv
    path = tmp_path / 'image.png'
    Image.new('RGB', (448, 448), 'red').save(path)
    env = CropEnv(str(path))
    receipt = env.step({'kind': 'grounding', 'bbox': [0.1, 0.1, 0.5, 0.5], 'source': 'original_image', 'step': 1})
    assert receipt and env.crops == 1
    # The caller rejects this receipt for insufficient context. Only the original is in its ledger.
    recipe = CC.view_recipe(str(path), env.boxes[1:], 1, 1024, 589824)
    assert recipe['boxes'] == []
    assert env.crops == 1  # cost remains executed crop count
    assert CC.view_recipe(str(path), env.boxes[1:], 2, 1024, 589824)['boxes'] == [list(env.boxes[1])]
    with pytest.raises(ValueError):
        CC.view_recipe(str(path), env.boxes[1:], 3, 1024, 589824)


@pytest.mark.parametrize('beta', ['0', '0.00390625'])
def test_four_update_hook_routes_once_after_reordering(monkeypatch, beta):
    calls = []
    class Actor:
        def __init__(self):
            self.config = SimpleNamespace(ppo_epochs=1, global_batch_size_per_device=4)
            self.actor_module = object()
            self._caption_expected_h0 = 32
            self.steps = 0
        def update_policy(self, data):
            for m in range(0, len(data.non_tensor_batch['caption_aux']), 4):
                self.current_rows = data.non_tensor_batch['caption_aux'][m:m+4]
                self._optimizer_step()
            return {}
        def _optimizer_step(self):
            self.steps += 1
    module = ModuleType('verl.workers.actor.dp_actor')
    module.DataParallelPPOActor = Actor
    monkeypatch.setitem(sys.modules, 'verl.workers.actor.dp_actor', module)
    monkeypatch.setenv('CAPTION_DL_BETA', beta)
    monkeypatch.setenv('CAPTION_DL_H0', '32')
    monkeypatch.setattr(DL, '_grad_norm', lambda model: 0.)
    def backward(actor, real, dummy, temperature, b, h0):
        expected, _ = DL.split_records(actor.current_rows)
        assert real == expected and dummy == {'dummy': True}
        assert temperature == 0.7 and h0 == 32
        calls.append([r['id'] for r in real])
        return {'loss': 0., 'p': 0., 'mass': 0., 'absd': 0., 'n': len(real), 'agree': 0, 'gz': 0.}
    monkeypatch.setattr(DL, '_aux_backward', backward)
    DL.install_actor_patch()
    # Uneven selected states per minibatch, including one empty minibatch, after a rank's row reorder.
    records = [None] * 16
    for i in (1, 2, 9, 11, 14):
        records[i] = {'real': {'id': i}, 'dummy': None}
    records[0] = {'real': None, 'dummy': {'dummy': True}}
    data = SimpleNamespace(non_tensor_batch={'caption_aux': np.array(records, dtype=object)}, meta_info={'temperature': .7})
    actor = Actor()
    actor.update_policy(data)
    assert actor.steps == 4
    assert calls == ([[1, 2], [], [9, 11], [14]] if float(beta) else [])
    assert actor._dl_pending is None
    if float(beta):
        monkeypatch.setenv('CAPTION_DL_H0', '128')
        with pytest.raises(RuntimeError, match='CAPTION_DL_H0'):
            actor.update_policy(data)


def test_all_correct_groups_can_have_rloo_signal():
    rows = [dict(uid='q', correct=1, reward=r, crops=n, terminal='answer', first_kind='grounding',
                 gen_tokens=10, proc_tokens=100) for r, n in [(0.99, 1), (.97, 3)]]
    out = SC.groups(rows)
    assert out['groups'] == {'all_correct': 1}
    assert out['nonzero_reward_variance_groups'] == 1
    assert out['rloo_abs_adv_mean'] == pytest.approx(.02)
    assert out['first_crops_mean'] == 1 and out['later_crops_mean'] == 1
