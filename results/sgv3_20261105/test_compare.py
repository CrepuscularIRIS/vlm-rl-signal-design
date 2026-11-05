"""固定评分与配对的小型回归检查。"""
import importlib.util
from pathlib import Path


def test_pair_counts():
    spec = importlib.util.spec_from_file_location('compare', Path(__file__).with_name('compare.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert module.four_cells(a, b) == {'both_correct': 1, 'new_only': 1, 'control_only': 1, 'both_wrong': 1, 'net': 0}
    assert module.four_cells([1], [0])['net'] == 1
    try:
        module.four_cells([1], [])
    except ValueError:
        pass
    else:
        raise AssertionError('未拒绝长度不等')
