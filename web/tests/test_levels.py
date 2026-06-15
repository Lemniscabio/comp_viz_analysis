import math
from web.worker.levels import normalized_delta_e, level_times


def test_normalized_divides_by_max():
    assert normalized_delta_e([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalized_all_zero_safe():
    assert normalized_delta_e([0.0, 0.0]) == [0.0, 0.0]


def test_level_times_first_crossing_matches_main_logic():
    # grand ΔE rising 0->10; normalized = [0,.2,.5,.9,.95,1.0]
    grand = [0.0, 2.0, 5.0, 9.0, 9.5, 10.0]
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = level_times(t, grand, levels=(0.90, 0.95, 0.99))
    assert out[0.90] == 3.0   # first idx where norm>=0.90
    assert out[0.95] == 4.0
    assert out[0.99] == 5.0


def test_level_times_not_reached_is_none():
    grand = [0.0, 1.0, 2.0]   # normalized maxes at 1.0 only at last; 0.99 reached, but test a never-case
    t = [0.0, 1.0, 2.0]
    out = level_times(t, grand, levels=(0.99,))
    # norm = [0,0.5,1.0] -> 0.99 first reached at idx 2
    assert out[0.99] == 2.0
    # a level above the curve's max-normalized (impossible >1) -> None
    out2 = level_times(t, grand, levels=(1.01,))
    assert out2[1.01] is None
