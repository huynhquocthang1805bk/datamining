"""The report's rule: a number is bold when it is better than the baseline at the precision shown."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_report import bnum, bcell, pnum, pct  # noqa: E402


def test_lower_error_is_bold():
    assert bnum(4.20, 4.30) == '**4.200**'
    assert bnum(4.40, 4.30) == '4.400'


def test_tie_on_screen_is_not_bold():
    assert bnum(5.1264, 5.1261, 3) == '5.126'


def test_higher_accuracy_is_bold():
    assert bnum(0.92, 0.91, 3, higher=True) == '**0.920**'
    assert bnum(0.90, 0.91, 3, higher=True) == '0.900'


def test_baseline_row_and_missing_values():
    assert bnum(7.516, None) == '7.516'
    assert bnum(float('nan'), 1.0) == '—'
    assert bcell([float('nan'), float('nan')], [1, 1]) == '—'


def test_composite_cells_are_bolded_per_component():
    assert bcell([5.0, 3.5], [5.1, 3.4]) == '**5.00** / 3.50'


def test_sd_parse_and_delta():
    assert bnum(7.495, 7.516, 3, sd=0.006) == '**7.495** ± 0.006'
    assert bnum(6.467, 7.516, 3, sd=0.0004) == '**6.467** ± <0.001'
    assert pnum('7.516 ± 0.031') == (7.516, 0.031)
    assert pct(6.19, 7.516) == '-17.6%'
