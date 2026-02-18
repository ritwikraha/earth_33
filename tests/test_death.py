"""Test: death condition triggers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_io.config import PhysiologyConfig
from config_io.schema import CauseOfDeath
from sim.physiology import OrganismState, check_death


def _make_state(**kwargs) -> OrganismState:
    defaults = dict(x=0, y=0, hydration=50, energy=50, core_temp_c=37.0,
                    fatigue=20, injury=10, infection=5)
    defaults.update(kwargs)
    return OrganismState(**defaults)


def test_death_dehydration():
    cfg = PhysiologyConfig()
    s = _make_state(hydration=0.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.DEHYDRATION
    assert s.alive is False


def test_death_starvation():
    cfg = PhysiologyConfig()
    s = _make_state(energy=0.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.STARVATION


def test_death_hypothermia():
    cfg = PhysiologyConfig()
    s = _make_state(core_temp_c=29.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.HYPOTHERMIA


def test_death_hyperthermia():
    cfg = PhysiologyConfig()
    s = _make_state(core_temp_c=43.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.HYPERTHERMIA


def test_death_trauma():
    cfg = PhysiologyConfig()
    s = _make_state(injury=100.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.TRAUMA


def test_death_infection():
    cfg = PhysiologyConfig()
    s = _make_state(infection=100.0)
    assert check_death(s, cfg) is True
    assert s.cause_of_death == CauseOfDeath.INFECTION


def test_alive():
    cfg = PhysiologyConfig()
    s = _make_state()
    assert check_death(s, cfg) is False
    assert s.alive is True
    assert s.cause_of_death == CauseOfDeath.ALIVE


if __name__ == "__main__":
    test_death_dehydration()
    test_death_starvation()
    test_death_hypothermia()
    test_death_hyperthermia()
    test_death_trauma()
    test_death_infection()
    test_alive()
    print("PASS: all death condition tests")
