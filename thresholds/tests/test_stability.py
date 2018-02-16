import pytest
from openmmtools.integrators import LangevinIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit

from thresholds import utils, stability

testsystem = AlanineDipeptideVacuum()
construct_sim = utils.sim_factory(testsystem)

tiny_dt = 0.1 * unit.femtoseconds
stable_sim = construct_sim(LangevinIntegrator(timestep=tiny_dt))

huge_dt = 100 * unit.femtoseconds
unstable_sim = construct_sim(LangevinIntegrator(timestep=huge_dt))


def test_check_stability():
    # check that stable simulation is labeled stable
    assert (stability.check_stability(stable_sim, n_steps=100))

    # check that unstable simulation is labeled unstable
    assert (not stability.check_stability(unstable_sim, n_steps=100))


def test_stability_oracle_factory():
    def set_initial_conditions(sim):
        sim.context.setPositions(testsystem.positions)
        sim.context.setVelocitiesToTemperature(298 * unit.kelvin)

    iterated_stability_oracle = stability.stability_oracle_factory(stable_sim, set_initial_conditions, n_steps=100)

    # check stable using default n_iterations
    assert (iterated_stability_oracle(tiny_dt / unit.femtoseconds))

    # check stable when n_iterations is set
    assert (iterated_stability_oracle(tiny_dt / unit.femtoseconds, n_iterations=1))

    # check unstable using default n_iterations
    assert (not iterated_stability_oracle(huge_dt / unit.femtoseconds))

    # check unstable when n_iterations is set
    assert (not iterated_stability_oracle(huge_dt / unit.femtoseconds, n_iterations=1))

    # check that it enforces receiving a float
    with pytest.raises(ValueError):
        iterated_stability_oracle(huge_dt)

    # check that n_iterations < 1 raises a ValueError
    with pytest.raises(ValueError):
        iterated_stability_oracle(huge_dt / unit.femtoseconds, n_iterations=0)
