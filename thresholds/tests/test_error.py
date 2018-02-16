import numpy as np
import pytest
from openmmtools.integrators import LangevinIntegrator, GHMCIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit

from thresholds import utils, error


def test_get_a_paired_kldiv_sample():
    # just make sure it returns floats?

    testsystem = AlanineDipeptideVacuum(constraints=None)
    construct_sim = utils.sim_factory(testsystem)

    equilibrium_sim = construct_sim(GHMCIntegrator(timestep=0.25 * unit.femtosecond))
    equilibrium_sim.minimizeEnergy()
    equilibrium_sim.step(1000)

    reference_sim = construct_sim(
        LangevinIntegrator(splitting='O V R V O', measure_shadow_work=True, measure_heat=True,
                           timestep=0.01 * unit.femtosecond))
    test_sim = construct_sim(
        LangevinIntegrator(splitting='O V R V O', measure_shadow_work=True, measure_heat=True,
                           timestep=3 * unit.femtosecond))

    kldiv_reference, kldiv_test = error.get_a_paired_kldiv_sample(equilibrium_sim, reference_sim, test_sim,
                                                                  protocol_length=100)

    assert (isinstance(kldiv_reference, float))
    assert (isinstance(kldiv_test, float))

    # more stringent test: check that median of reference here is lower than median of test_sim
    reference_samples, test_samples = [], []
    for _ in range(100):
        equilibrium_sim.step(100)
        kldiv_reference, kldiv_test = error.get_a_paired_kldiv_sample(equilibrium_sim, reference_sim, test_sim,
                                                                      protocol_length=100)
        reference_samples.append(kldiv_reference)
        test_samples.append(kldiv_test)
    assert (np.median(reference_samples) <= np.median(test_samples))

    # check that recycle_v works
    _ = error.get_a_paired_kldiv_sample(equilibrium_sim, reference_sim, test_sim,
                                                                  protocol_length=100, recycle_v=True)

    # check that it doesn't fail silently when reference_sim and test_sim are at different temperatures
    with pytest.raises(ValueError):
        test_sim.integrator.setTemperature(200 * unit.kelvin)
        error.get_a_paired_kldiv_sample(equilibrium_sim, reference_sim, test_sim)
