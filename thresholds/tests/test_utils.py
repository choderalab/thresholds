import numpy as np
from openmmtools.integrators import LangevinIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit
from simtk.openmm import app

from thresholds import utils


def test_clone_state():
    # construct source system
    construct_sim = utils.sim_factory(AlanineDipeptideVacuum())
    sim_a = construct_sim(LangevinIntegrator())
    sim_b = construct_sim(LangevinIntegrator())
    sim_a.step(50)
    sim_b.step(50)

    utils.clone_state(sim_a, sim_b)
    state_a = sim_a.context.getState(getPositions=True, getVelocities=True)
    state_b = sim_b.context.getState(getPositions=True, getVelocities=True)

    # check that the positions were cloned
    assert (np.isclose(state_a.getPositions(asNumpy=True), state_b.getPositions(asNumpy=True)).all())

    # check that the velocities were cloned
    assert (np.isclose(state_a.getVelocities(asNumpy=True), state_b.getVelocities(asNumpy=True)).all())

    # TODO: construct a test where the box vectors have changed...


def test_measure_shadow_work():
    # test that it's consistent with openmmtools
    construct_sim = utils.sim_factory(AlanineDipeptideVacuum(constraints=None))
    sim = construct_sim(LangevinIntegrator(
        splitting='O V R V O', measure_heat=True, measure_shadow_work=True, timestep=2.0 * unit.femtoseconds))
    sim.runForClockTime(1 * unit.seconds)

    w_shads_here = []
    w_shads_openmmtools = []
    for _ in range(10):
        w_shad_prev = sim.integrator.get_shadow_work(dimensionless=True)
        w_shads_here.append(utils.measure_shadow_work(sim, 10))
        w_shad_new = sim.integrator.get_shadow_work(dimensionless=True)
        w_shads_openmmtools.append(w_shad_new - w_shad_prev)
    assert (np.isclose(w_shads_here, w_shads_openmmtools).all())


def test_sim_factory():
    # check sim_factory returns a function when passed a testsystem
    construct_sim = utils.sim_factory(AlanineDipeptideVacuum())
    assert (callable(construct_sim))

    # check that the resulting function returns an app.Simulation when passed an integrator
    sim = construct_sim(LangevinIntegrator())
    assert (isinstance(sim, app.Simulation))
