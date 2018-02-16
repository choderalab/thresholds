from simtk import openmm as mm
from simtk.openmm import app


def clone_state(source_sim, target_sim):
    """Clone the state of source_sim to target_sim, where state = (positions, box-vectors, velocities)"""
    source_state = source_sim.context.getState(getPositions=True, getVelocities=True)

    # get positions, velocities, box vectors of source_sim
    pos = source_state.getPositions(asNumpy=True)
    vel = source_state.getVelocities(asNumpy=True)
    box = source_state.getPeriodicBoxVectors(asNumpy=True)

    # set positions, velocities box vectors of target_sim
    target_sim.context.setPositions(pos)
    target_sim.context.setVelocities(vel)
    target_sim.context.setPeriodicBoxVectors(*box)


def measure_shadow_work(sim, n_steps):
    """Run the sim for n_steps and return the accumulated shadow work."""
    w0 = sim.integrator.get_shadow_work(dimensionless=True)
    sim.step(n_steps)
    w1 = sim.integrator.get_shadow_work(dimensionless=True)
    return w1 - w0


def sim_factory(testsystem, platform=None):
    """Convenience method for constructing multiple simulations using the same openmmtools testsystem
    but different integrators"""

    if not isinstance(platform, mm.Platform):
        platform = mm.Platform.getPlatformByName("Reference")

    def construct_sim(integrator):
        sim = app.Simulation(testsystem.topology, testsystem.system, integrator,
                             platform=platform)
        sim.context.setPositions(testsystem.positions)
        sim.context.setVelocitiesToTemperature(integrator.getTemperature())
        return sim

    return construct_sim
