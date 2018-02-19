"""Here, we'll do a simple test where we vary the mass of hydrogen,
and see what this does to max stable timestep..."""


from pickle import dump

import numpy as np
from openmmtools.integrators import LangevinIntegrator, GHMCIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit

from thresholds import utils, bisect, stability

if __name__ == '__main__':
    testsystem = AlanineDipeptideVacuum()


    def make_integrator(splitting):
        return LangevinIntegrator(splitting=splitting, timestep=2 * unit.femtoseconds,
                                  measure_shadow_work=False, measure_heat=False)


    construct_sim = utils.sim_factory(testsystem)
    equilibrium_sim = construct_sim(GHMCIntegrator(timestep=0.5 * unit.femtoseconds))
    equilibrium_sim.context.setVelocitiesToTemperature(equilibrium_sim.integrator.getTemperature())
    equilibrium_sim.step(1000)


    def set_initial_conditions(sim):
        equilibrium_sim.step(100)
        utils.clone_state(equilibrium_sim, sim)


    final_beliefs = {}

    hmr_range = np.linspace(1, 4)

    for hydrogen_mass in hmr_range:
        testsystem = AlanineDipeptideVacuum(hydrogenMass=hydrogen_mass * unit.atom_mass_units)
        test_sim = utils.sim_factory(testsystem)(make_integrator('V R O R V'))

        iterated_stability_oracle = stability.stability_oracle_factory(test_sim, set_initial_conditions,
                                                                       n_steps=1000)


        def noisy_oracle(dt):
            return iterated_stability_oracle(dt, n_iterations=20)


        x, zs, fs = bisect.probabilistic_bisection(noisy_oracle, search_interval=(0, 10), p=0.8,
                                                   early_termination_width=0.001)

        final_beliefs[hydrogen_mass] = (x, fs[-1])

        print('measured stability threshold for hydrogen_mass={:.3f} a.m.u.: {:.3f}fs'.format(hydrogen_mass,
                                                                                       x[np.argmax(fs[-1])]))

        # over-write file with each new result
        with open('data/hmr_stability.pkl', 'wb') as f:
            dump(final_beliefs, f)
