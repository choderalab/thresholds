"""Here, we'll enumerate a bunch of multi-timestep (MTS) schemes and see what they do to the stability threshold"""

import itertools
from pickle import dump

import numpy as np
from openmmtools.integrators import LangevinIntegrator, GHMCIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit
from tqdm import tqdm

from thresholds import utils, bisect, stability


def generate_sequential_BAOAB_string(force_group_list):
    """Generate BAOAB-like schemes that break up the "V R" step
    into multiple sequential updates

    >>> generate_sequential_BAOAB_string(force_group_list=(0,1,2))
     ... "V0 R V1 R V2 R O R V2 R V1 R V0"
    """

    VR = []
    for i in force_group_list:
        VR.append("V{}".format(i))
        VR.append("R")
    return " ".join(VR + ["O"] + VR[::-1])


def generate_all_BAOAB_permutation_strings(n_force_groups):
    """Generate all of the permutations of range(n_force_groups)"""
    return [(perm, generate_sequential_BAOAB_string(perm)) for perm in
            itertools.permutations(range(n_force_groups))]


if __name__ == '__main__':
    testsystem = AlanineDipeptideVacuum()


    def make_integrator(splitting):
        return LangevinIntegrator(splitting=splitting, timestep=2 * unit.femtoseconds,
                                  measure_shadow_work=False, measure_heat=False)


    for i, force in enumerate(testsystem.system.getForces()):
        force.setForceGroup(i)
    n_force_groups = testsystem.system.getNumForces()

    construct_sim = utils.sim_factory(testsystem)
    equilibrium_sim = construct_sim(GHMCIntegrator(timestep=0.5 * unit.femtoseconds))
    equilibrium_sim.context.setVelocitiesToTemperature(equilibrium_sim.integrator.getTemperature())
    equilibrium_sim.step(1000)


    def set_initial_conditions(sim):
        equilibrium_sim.step(100)
        utils.clone_state(equilibrium_sim, sim)


    final_beliefs = {}

    for (perm, splitting) in tqdm(utils.generate_all_BAOAB_permutation_strings(n_force_groups)):
        test_sim = construct_sim(make_integrator(splitting))
        iterated_stability_oracle = stability.stability_oracle_factory(test_sim, set_initial_conditions,
                                                                       n_steps=100)


        def noisy_oracle(dt):
            return iterated_stability_oracle(dt, n_iterations=20)


        x, zs, fs = bisect.probabilistic_bisection(noisy_oracle, search_interval=(0, 30), p=0.8,
                                                   early_termination_width=0.01)

        final_beliefs[(perm, splitting)] = (x, fs)

        condensed_scheme = "".join(splitting.split())
        print('measured stability threshold for {}: {:.3f}fs'.format(condensed_scheme, x[np.argmax(fs[-1])]))

        # over-write file with each new result
        with open('data/mts_stability.pkl', 'wb') as f:
            dump(final_beliefs, f)
