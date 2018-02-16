"""Here, we'll do a simple test where we vary the mass of hydrogen,
and see what this does to max stable timestep..."""

from copy import deepcopy
from pickle import dump

import numpy as np
from openmmtools.integrators import LangevinIntegrator, GHMCIntegrator
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit

from thresholds import utils, bisect, stability


def is_hydrogen(atom):
    """Check if this atom is a hydrogen"""
    return atom.element.symbol == "H"


def get_hydrogens(topology):
    """Get the indices of hydrogen atoms"""

    atom_indices = []
    for (atom_index, atom) in enumerate(topology.atoms()):
        if is_hydrogen(atom):
            atom_indices.append(atom_index)
    return atom_indices


def get_mass(system, atom_index):
    """Get mass of a single particle"""
    return system.getParticleMass(atom_index).value_in_unit(unit.amu)


def scale_particle_masses(system, atom_indices, scale_factor):
    """Multiply the masses of all atoms in `atom_indices` by `scale_factor`"""
    for atom_index in atom_indices:
        current_mass = get_mass(system, atom_index)
        system.setParticleMass(atom_index, current_mass * scale_factor)


def get_atoms_bonded_to_hydrogen(topology):
    """Get the indices of particles bonded to hydrogen

    Returns
    -------
    atom_indices : list of ints
        Will include repeated entries, if an atom is bonded to
        more than one hydrogen
    """

    atom_indices = []
    for (a, b) in topology.bonds():
        a_H, b_H = is_hydrogen(a), is_hydrogen(b)
        if a_H + b_H == 1:  # if exactly one of these is a hydrogen
            if is_hydrogen(a):
                atom_indices.append(b.index)
            else:
                atom_indices.append(a.index)
    return atom_indices


def decrement_particle_masses(system, atom_indices, decrement):
    """Reduce the masses of all atoms in `atom_indices` by `decrement`"""
    for atom_index in atom_indices:
        current_mass = get_mass(system, atom_index)
        target_mass = current_mass - decrement
        if target_mass <= 0:
            raise (RuntimeError("Trying to remove too much mass!"))
        system.setParticleMass(atom_index, target_mass)


def repartition_hydrogen_mass_amber(topology, system, scale_factor=3):
    """Scale up hydrogen mass, subtract added mass from bonded heavy atoms

    References
    ----------
    [1] Long-Time-Step Molecular Dynamics through Hydrogen Mass Repartitioning
        (Hopkins, Grand, Walker, Roitberg, 2015, JCTC) http://pubs.acs.org/doi/abs/10.1021/ct5010406
    """
    hmr_system = deepcopy(system)

    # scale hydrogen mass by 3x, keeping track of how much mass was added to each
    hydrogens = get_hydrogens(topology)
    initial_h_masses = [get_mass(system, h) for h in hydrogens]
    if len(set(initial_h_masses)) > 1:
        raise (NotImplementedError("Initial hydrogen masses aren't all equal. "
                                   "Implementation currently assumes all hydrogen masses are initially equal."))
        # TODO: Relax this assumption

    scale_particle_masses(hmr_system, hydrogens, scale_factor)
    mass_added_to_each_hydrogen = get_mass(hmr_system, hydrogens[0]) - get_mass(system, hydrogens[0])

    # for each heavy-atom-hydrogen bond, subtract that amount of mass from the heavy atom
    for heavy_atom in get_atoms_bonded_to_hydrogen(topology):
        decrement_particle_masses(hmr_system, [heavy_atom], mass_added_to_each_hydrogen)

    return hmr_system


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
        testsystem = AlanineDipeptideVacuum()
        testsystem.system = repartition_hydrogen_mass_amber(testsystem.topology, testsystem.system,
                                                            scale_factor=hydrogen_mass)
        test_sim = utils.sim_factory(testsystem)(make_integrator('V R O R V'))

        iterated_stability_oracle = stability.stability_oracle_factory(test_sim, set_initial_conditions,
                                                                       n_steps=100)


        def noisy_oracle(dt):
            return iterated_stability_oracle(dt, n_iterations=20)


        x, zs, fs = bisect.probabilistic_bisection(noisy_oracle, search_interval=(0, 10), p=0.8,
                                                   early_termination_width=0.01)

        final_beliefs[hydrogen_mass] = (x, fs[-1])

        print('measured stability threshold for hydrogen_mass={:.3f}: {:.3f}fs'.format(hydrogen_mass,
                                                                                       x[np.argmax(fs[-1])]))

        # over-write file with each new result
        with open('data/hmr_stability.pkl', 'wb') as f:
            dump(final_beliefs, f)
