from simtk import unit


def check_stability(simulation, n_steps=1000, n_rounds=10, potential_energy_threshold=1000 * unit.kilojoule_per_mole):
    """Run simulation for n_steps, periodically checking if the potential energy exceeds a threshold.
    If the potential energy ever exceeds the threshold or becomes NaN, terminate and return False.
    
    Parameters
    ----------
    simulation : openmm.app.Simulation
        simulation whose stability we are studying
    n_steps : int, default 1000
        how many timesteps to simulate
    n_rounds : int, default 10
        how many rounds to use to run n_steps
        e.g. if n_steps is 1000, specifying n_rounds as 10 will run 10 rounds of 100 steps
    potential_energy_threshold : simtk.unit (energy)
        if the potential energy of the simulation exceeds this threshold, NaNs are nigh

    """

    for _ in range(n_rounds):
        simulation.step(round(n_steps / n_rounds))
        potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        if not (potential_energy <= potential_energy_threshold):
            return False
    return True


def stability_oracle_factory(simulation, set_initial_conditions,
                             n_steps=1000, potential_energy_threshold=1000 * unit.kilojoule_per_mole):
    """Construct a stochastic function that accepts a scalar (timestep, in femtoseconds)
    and checks whether integration at that timestep appears stable.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        simulation whose stability we are studying
    set_initial_conditions : callable
        accepts a simulation, and modifies its state (positions, velocities, box vectors) appropriately.
        may set state to a sample from equilibrium, or from some other interesting ensemble of initial conditions.
    n_steps : int
        how many timesteps to simulate
    potential_energy_threshold : simtk.unit (energy)
        if the potential energy of the simulation exceeds this threshold, NaNs are nigh

    Returns
    -------
    iterated_stability_oracle : callable
        accepts dt (float) and n_iterations (int)
    """

    def stability_oracle(dt):
        """Sample whether the simulation blows up at the given timestep dt"""

        dt *= unit.femtosecond
        if not (dt.unit.is_compatible(unit.femtosecond)):
            raise (ValueError('dt is assumed to be a float'))

        simulation.integrator.setStepSize(dt)

        set_initial_conditions(simulation)

        return check_stability(simulation, n_steps, potential_energy_threshold)

    def iterated_stability_oracle(dt, n_iterations=10):
        """Return True if stability_oracle is True n_iterations times, terminating early when possible.
        This allows to find the timestep that causes the simulation to blow up only (1/2)^n_iterations of the time
        # TODO: Rewrite in terms of failure probability, rather than number of iterations
        """
        if (n_iterations < 1) or (not isinstance(n_iterations, int)):
            raise (ValueError('n_iterations must be a positive integer'))

        for _ in range(n_iterations):
            if not stability_oracle(dt):
                return False
        return True

    return iterated_stability_oracle
