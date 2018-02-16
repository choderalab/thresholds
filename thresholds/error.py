from .utils import clone_state, measure_shadow_work


def get_a_paired_kldiv_sample(equilibrium_sim, reference_sim, test_sim,
                              protocol_length=1000, recycle_v=False):
    """Does test_sim introduce more or less configuration-space error than reference_sim?

    Parameters
    ----------
        equilibrium_sim : openmm.app.Simulation
            simulation whose state is assumed to be at equilibrium
        reference_sim : openmm.app.Simulation
            simulation that introduces a *tolerable* amount of configuration-space error
            (for example, this might be a standard Langevin integrator at dt=2fs)
        test_sim : openmm.app.Simulation
            simulation that introduces an unknown amount of configuration-space error
            (for example, this might be a new multiple-timestep scheme at a timestep of 100fs)
        protocol_length : int
            number of timesteps to simulate per protocol sample
        recycle_v : bool
            if recycle_v=True, use the same equilibrium velocity sample to prepare reference_sim and equilibrium_sim
            in nonequilibrium state omega(x,v) = rho(x) pi(v | x)

    Returns
    -------
        kldiv_reference : float
            1-sample estimate of the configuration-space KL-divergence
        kldiv_test : float
            1-sample estimate of the configuration-space KL-divergence, correlated with kldiv_reference

    Notes
    -----
        Algorithm:
        1. set state of test_sim and reference_sim to the current state of equilibrium_sim
            * assume that before this is called, the state of equilibrium_sim has been set to a sample from equilibrium
              e.g. by propagating for a while, or by drawing uniformly from a sample cache or something.
              --> will want to do an experiment where we use a sequence of correlated samples,
                and compare to the ideal case when we can draw independently from the target...
        2. sample from work distribution p(w | pi, Lambda)
            * w_pi_reference <-- propagate reference_sim for protocol_length steps, record shadow work
            * w_pi_test <-- propagate test_sim for protocol_length steps, record shadow work
        3. prepare reference_sim, test_sim in state omega(x,v) = rho(x) pi(v | x)
            * randomize velocities in reference_sim
            * randomize velocities in test_sim
                * If recycle_v=True, then set test_sim to the same v as reference_sim just sampled
        4. sample from work distributions p(w | omega, Lambda)
            * w_omega_reference <-- propagate reference_sim for protocol_length steps, record shadow work
            * w_omega_test <-- propagate test_sim for protocol_length steps, record shadow work
        5. estimate the configuration-space KL-divergence for reference and test simulations
            kldiv_reference = 0.5 * (w_pi_ref - w_omega_ref)
            kldiv_test = 0.5 * (w_pi_test - w_omega_ref)

    References
    ----------
    [1] Using nonequilibrium fluctuation theorems to understand and correct errors in equilibrium and nonequilibrium
        discrete Langevin dynamics simulations (Sivak, Chodera, Crooks, 2013) https://arxiv.org/abs/1107.2967
    """

    # make sure that the simulations we're trying to compare are at the same temperature...
    temperature = reference_sim.integrator.getTemperature()
    if test_sim.integrator.getTemperature() != temperature:
        raise (ValueError('reference_sim and test_sim must be prepared at the same temperature'))

    any_constraints = ((reference_sim.system.getNumConstraints() + test_sim.system.getNumConstraints()) > 0)
    if recycle_v and any_constraints:
        raise (NotImplementedError(
            "haven't yet implemented ability to recycle samples from constrained velocity distributions"))
        # TODO: support recycle_v = True in the presence of constraints

    # set positions, velocities, box vectors to equilibrium values
    clone_state(equilibrium_sim, reference_sim)
    clone_state(equilibrium_sim, test_sim)

    # sample from work distribution w.r.t. pi
    w_pi_reference = measure_shadow_work(reference_sim, protocol_length)
    w_pi_test = measure_shadow_work(test_sim, protocol_length)

    # randomize velocities
    reference_sim.context.setVelocitiesToTemperature(temperature)
    if recycle_v:
        v = reference_sim.context.getState(getVelocities=True).getVelocities(asNumpy=True)
        test_sim.context.setVelocities(v)
    else:
        test_sim.context.setVelocitiesToTemperature(temperature)

    # sample from work distribution w.r.t. omega
    w_omega_reference = measure_shadow_work(reference_sim, protocol_length)
    w_omega_test = measure_shadow_work(test_sim, protocol_length)

    # compute 1-sample estimates of configuration-space KL-divergence
    kldiv_reference = 0.5 * (w_pi_reference - w_omega_reference)
    kldiv_test = 0.5 * (w_pi_test - w_omega_test)

    return kldiv_reference, kldiv_test
