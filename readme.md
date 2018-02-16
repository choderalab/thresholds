[![Build Status](https://travis-ci.org/choderalab/thresholds.svg?branch=master)](https://travis-ci.org/choderalab/thresholds?branch=master)

# thresholds

## Motivation
* What is the maximum timestep I can use with this integrator on this system, if I don't want my trajectory to be all NaNs?
* What is the maximum timestep I can use with this fancy new integrator on this system, if I want to introduce no more error than usual?

We'd like to get quick answers to these questions.

## Methods
* utilities for estimating whether integration at a given timestep is stable or not
* utilities for estimating whether a given timestep introduces more error than some tolerable reference method
* noisy binary search tools, for identifying timesteps that meet stability or error criteria

### Definitions
* *"Stability"* -- If you wait long enough, your finite-timestep simulation with finite-precision arithmetic will eventually try to set particle positions to NaN. Further, if you initialize your simulation in weird enough starting conditions, you can make your simulation encounter NaNs as quickly as you want. Thus, we define "stability" in terms of the probability that the simulation will crash within a _specified number of steps_, given a _specified ensemble of initial conditions_.
* *"Error"* -- Finite-timestep simulations will always introduce bias into the sampled distribution (unless Metropolis-corrected). We define "error" as the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the sampled distribution and the target distribution.

### Details
The threshold-finding problem is naturally phrased as a stochastic root-finding problem, for which many efficient algorithms are available.
In this setting, we assume access to noisy measurements of some monotonic function over some timestep range (e.g. "the probability that my simulation won't NaN" as a function of timestep, or "how much more sampling bias could we tolerate" as a function of timestep).
We define a noisy "oracle" that reports on the sign of our chosen function, corrupted by noise (e.g. "run a simulation at this timestep, and see whether it NaN'd or not").
Defining an oracle for "stability" is relatively straightforward, but defining an oracle for "tolerable error" is trickier...

With a suitable "oracle" defined, we then use a stochastic root-finding algorithm to find a zero of this function.
Initially, we are using a stylized version of the probabilistic bisection algorithm (i.e. noisy binary search).
This converges exponentially fast under the (unrealistic) assumption that the probability of getting the sign wrong is constant over the interesting range of timesteps.
In the next step of this project, we plan to apply variants of probabilistic bisection that don't make that assumption (e.g. as described in Chapter 3 of https://people.orie.cornell.edu/shane/theses/ThesisRolfWaeber.pdf).

## References
The algorithms we use for noisy binary search here are mostly based on the description of probabilistic bisection algorithm (PBA) by Rolf Waeber, Peter Frazier and colleagues. See https://people.orie.cornell.edu/pfrazier/Presentations/2014.01.Lancaster.bisection.pdf for a complete description and pointers to references.

The "error oracle" is very experimental, and is based on [our ongoing work](https://github.com/choderalab/integrator-benchmark) to measure configuration-sampling error of Langevin integrators using near-equilibrium approximations.
