from pickle import load

import matplotlib.pyplot as plt
import numpy as np

with open('data/hmr_stability.pkl', 'rb') as f:
    final_beliefs = load(f)

hydrogen_masses = sorted(final_beliefs.keys())
y = []
yerr = []

for hydrogen_mass in hydrogen_masses:
    x, f = final_beliefs[hydrogen_mass]

    median = x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - 0.5))]

    bottom = median - x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - 0.025))]
    top = x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - 0.975))] - median

    y.append(median)
    yerr.append((bottom, top))

yerr = np.array(yerr).T

plt.plot(hydrogen_masses, y)
plt.errorbar(hydrogen_masses, y, yerr=yerr, fmt='none')

plt.xlabel('H mass (a.m.u.)')
plt.ylabel('stability threshold (fs)')

plt.savefig('figures/hmr_stability.pdf')
