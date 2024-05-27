import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from simple_example import UnperturbedSystem

rng = default_rng(seed=42)
min_E = 1.0


def generate_random_energy_levels(num):
    random_energy_levels = min_E + rng.exponential(scale=1, size=num)
    return random_energy_levels


def generate_single_J1_value_given_energy(energy, system, min_J1):
    max_J1 = system.maximal_J1(energy)
    return rng.triangular(min_J1, min_J1, max_J1)



def assign_random_J1(energies, system, min_J1):

    return np.array([generate_single_J1_value_given_energy(energy, system, 0.1) for energy in energies])



e = generate_random_energy_levels(100000)

system = UnperturbedSystem(3.4)
J1 = assign_random_J1(e, system, 0.1)
J2 = system.J2_at_constant_energy(J1, e)
theta1 = rng.uniform(0, 2 * np.pi, size=e.size)
theta2 = rng.uniform(0, 2 * np.pi, size=e.size)

plt.hist(J2, bins=100, density=True)

plt.show()
