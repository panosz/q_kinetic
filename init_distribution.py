import matplotlib.pyplot as plt
import numpy as np
import q_kin_cpp as qkc
from numpy.random import default_rng

from simple_example import UnperturbedSystem

# Ideas to try:

# Reverse the distribution. 
#  - Put more points at high J1.
#  - Make the energies colder at low J1.
#  - simplify generating random J1. Currently, the end of the triangle varies with the given energy. This
#  makes it hard to reason and to control the "spatial" distribution of energies.
#  - Maybe I should first generate the J1 values and then distribute energies accordingly.

rng = default_rng(seed=42)
min_E = 1.52
max_E = 1.9

def generate_random_energy_levels(num):
    random_energy_levels = min_E + rng.exponential(scale=0.1, size=num)
    return random_energy_levels[random_energy_levels < max_E]


def generate_single_J1_value_given_energy(energy, system, min_J1):
    max_J1 = system.maximal_J1(energy)
    return rng.triangular(min_J1, min_J1, max_J1)


def assign_random_J1(energies, system, min_J1):

    return np.array(
        [
            generate_single_J1_value_given_energy(energy, system, 0.25)
            for energy in energies
        ]
    )


e = generate_random_energy_levels(4000000)

unperturbed_system = UnperturbedSystem(3.4)
J1 = assign_random_J1(e, unperturbed_system, 0.1)
J2 = unperturbed_system.J2_at_constant_energy(J1, e)
theta1 = rng.uniform(0, 2 * np.pi, size=e.size)
theta2 = rng.uniform(0, 2 * np.pi, size=e.size)

fig, ax = plt.subplots()
ax.hist(J2, bins=100, density=True)
ax.set_xlabel("J2")

fig, ax = plt.subplots()
ax.hist(e, bins=100, density=True)
ax.set_xlabel("Energy E")

fig, ax = plt.subplots()
ax.hist(e[J1 > 0.6], bins=50, density=False)
ax.set_xlabel("Energy E for J1 > 0.6")


fig, ax = plt.subplots()
ax.hist(J1[e>1.5], bins=100, density=True)
ax.set_xlabel("J1")

init_points = np.vstack([J1, J2, theta1, theta2]).T

epsilon = 6e-3

epsilon_array = np.full(3, epsilon)
n1_array = np.full(3, 1)
m2_array = np.array([-4, -5, -6])

sys = qkc.PerturbedSystem(
    a=unperturbed_system.a,
    epsilon_vector=epsilon_array,
    n_vector=n1_array,
    m_vector=m2_array,
)

evolved_points = []
total_points = init_points.shape[0]
prev_per_milleage = 0
for i, p in enumerate(init_points):
    current_per_milleage = (1000 * i) // total_points
    if current_per_milleage > prev_per_milleage:
        prev_per_milleage = current_per_milleage
        print(f"completed {0.1 * current_per_milleage:.1f} %")

    evolved_points.append(qkc.evolve(sys, p, 1000, 0.01))


evolved_points = np.vstack(evolved_points)

final_energies = np.array([unperturbed_system.unperturbed_hamiltonian(p[0], p[1]) for p in evolved_points])

fig, ax = plt.subplots()
ax.hist(final_energies[evolved_points[:, 0] > 0.6], bins=100, density=True)
ax.set_xlabel("Final Energy E for J1 > 0.6")


plt.show()
