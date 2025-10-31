from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import q_kin_cpp as qkc
from numpy.random import default_rng

from simple_example import UnperturbedSystem

import warnings
warnings.filterwarnings("error")
# Ideas to try:

# Reverse the distribution.
#  - Put more points at high J1.
#  - Make the energies colder at low J1.
#  - simplify generating random J1. Currently, the end of the triangle varies with the given energy. This
#  makes it hard to reason and to control the "spatial" distribution of energies.
#  - Maybe I should first generate the J1 values and then distribute energies accordingly.

rng = default_rng(seed=42)
max_J1 = 0.65
min_J1 = 0.1

T_max = 0.4
T_min = 0.01

unperturbed_system = UnperturbedSystem(a=3.4)
E_min = unperturbed_system.unperturbed_hamiltonian(max_J1, 0)

def get_T_at_J1(x):
    return np.interp(x, [min_J1, max_J1], [T_min, T_max])


def generate_random_J1_levels(num):
    return np.sqrt(rng.triangular(min_J1**2, max_J1**2, max_J1**2, num))


def generate_random_e_at_T(temp):
    return rng.exponential(scale=temp) + E_min


def assign_random_energy_based_on_j1(j1):
    temps = get_T_at_J1(j1)
    return generate_random_e_at_T(temps)


J1 = generate_random_J1_levels(400000)

e = assign_random_energy_based_on_j1(J1)

J2 = unperturbed_system.J2_at_constant_energy(J1, e)


theta1 = rng.uniform(0, 2 * np.pi, size=e.size)
theta2 = rng.uniform(0, 2 * np.pi, size=e.size)

fig, ax = plt.subplots()
ax.hist(J2, bins=100, density=False)
ax.set_xlabel("J2")

fig, ax = plt.subplots()
ax.hist(e, bins=100, density=False)
ax.set_xlabel("Energy E")

fig, ax = plt.subplots()
ax.hist(e[J1 < 0.4], bins=100, density=False)
ax.set_xlabel("Energy E for J1 < 0.4")


fig, ax = plt.subplots()
ax.hist(J1, density=False, bins=100)
ax.set_xlabel("J1")

init_points = np.vstack([J1, J2, theta1, theta2]).T

epsilon = 7e-3

epsilon_array = np.full(3, epsilon)
n1_array = np.full(3, 1)
m2_array = np.array([-4, -5, -6])


evolved_points = np.zeros_like(init_points)
total_points = init_points.shape[0]
prev_per_milleage = 0


def evolve(p):
    sys = qkc.PerturbedSystem(
        a=unperturbed_system.a,
        epsilon_vector=epsilon_array,
        n_vector=n1_array,
        m_vector=m2_array,
    )
    return qkc.evolve(sys, p, 10000, 0.01)


with Pool(10) as pool:
    for i, res in enumerate(pool.imap(evolve, init_points, chunksize=20)):
        current_per_milleage = (1000 * i) // total_points
        if current_per_milleage > prev_per_milleage:
            prev_per_milleage = current_per_milleage
            print(f"completed {0.1 * current_per_milleage:.1f} %")
        evolved_points[i] = res


#  evolved_points = np.vstack(evolved_points)

final_energies = np.array(
    [unperturbed_system.unperturbed_hamiltonian(p[0], p[1]) for p in evolved_points]
)

fig, ax = plt.subplots()
ax.hist(final_energies[evolved_points[:, 0] < 0.4], bins=100, density=False)
ax.set_xlabel("Final Energy E for J1 < 0.4")


plt.show()
