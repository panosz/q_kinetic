from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import q_kin_cpp as qkc

from island_width import (
    ResonanceCondition,
    find_delta_J1_at_resonance,
    find_resonances_at_constant_energy,
)
from simple_example import UnperturbedSystem


@dataclass
class Resonance:
    J1: float
    J2: float
    n1: int
    m2: int


@dataclass
class Island:
    resonance: Resonance
    J1_half_width: float

    def J1_domain(self):
        half_width = self.J1_half_width
        return (self.resonance.J1 - half_width, self.resonance.J1 + half_width)


def wrap_minus_pi_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def poincare_section(sys: qkc.PerturbedSystem, init_conditions, max_time):
    return qkc.get_poincare_points(
        sys,
        initial_state=init_conditions,
        t_max=max_time,
        dt=0.01,
        theta_c=0,
        direction=1,
        delta_theta_max=3.14,
    )


def calculate_poincare_at_given_energy_single_J1(
    system: qkc.PerturbedSystem, j1_init, E, max_time=1000
):
    initial_conditions = np.array(
        [j1_init, unperturbed_system.J2_at_constant_energy(j1_init, E), 0, 0]
    )

    return poincare_section(system, initial_conditions, max_time=max_time)


def calculate_poincare_at_given_energy(
    system: qkc.PerturbedSystem, j1s, E, max_time=1000
):
    poincare = []
    for j1 in j1s:
        poincare.append(
            calculate_poincare_at_given_energy_single_J1(
                system, j1, E, max_time=max_time
            )
        )
    return np.vstack(poincare)


if __name__ == "__main__":

    unperturbed_system = UnperturbedSystem(a=3.4)
    E = 1.75

    epsilon = 7e-3

    epsilon_array = np.full(3, epsilon)
    n1_array = np.full(3, 1)
    m2_array = np.array([-4, -5, -6])

    resonance_conditions = [
        ResonanceCondition(n1, m2) for n1, m2 in zip(n1_array, m2_array)
    ]

    resonances = {}
    for rc in resonance_conditions:
        n1 = rc.n1
        m2 = rc.m2
        resonances[rc] = [
            Resonance(J1_res, J2_res, n1, m2)
            for J1_res, J2_res in find_resonances_at_constant_energy(
                unperturbed_system, rc, E, 0.1, 0.999999*unperturbed_system.maximal_J1(E)
            )
        ]

    resonance_islands = []
    for rc, epsilon in zip(resonance_conditions, epsilon_array):
        n1 = rc.n1
        m2 = rc.m2
        for resonance in resonances[rc]:
            delta_J1 = find_delta_J1_at_resonance(
                unperturbed_system,
                epsilon,
                rc,
                resonance.J1,
                resonance.J2,
            )
            island = Island(
                resonance,
                delta_J1,
            )
            resonance_islands.append(island)

    resonance_islands = sorted(
        resonance_islands, key=lambda island: island.resonance.J1
    )

    sys = qkc.PerturbedSystem(
        a=unperturbed_system.a,
        epsilon_vector=epsilon_array,
        n_vector=n1_array,
        m_vector=m2_array,
    )

    j1_init = np.linspace(0.1, 0.99999 * unperturbed_system.maximal_J1(E), 40)
    j1_init = np.append(j1_init, unperturbed_system.J1_critical(E))
    poincare = calculate_poincare_at_given_energy(sys, j1_init, E, max_time=10000)
    fig, ax = plt.subplots(num=f"Poincare section at E={E}, epsilon={epsilon:.2e}")
    J1_cross = poincare[:, 0]
    theta2_cross = poincare[:, 3]

    ax.plot(wrap_minus_pi_pi(theta2_cross), J1_cross, ",k", alpha=0.5)
    ax.plot(np.zeros_like(j1_init), j1_init, "+r", alpha=0.5)
    ax.set_ylim(0.2, None)

    # set x ticks at -pi, -pi/2, 0, pi/2, pi
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.set_ylabel(r"$J_1$")
    ax.set_xlabel(r"$\theta_2$")
    ax.set_xlim(-np.pi, np.pi)

    for island in resonance_islands:
        ax.axhline(island.resonance.J1, ls="--", color="blue", alpha=0.7)
        #  ax.axhspan(*island.J1_domain(), color="red", alpha=0.3)

    ax.axhline(
        unperturbed_system.J1_critical(E),
        ls="--",
        color="red",
        lw=1.5,
        label="J1 critical",
    )
    plt.show()
