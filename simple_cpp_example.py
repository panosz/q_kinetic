import matplotlib.pyplot as plt
import numpy as np
import q_kin_cpp as qkc

from simple_example_functional_system import UnperturbedSystem


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
        delta_theta_max=3.14
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
    E = 1.55

    epsilon = 7e-3

    epsilon_array = np.full(3, epsilon)
    n1_array = np.full(3, 1)
    m2_array = np.array([-4, -5, -6])

    sys = qkc.PerturbedSystem(
        a=unperturbed_system.a,
        epsilon_vector=epsilon_array,
        n_vector=n1_array,
        m_vector=m2_array,
    )

    j1_init = np.linspace(0.6, 0.99999*unperturbed_system.maximal_J1(E), 40)
    poincare = calculate_poincare_at_given_energy(sys, j1_init, E, max_time=5000)
    fig, ax = plt.subplots()
    J1_cross = poincare[:, 0]
    theta2_cross = poincare[:, 3]

    ax.plot(wrap_minus_pi_pi(theta2_cross), J1_cross, ",k", alpha=0.8)

    plt.show()
