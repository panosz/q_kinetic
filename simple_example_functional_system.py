import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


class UnperturbedSystem:
    """
    A simple hamiltonian with hamiltonian
    H = a J1**2 + J2**2 * J1
    """

    def __init__(self, a):
        self.a = a

    def unperturbed_hamiltonian(self, J1, J2):
        return self.a * J1**2 + J2**2 * J1

    def omega1(self, J1, J2):
        return omega1(self.a, J1, J2)

    def omega2(self, J1, J2):
        return omega2(J1, J2)

    def J2_at_constant_energy(self, J1, E):
        return np.sqrt((E - self.a * J1**2) / J1)

    def maximal_J1(self, E):
        return np.sqrt(E / self.a)

    def kinetic_q(self, J1, J2):
        return self.omega1(J1, J2) / self.omega2(J1, J2)

    def kinetic_q_at_constant_energy(self, J1, E):
        J2 = self.J2_at_constant_energy(J1, E)
        return self.kinetic_q(J1, J2)

    def dydt(self, t: float, y: np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        return unperturbed_dydt(t, y, self.a)


@njit
def omega1(a, J1, J2):
    return 2 * a * J1 + J2**2


@njit
def omega2(J1, J2):
    return 2 * J2 * J1


@njit
def unperturbed_dydt(t: float, y: np.ndarray, a: float):
    J1, J2, theta1, theta2 = y
    return np.array([0, 0, omega1(a, J1, J2), omega2(J1, J2)])


class Perturbation:
    def __init__(self, epsilon, n1, m2):
        self.epsilon = epsilon
        self.n1 = n1
        self.m2 = m2

    def hamiltonian(self, theta1, theta2):
        return self.epsilon * np.sin(self.n1 * theta1 + self.m2 * theta2)

    def dydt(self, t: float, y: np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        return perturbed_dydt(t, y, self.epsilon, self.n1, self.m2)


@njit
def perturbed_dydt(t: float, y: np.ndarray, epsilon: float, n1: int, m2: int):
    J1, J2, theta1, theta2 = y
    phase = n1 * theta1 + m2 * theta2
    cos = np.cos(phase)
    return epsilon * np.array([-n1 * cos, -m2 * cos, 0, 0])


class PerturbedSystem:
    def __init__(
        self, unperturbed_system: UnperturbedSystem, perturbations: list[Perturbation]
    ):
        self.unperturbed_system = unperturbed_system
        self.perturbations = perturbations

    def hamiltonian(self, J1, J2, theta1, theta2):
        return self.unperturbed_system.unperturbed_hamiltonian(J1, J2) + sum(
            [
                perturbation.hamiltonian(theta1, theta2)
                for perturbation in self.perturbations
            ]
        )

    def dydt(self, t: float, y: np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        return self.unperturbed_system.dydt(t, y) + sum(
            [perturbation.dydt(t, y) for perturbation in self.perturbations]
        )


def wrap_minus_pi_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def wrap_zero_two_pi(theta):
    return theta % (2 * np.pi)


class Theta1CrossEvent:
    def __init__(self, direction=1):
        self.direction = direction
        self.terminal = False

    def __call__(self, t, y):
        theta1 = y[2]
        return np.sin(theta1)


@njit
def theta1_cross_event(t, y, *_):
    theta1 = y[2]
    return np.sin(theta1)

def is_close_to_zero(theta):
    return np.abs(theta) < 1e-3


def poincare_section(
    system: PerturbedSystem, initial_conditions: np.ndarray, max_time=1000
):

    a = system.unperturbed_system.a
    epsilons = np.array([p.epsilon for p in system.perturbations])
    n1s = np.array([p.n1 for p in system.perturbations])
    m2s = np.array([p.m2 for p in system.perturbations])


    sol = solve_ivp(
        dydt_system,
        (0, max_time),
        initial_conditions,
        events=theta1_cross_event,
        args=(a, epsilons, n1s, m2s),
        rtol=1e-8,
        atol=1e-8,
    )

    crossing_points = sol.y_events[0]

    ## filter out possible events at theta1 = pi
    is_crossing_at_zero = is_close_to_zero(wrap_minus_pi_pi(crossing_points[:, 2]))

    return crossing_points[is_crossing_at_zero]


def calculate_poincare_at_given_energy_single_J1(
    system: PerturbedSystem, j1_init, E, max_time=1000
):
    initial_conditions = np.array(
        [j1_init, unperburbed_system.J2_at_constant_energy(j1_init, E), 0, 0]
    )

    return poincare_section(system, initial_conditions, max_time=max_time)


def calculate_poincare_at_given_energy(system: PerturbedSystem, j1s, E, max_time=1000):
    poincare = []
    for j1 in j1s:
        poincare.append(
            calculate_poincare_at_given_energy_single_J1(
                system, j1, E, max_time=max_time
            )
        )
    return np.vstack(poincare)


@njit
def dydt_system(
    t: float,
    y: np.ndarray,
    a: float,
    epsilons: np.ndarray,
    n1s: np.ndarray,
    m2s: np.ndarray,
):
    dydt_unperturbed = unperturbed_dydt(t, y, a)
    dydt_perturbations = np.zeros(4)
    for epsilon, n1, m2 in zip(epsilons, n1s, m2s):
        dydt_perturbations += perturbed_dydt(t, y, epsilon, n1, m2)
    return dydt_unperturbed + dydt_perturbations


if __name__ == "__main__":

    unperburbed_system = UnperturbedSystem(a=3.4)

    E = 1.59

    j1 = np.linspace(0.1, 0.8, 100)
    epsilon = 1e-2

    epsilon_array = np.full(3, epsilon)
    n1_array = np.full(3, 1)
    m2_array = np.array([-4, -5, -6])

    perburations = [
        Perturbation(epsilon=epsilon_i, n1=n1_i, m2=m2_array)
        for epsilon_i, n1_i, m2_array in zip(epsilon_array, n1_array, m2_array)
    ]

    system = PerturbedSystem(unperburbed_system, perburations)

    fig, ax = plt.subplots()
    ax.plot(j1, unperburbed_system.kinetic_q_at_constant_energy(j1, E))
    ax.set_xlabel("J1")
    ax.set_ylabel("q")
    ax.set_title(f"Kinetic q at E = {E}, alpha = {system.unperturbed_system.a}")
    ax.set_ylim(0, 8)
    ax.axhline(4, color="black", lw=0.5)
    ax.axhline(5, color="black", lw=0.5)
    ax.axhline(6, color="black", lw=0.5)

    j1_init = [
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    poincare = calculate_poincare_at_given_energy(system, j1_init, E, max_time=10000)
    fig, ax = plt.subplots()
    J1_cross = poincare[:, 0]
    theta2_cross = poincare[:, 3]

    ax.plot(wrap_minus_pi_pi(theta2_cross), J1_cross, ",k", alpha=0.8)

    plt.show()
