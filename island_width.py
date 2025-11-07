import matplotlib.pyplot as plt
import numpy as np

from roots import roots
from simple_example import UnperturbedSystem


class ResonanceCondition:
    def __init__(self, n1: int, m2: int):
        self.n1 = n1
        self.m2 = m2

    def __call__(self, omega1, omega2):
        return self.n1 * omega1 + self.m2 * omega2

    def __str__(self):
        return f"{self.n1}*omega1 + {self.m2}*omega2 = 0"

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(n1={self.n1}, m2={self.m2})"

    def __hash__(self):
        return hash((self.n1, self.m2))


def find_resonances_at_constant_energy(
    us: UnperturbedSystem,
    rc: ResonanceCondition,
    E: float,
    J1_min: float,
    J1_max: float,
    num_points: int = 1000,
):
    def f(J1):
        J2 = us.J2_at_constant_energy(J1, E)
        omega1 = us.omega1(J1, J2)
        omega2 = us.omega2(J1, J2)
        return rc(omega1, omega2)

    window = (J1_min, J1_max)
    roots_J1 = roots(f, window, n_samples=num_points)
    resonances = []
    for J1, converged in roots_J1:
        if not converged:
            raise RuntimeError("Root finding did not converge")
        J2 = us.J2_at_constant_energy(J1, E)
        resonances.append((J1, J2))
    return resonances


def find_effective_mass_at_resonance(
    us: UnperturbedSystem, rc: ResonanceCondition, J1: float, J2: float
):
    n1 = rc.n1

    m2 = rc.m2

    d2H_dJ1 = us.d2H_J1(J1, J2)
    d2H_dJ2 = us.d2H_J2(J1, J2)
    d2H_dJ1_J2 = us.d2H_J1_J2(J1, J2)

    return d2H_dJ1 * n1**2 + d2H_dJ2 * m2**2 + 2 * d2H_dJ1_J2 * n1 * m2


def find_delta_J1_at_resonance(
    us: UnperturbedSystem,
    perturbation_strength: float,
    rc: ResonanceCondition,
    J1: float,
    J2: float,
):
    n1 = rc.n1
    m2 = rc.m2
    effective_mass = find_effective_mass_at_resonance(us, rc, J1, J2)

    return 2 * np.sqrt(perturbation_strength * n1**2 / abs(effective_mass))


if __name__ == "__main__":

    unperturbed_system = UnperturbedSystem(a=3.4)

    rc = ResonanceCondition(n1=1, m2=-4)

    E = 1.62

    j1 = np.linspace(0.1, 0.8, 100)

    fig, ax = plt.subplots()
    ax.plot(j1, unperturbed_system.kinetic_q_at_constant_energy(j1, E))
    ax.set_xlabel("J1")
    ax.set_ylabel("q")

    resonances = find_resonances_at_constant_energy(
        unperturbed_system, rc, E, J1_min=0.1, J1_max=unperturbed_system.maximal_J1(E)
    )

    for res in resonances:
        ax.axvline(
            res[0], color="red", linestyle="--", label=f"Resonance at J1={res[0]:.3f}"
        )

    ax.axhline(4, color="black", lw=0.5)
    ax.axhline(5, color="black", lw=0.5)
    ax.axhline(6, color="black", lw=0.5)

    for res in resonances:
        effective_mass = find_effective_mass_at_resonance(
            unperturbed_system, rc, res[0], res[1]
        )
        print(
            f"Resonance at J1={res[0]:.3f}, J2={res[1]:.3f}, Effective Mass={effective_mass:.3f}"
        )
        delta_J1 = find_delta_J1_at_resonance(
            unperturbed_system, perturbation_strength=7e-3, rc=rc, J1=res[0], J2=res[1]
        )
        print(f"delta_J1 = {delta_J1:.5f}")
        resonance_domain = (res[0] - delta_J1, res[0] + delta_J1)
        print(
            f"resonance domain: {resonance_domain[0]:.5f} to {resonance_domain[1]:.5f}"
        )

    plt.show()
