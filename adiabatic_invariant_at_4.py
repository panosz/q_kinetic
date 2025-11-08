import matplotlib.pyplot as plt
import numpy as np

from island_width import ResonanceCondition, find_resonances_at_constant_energy
from simple_example import UnperturbedSystem


def adiabatic_invariant_at_4(
    alpha: float,
    E: float,
    theta1: float,
    epsilon: float,
    J1: np.array,
    theta2: np.array,
):
    J1 = np.asarray(J1)
    theta2 = np.asarray(theta2)
    numerator = E - epsilon * np.sin(theta1 - 4 * theta2) - alpha * J1**2

    return np.sqrt(numerator / J1) + 4 * J1


if __name__ == "__main__":

    unperturbed_system = UnperturbedSystem(a=3.4)

    resonance_condition = ResonanceCondition(n1=1, m2=-4)

    E = 1.75

    resonace_locations = find_resonances_at_constant_energy(
        unperturbed_system,
        resonance_condition,
        E,
        0.3,
        unperturbed_system.maximal_J1(E),
    )

    J1_res, J2_res = zip(*resonace_locations)

    epsilon = 7e-3

    theta2 = np.linspace(-np.pi, np.pi, 1000)
    J1 = np.linspace(0.2, 0.65, 1000)

    Theta2, J1_mesh = np.meshgrid(theta2, J1)

    AdiabaticInvariant = adiabatic_invariant_at_4(
        unperturbed_system.a, E, 0, epsilon, J1_mesh, Theta2
    )

    j1_levels_fill = np.linspace(J1.min(), J1.max(), 15)

    j2_levels = np.concatenate(
        [
            adiabatic_invariant_at_4(
                unperturbed_system.a, E, 0, epsilon, J1_res, np.pi / 8
            ),
            adiabatic_invariant_at_4(
                unperturbed_system.a, E, 0, epsilon, J1_res, -np.pi / 8
            ),
        ]
    )

    j2_levels_fill = np.concatenate(
        [
            adiabatic_invariant_at_4(
                unperturbed_system.a, E, 0, epsilon, j1_levels_fill, np.pi / 8
            ),
            adiabatic_invariant_at_4(
                unperturbed_system.a, E, 0, epsilon, j1_levels_fill, -np.pi / 8
            ),
        ]
    )

    j2_levels = np.unique(j2_levels)
    j2_levels = sorted(j2_levels)
    j2_levels_fill = np.unique(j2_levels_fill)
    j2_levels_fill = sorted(j2_levels_fill)

    fig, ax = plt.subplots(
        num=f"adiabatic phase space for E={E}, epsilon={epsilon:.2e}", figsize=(8, 6)
    )
    contour = ax.contour(
        Theta2, J1_mesh, AdiabaticInvariant, levels=j2_levels_fill, colors="k"
    )
    contour = ax.contour(
        Theta2, J1_mesh, AdiabaticInvariant, levels=j2_levels, colors="r"
    )

    #  for J1_res_i in J1_res:
    #  ax.axhline(J1_res_i, color="red", linestyle="--")

    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.set_ylabel(r"$J_1$")
    ax.set_xlabel(r"$\theta_2$")
    ax.set_xlim(-np.pi, np.pi)

    plt.show()
