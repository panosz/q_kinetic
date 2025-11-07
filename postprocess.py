import matplotlib.pyplot as plt
import numpy as np

from simple_example import UnperturbedSystem

data = np.load("./runs1.npz")


unperturbed_system = UnperturbedSystem(a=3.4)

init_points = data["init_points"]


initial_energies = unperturbed_system.unperturbed_hamiltonian(
    init_points[:, 0], init_points[:, 1]
)

evolved_points = data["evolved_points"]


final_energies = np.array(
    [unperturbed_system.unperturbed_hamiltonian(p[0], p[1]) for p in evolved_points]
)

plt.hist(initial_energies, bins=100, density=False)


fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

nbins = 500
transport_window = (1.65, 1.75)

axs[0].hist(initial_energies[init_points[:, 0] < 0.4], bins=nbins, density=False)
axs[0].set_xlabel("Initial Energy E for J1 < 0.4")
axs[0].axvspan(transport_window[0], transport_window[1], color="red", alpha=0.3)

axs[1].hist(final_energies[evolved_points[:, 0] < 0.4], bins=nbins, density=False)
axs[1].set_xlabel("Final Energy E for J1 < 0.4")
axs[1].set_xlim(1.4, 2.5)
axs[1].axvspan(transport_window[0], transport_window[1], color="red", alpha=0.3)

fig, ax = plt.subplots()

ax.hist(final_energies[init_points[:, 0] < 0.4], bins=100, density=False)
ax.set_xlabel("Final Energy distribution for points initially atJ1 < 0.4")


plt.show()
