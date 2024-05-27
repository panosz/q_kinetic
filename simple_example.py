from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt



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
        return 2 * self.a * J1 + J2**2

    def omega2(self, J1, J2):
        return 2 * J2 * J1

    def J2_at_constant_energy(self, J1, E):
        return np.sqrt((E - self.a * J1**2) / J1)


    def kinetic_q(self, J1, J2):
        return self.omega1(J1, J2)/ self.omega2(J1, J2)

    def kinetic_q_at_constant_energy(self, J1, E):
        J2 = self.J2_at_constant_energy(J1, E)
        return self.kinetic_q(J1, J2)

    def dydt(self, t:float, y:np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        J1, J2, theta1, theta2 = y
        return np.array([0, 0, self.omega1(J1, J2), self.omega2(J1, J2)])




class Perturbation:
    def __init__(self, epsilon, n1, m2):
        self.epsilon = epsilon
        self.n1 = n1
        self.m2 = m2

    def hamiltonian(self, theta1, theta2):
        return self.epsilon * np.sin(self.n1 * theta1 + self.m2 * theta2)

    def dydt(self, t:float, y:np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        J1, J2, theta1, theta2 = y
        phase = self.n1 * theta1 + self.m2 * theta2
        cos = np.cos(phase)
        return self.epsilon * np.array([- self.n1 * cos, - self.m2 * cos, 0, 0])


class PerturbedSystem:
    def __init__(self, unperturbed_system:UnperturbedSystem, perturbations:list[Perturbation]):
        self.unperturbed_system = unperturbed_system
        self.perturbations = perturbations

    def hamiltonian(self, J1, J2, theta1, theta2):
        return self.unperturbed_system.unperturbed_hamiltonian(J1, J2) + sum([perturbation.hamiltonian(theta1, theta2) for perturbation in self.perturbations])

    def dydt(self, t:float, y:np.ndarray):
        """
        returns the time derivative of the state vector (J1, J2, theta1, theta2)
        """
        return self.unperturbed_system.dydt(t, y) + sum([perturbation.dydt(t, y) for perturbation in self.perturbations])


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


def is_close_to_zero(theta):
    return np.abs(theta) < 1e-3
    

def poincare_section(system:PerturbedSystem, initial_conditions:np.ndarray, max_time=1000):
    sol = solve_ivp(system.dydt, (0, max_time), initial_conditions, events=Theta1CrossEvent(), rtol=1e-8, atol=1e-8)

    crossing_points = sol.y_events[0]

    ## filter out possible events at theta1 = pi
    is_crossing_at_zero = is_close_to_zero(wrap_minus_pi_pi(crossing_points[:, 2]))

    return crossing_points[is_crossing_at_zero]



def calculate_poincare_at_given_energy_single_J1(system:PerturbedSystem, j1_init, E, max_time=1000):
    initial_conditions = np.array([j1_init, unperburbed_system.J2_at_constant_energy(j1_init, E), 0, 0])

    return poincare_section(system, initial_conditions, max_time=max_time)

def calculate_poincare_at_given_energy(system:PerturbedSystem, j1s, E, max_time=1000):
    poincare = []
    for j1 in j1s:
        poincare.append(calculate_poincare_at_given_energy_single_J1(system, j1, E, max_time=max_time))
    return np.vstack(poincare)
    

unperburbed_system = UnperturbedSystem(a=3.4)

E = 1.59

j1 = np.linspace(0.1, 0.8, 100)
epsilon = 1e-2

perburations = [Perturbation(epsilon=epsilon, n1=1, m2=-4), Perturbation(epsilon=epsilon, n1=1, m2=-5), Perturbation(epsilon=epsilon, n1=1, m2=-6)]

system = PerturbedSystem(unperburbed_system, perburations)



fig, ax = plt.subplots()
ax.plot(j1, unperburbed_system.kinetic_q_at_constant_energy(j1, E))
ax.set_xlabel('J1')
ax.set_ylabel('q')
ax.set_title(f'Kinetic q at E = {E}, alpha = {system.unperturbed_system.a}')
ax.set_ylim(0, 8)
ax.axhline(4, color='black', lw=0.5)
ax.axhline(5, color='black', lw=0.5)
ax.axhline(6, color='black', lw=0.5)

j1_init = [ 0.25, 0.3, 
           0.4,
           0.5, 
           0.6,
           ]
poincare = calculate_poincare_at_given_energy(system, j1_init, E, max_time=10000)
fig, ax = plt.subplots()
J1_cross = poincare[:, 0]
theta2_cross = poincare[:, 3]

ax.plot(wrap_minus_pi_pi(theta2_cross), J1_cross, ',k', alpha=0.8)

plt.show()
