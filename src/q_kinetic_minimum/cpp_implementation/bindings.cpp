#include <Eigen/Core>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using State = Eigen::Vector4d;

using ErrorStepperType = boost::numeric::odeint::runge_kutta_cash_karp54<
    State, double, State, double, boost::numeric::odeint::vector_space_algebra>;

// Define a simple C++ function
inline constexpr double omega1(double a, double J1, double J2) noexcept {
  return 2 * a * J1 + J2 * J2;
}

inline constexpr double omega2(double J1, double J2) noexcept {
  return 2 * J2 * J1;
}

inline State unperturbed_system(const State &s, double a) noexcept {
  const double J1 = s(0);
  const double J2 = s(1);
  return State{0.0, 0.0, omega1(a, J1, J2), omega2(J1, J2)};
}

/* assumes that epsilon_vector, n_vector, and m_vector have the same length */
inline State perturbed_dydt(const State &s, double a,
                            std::vector<double> epsilon_vector,
                            const std::vector<int> &n_vector,
                            const std::vector<int> &m_vector) noexcept {
  State dsdt = unperturbed_system(s, a);

  for (size_t k = 0; k < epsilon_vector.size(); ++k) {
    const double epsilon_k = epsilon_vector[k];
    const int n_k = n_vector[k];
    const int m_k = m_vector[k];
    const double phase = n_k * s(2) + m_k * s(3);
    const double cos_phase = std::cos(phase);
    dsdt(0) += -epsilon_k * n_k * cos_phase;
    dsdt(1) += -epsilon_k * m_k * cos_phase;
  }
  return dsdt;
}

struct PerturbedSystem {
  double a;
  std::vector<double> epsilon_vector;
  std::vector<int> n_vector;
  std::vector<int> m_vector;
  // Constructor asserting equal lengths of perturbation vectors
  PerturbedSystem(double a_, const std::vector<double> &epsilon_vector_,
                  const std::vector<int> &n_vector_,
                  const std::vector<int> &m_vector_)
      : a(a_), epsilon_vector(epsilon_vector_), n_vector(n_vector_),
        m_vector(m_vector_) {
    if (epsilon_vector.size() != n_vector.size() ||
        epsilon_vector.size() != m_vector.size()) {
      throw std::invalid_argument(
          "Perturbation vectors must have the same length");
    }
  }
  inline State eoms(const State &s) const noexcept {
    return perturbed_dydt(s, a, epsilon_vector, n_vector, m_vector);
  }

  inline State eoms_perp_theta1(const State &s) const noexcept {
    State dsdt = eoms(s);
    const double dtheta1_dt = dsdt(2);
    return dsdt / dtheta1_dt;
  }

  void sys_perp_theta1(const State &s, State &dsdt,
                       double /*t*/) const noexcept {
    dsdt = eoms_perp_theta1(s);
  }

  void operator()(const State &s, State &dsdt, double /*t*/) const noexcept {
    dsdt = eoms(s);
  }
};

double wrap_2pi(double angle) noexcept {
  using boost::math::double_constants::one_div_two_pi;
  using boost::math::double_constants::two_pi;

  return angle - two_pi * floor(angle * one_div_two_pi);
}

double wrap_minus_pi_pi(double angle) noexcept {
  using boost::math::double_constants::pi;

  return wrap_2pi(angle + pi) - pi;
}

constexpr inline bool different_sign(double a, double b) noexcept {
  return ((a < 0) & (b >= 0)) | ((a > 0) & (b <= 0));
}

bool exists_angle_crossing(double angle_1, double angle_2, double angle_c,
                           double delta_angle_max) {

  double delta_angle_1 = wrap_minus_pi_pi(angle_1 - angle_c);
  double delta_angle_2 = wrap_minus_pi_pi(angle_2 - angle_c);

  // Filter out spurious crossings due to mod 2*pi discontinuities
  if (std::abs(delta_angle_1 - delta_angle_2) >= delta_angle_max) {
    return false;
  }

  // The order below matters, to avoid early detection.
  // We want our detector to return true if delta_angle_2 == 0,
  // i.e., when we happen to land exactly on the surface for the first time.
  // If it returned true for delta_angle_1 == 0, then we may suffer
  // from early detection, when, as is often the case,
  // the first point in the sequence is on the angle = angle_c surface.
  return different_sign(delta_angle_2, delta_angle_1);
}

class PoincareObserver
/*
    Callable class for detecting if two orbit points are on either side of a
    given poloidal ray in a given direction.

    Parameters:
    -----------
    theta_c:
        the angle of the poloidal ray.

    direction:
      the direction in which the crossing should be detected. Can be either 1 or
   -1.

    delta_theta_max:
        The maximum allowed angle difference for considering a crossing to be
   valid. Default is 1.0. Spurious crossings caused by discontinuities at modulo
   2 pi boundaries are excluded if the angle difference exceeds
   `delta_theta_max`.

    Notes:
    --------
    operator()(y1, y2) accepts as input two adjacent samples
    of a gyrocenter orbit, `y1` and `y2`, and
    returns `true` if `y1` and `y2`
    are on opposite sides of the
    theta == `theta_c` modulo 2 * pi
    poloidal ray, or if `y2` is exactly on that ray.

    See documentation of `exists_angle_crossing` for more details about the
   crossing detection method.
*/
{
private:
  double _theta_c;
  int _direction;
  double _delta_theta_max;
  State _prev_state;
  bool _first_call;
  PerturbedSystem _system;

public:
  std::vector<State> observations;
  PoincareObserver(PerturbedSystem system, double theta_c, int direction,
                   double delta_theta_max = 3.0)
      : _theta_c(theta_c), _direction(direction),
        _delta_theta_max(delta_theta_max), _prev_state{}, _first_call{true},
        _system{system}, observations{} {
    if (direction != 1 && direction != -1) {
      throw std::invalid_argument("Direction must be either 1 or -1.");
    }
  };

  bool check_event(const State &y1, const State &y2) const noexcept {
    if ((y2[2] - y1[2]) * _direction < 0)
      return false;

    return exists_angle_crossing(y2[2], y1[2], _theta_c, _delta_theta_max);
  }

  void operator()(const State &y, double /*t*/) {
    if (_first_call) {
      _prev_state = y;
      _first_call = false;
      return;
    }

    if (check_event(_prev_state, y)) {
      const double delta_theta_c = wrap_minus_pi_pi(_theta_c - _prev_state[2]);
      const double delta_theta = wrap_minus_pi_pi(y[2] - _prev_state[2]);

      auto stepper = ErrorStepperType{};

      State crossing_state = _prev_state;

      stepper.do_step([this](const State &s, State &dsdt,
                             double t) { _system.sys_perp_theta1(s, dsdt, t); },
                      crossing_state, 0.0, delta_theta_c);

      observations.push_back(crossing_state);
    }
    _prev_state = y;
  }
};

std::vector<State> get_poincare_points(PerturbedSystem system,
                                       const State &initial_state, double t_max,
                                       double dt, double theta_c, int direction,
                                       double delta_theta_max) {
  PoincareObserver observer(system, theta_c, direction, delta_theta_max);
  State s = initial_state;

  using namespace boost::numeric::odeint;

  integrate_adaptive(make_controlled<ErrorStepperType>(1e-9, 1e-9), system, s,
                     0.0, t_max, dt,
                     [&observer](const State &x, double t) { observer(x, t); });

  return observer.observations;
}


std::vector<State> get_orbit(PerturbedSystem system,
                                       const State &initial_state, double t_max,
                                       double dt) {
  std::vector<State> out;
  State s = initial_state;


  using namespace boost::numeric::odeint;

  integrate_adaptive(make_controlled<ErrorStepperType>(1e-9, 1e-9), system, s,
                    0.0, t_max, dt,
                     [&out](const State &x, double t) { out.push_back(x); });

  return out;
}

State evolve(PerturbedSystem system,
                                       const State &initial_state, double t_end,
                                       double dt) {
  State s = initial_state;


  using namespace boost::numeric::odeint;

  integrate_adaptive(make_controlled<ErrorStepperType>(1e-9, 1e-9), system, s,
                    0.0, t_end, dt);

  return s;
}


// Define the Python module
PYBIND11_MODULE(q_kin_cpp, m) {
  m.doc() = "pybind11 example plugin with a simple plus function";
  m.def("omega1", &omega1, "unperturbed frequency of DOF 1", "a"_a, "J1"_a,
        "J2"_a)
      .def("omega2", &omega2, "unperturbed frequency of DOF 2", "J1"_a, "J2"_a);

  py::class_<PerturbedSystem>(m, "PerturbedSystem")
      .def_readonly("a", &PerturbedSystem::a)
      .def_readonly("epsilon_vector", &PerturbedSystem::epsilon_vector)
      .def_readonly("n_vector", &PerturbedSystem::n_vector)
      .def_readonly("m_vector", &PerturbedSystem::m_vector)
      .def(py::init<double, const std::vector<double> &,
                    const std::vector<int> &, const std::vector<int> &>(),
           "a"_a, "epsilon_vector"_a, "n_vector"_a, "m_vector"_a)
      .def("eoms", &PerturbedSystem::eoms, "s"_a)
      .def("__call__", &PerturbedSystem::operator());

  m.def("get_poincare_points", &get_poincare_points,
        "Compute Poincare points for a given perturbed system", "system"_a,
        "initial_state"_a, "t_max"_a, "dt"_a, "theta_c"_a, "direction"_a,
        "delta_theta_max"_a = 5.0);

  m.def("get_orbit", &get_orbit,
        "Compute orbit for a given perturbed system", "system"_a,
        "initial_state"_a, "t_max"_a, "dt"_a);

  m.def("evolve", &evolve,
        "Evolve the system for a given perturbed system", "system"_a,
        "initial_state"_a, "t_end"_a, "dt"_a);
}
