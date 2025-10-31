cmake_minimum_required(VERSION 3.15...3.19)
include(CMakePrintHelpers)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

find_package(BLAS REQUIRED)



find_package(GSL REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)


message(STATUS "BLAS_LIBRARIES ${BLAS_LIBRARIES}")
# find_package (Boost 1.79.0 COMPONENTS math_tr1 REQUIRED) find_package (Boost
# 1.79.0 REQUIRED)


# specify the c++ source files
file(GLOB PACKAGE_CPP_IMPLEMENTATION
  "src/q_kinetic_minimum/cpp_implementation/*.cpp"
)

pybind11_add_module(q_kin_cpp MODULE ${PACKAGE_CPP_IMPLEMENTATION})


target_compile_definitions(q_kin_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_compile_features(q_kin_cpp PUBLIC cxx_std_20)

set(COMPILE_OPTS
    -Wall
    -Wextra
    -Wshadow
    -Wnon-virtual-dtor
    -pedantic
    # -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    # -Wsign-conversion
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wnull-dereference
    # -Wuseless-cast
    -Wdouble-promotion
    -Wformat=2
    -static-libstdc++
    -fPIC
    # -frounding-math
    $<$<CONFIG:Debug>:
    -g
    >
    $<$<CONFIG:Release>:
    -O2
    # -march=native
    >)

target_compile_options(q_kin_cpp PRIVATE ${COMPILE_OPTS})

target_link_libraries(
  q_kin_cpp
  PUBLIC GSL::gsl
  Eigen3::Eigen
  PRIVATE ${BLAS_LIBRARIES})

install(TARGETS q_kin_cpp DESTINATION .)


