#ifndef FERMI_HPP
#define FERMI_HPP

#include <armadillo>
#include <cmath>

inline double fermi(double E, double F) {
    // just the fermi distribution function for fermions
    return 1.0 / (1.0 + std::exp((E - F) * c::e / c::T / c::k_B));
}

inline arma::vec fermi(const arma::vec & E, double F) {
    // returns a vector of occupations for every energy in E
    using namespace arma;

    vec ret(E.size());
    for (unsigned i = 0; i < E.size(); ++i) {
        ret(i) = fermi(E(i), F);
    }

    return ret;
}

template<bool smooth>
inline double fermi(double f, double delta_E, double slope = 600) {
    /* the statistical distribution is not continuous at the branching point.
     * To prevent the adaptive integration scheme from creating a very high (and practically useless)
     * number of energy-levels/wavefunctions, smooth the distribution function around the branching-point energy. */
    if (std::isfinite(delta_E)) {
        if (smooth) {
            f -= 0.5 - 0.5 * std::tanh(delta_E * slope); // add a tangens hyperbolicus instead of a heaviside-funtion
        } else {
            f -= (delta_E < 0) ? 1.0 : 0.0;
        }
    } else {
        f = 0;
    }
    return f;
}

#endif
