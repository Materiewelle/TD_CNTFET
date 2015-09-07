#ifndef CHARGE_DENSITY_HPP_HEADER
#define CHARGE_DENSITY_HPP_HEADER

#include <armadillo>
#include <stack>

#include "device_params.hpp"
#include "util/fermi.hpp"
#include "util/integral.hpp"

// forward declarations
class potential;
class wave_packet;
template<bool source>
static inline arma::cx_vec green_col(const device_params & p, const potential &, double, arma::cx_double &, arma::cx_double &);

class charge_density {
public:
    // adaptive integration parameters
    static constexpr double E_min = -1.5;         // relative to potential
    static constexpr double E_max = +1.5;         // relative to potential
    static constexpr double rel_tol = 5e-3;       // subdivide an interval as long as the relative deviation is larger
    static constexpr int initial_waypoints = 300; // divide into at least this number of smaller intervalls

    arma::vec lv;    // density due to states in v-band, induced from left
    arma::vec rv;    // density due to states in v-band, induced from right
    arma::vec lc;    // density due to states in c-band, induced from left
    arma::vec rc;    // density due to states in c-band, induced from right
    arma::vec total; // total density

    inline charge_density();
    inline charge_density(const device_params & p, const potential & phi, arma::vec E0[4], arma::vec W[4]);
    inline charge_density(const device_params & p, const potential & phi, const wave_packet psi[4]);

private:
    static inline arma::vec get_bound_states(const device_params & p, const potential & phi);
    static inline arma::vec get_bound_states(const device_params & p, const potential & phi, double E0, double E1);
    static inline arma::vec get_intervals(const arma::vec & E_bound, double E0, double E1);

    template<bool source>
    static inline arma::vec get_A(const device_params & p, const potential & phi, double E);

public:    static inline arma::vec get_n0(const device_params & p);
};

#endif

//----------------------------------------------------------------------------------------------------------------------

#ifndef CHARGE_DENSITY_HPP_BODY
#define CHARGE_DENSITY_HPP_BODY

charge_density::charge_density() {
}

charge_density::charge_density(const device_params & p, const potential & phi, arma::vec E0[4], arma::vec W[4]) {
    using namespace arma;

    // get a prediction of bound-state levels in the channel region
    vec E_bound = get_bound_states(p, phi);

    // get integration intervals
    vec i_lv = get_intervals(E_bound, phi.s() + E_min, phi.s() - 0.5 * p.E_gc);
    vec i_rv = get_intervals(E_bound, phi.d() + E_min, phi.d() - 0.5 * p.E_gc);
    vec i_lc = get_intervals(E_bound, phi.s() + 0.5 * p.E_gc, phi.s() + E_max);
    vec i_rc = get_intervals(E_bound, phi.d() + 0.5 * p.E_gc, phi.d() + E_max);

    // integrand function definitions
    auto I_l = [&] (double E) -> vec {
        // spectral function from source
        vec A = get_A<true>(p, phi, E);

        // fermi distribution in source
        double f = fermi(E - phi.s(), p.F[S]);

        // multiply with:
        // f if counted as electron
        // or with
        // (f - 1) if counted as hole
        for (int i = 0; i < p.N_x; ++i) {
            A(i) *= fermi<true>(f, E - phi(i));
        }

        return A;
    };
    auto I_r = [&] (double E) -> vec {
        // spectral function from drain
        vec A = get_A<false>(p, phi, E);

        // fermi distribution in drain
        double f = fermi(E - phi.d(), p.F[D]);

        // see above...
        for (int i = 0; i < p.N_x; ++i) {
            A(i) *= fermi<true>(f, E - phi(i));
        }

        return A;
    };

    // scale
    double scale = - c::e / M_PI;

    // calculate integrals (adaptively) and output the energy lattice and weights that were used
    lv = scale * integral(I_l, p.N_x, i_lv, rel_tol, c::epsilon(), E0[LV], W[LV]);
    rv = scale * integral(I_r, p.N_x, i_rv, rel_tol, c::epsilon(), E0[RV], W[RV]);
    lc = scale * integral(I_l, p.N_x, i_lc, rel_tol, c::epsilon(), E0[LC], W[LC]);
    rc = scale * integral(I_r, p.N_x, i_rc, rel_tol, c::epsilon(), E0[RC], W[RC]);

    // calculate total charge density and add the background charge introduced by dopants
    total = lv + rv + lc + rc + get_n0(p);
}
charge_density::charge_density(const device_params & p, const potential & phi, const wave_packet psi[4]) {
    using namespace arma;

    // calculate charge_density for specific wave_packet
    auto get_n = [&p, &phi] (const wave_packet & psi) -> vec {
        // initialize return vector
        vec n(p.N_x);
        n.fill(0.0);

        #pragma omp parallel
        {
            // each thread gets its own container to work with
            vec n_thread(n.size());
            n_thread.fill(0.0);

            // loop over all energies
            #pragma omp for schedule(static) nowait
            for (unsigned i = 0; i < psi.E0.size(); ++i) {
                // load fermi factor and weight
                double f = psi.F0(i);
                double W = psi.W(i);

                // loop over all unit cells
                for (int j = 0; j < p.N_x; ++j) {
                    // calculate norm for both orbitals
                    double a = std::norm((*psi.data)(2 * j    , i));
                    double b = std::norm((*psi.data)(2 * j + 1, i));

                    // add to thread specific vector
                    n_thread(j) += (a + b) * W * fermi<true>(f, psi.E(j, i) - phi(j));
                }
            }
            // no implied barrier (nowait clause)

            // add everything together
            #pragma omp critical
            {
                n += n_thread;
            }
        }

        return n;
    };

    // scaling
    double scale = - c::e / M_PI;

    // calculate charge_density for each wave_packet
    lv = scale * get_n(psi[LV]);
    rv = scale * get_n(psi[RV]);
    lc = scale * get_n(psi[LC]);
    rc = scale * get_n(psi[RC]);

    // calculate total charge density and add the background charge introduced by dopants
    total = lv + rv + lc + rc + get_n0(p);
}

arma::vec charge_density::get_bound_states(const device_params & p, const potential & phi) {
    double phi0, phi1, phi2, limit;

    // check for bound states in valence band
    phi0 = arma::min(phi.data(p.sg)) - 0.5 * p.E_g;
    phi1 = arma::max(phi.data(p.g )) - 0.5 * p.E_g;
    phi2 = arma::min(phi.data(p.dg)) - 0.5 * p.E_g;
    limit = phi0 > phi2 ? phi0 : phi2;
    if (limit < phi1) {
        return get_bound_states(p, phi, limit, phi1);
    }

    // check for bound states in conduction band
    phi0 = arma::max(phi.data(p.sg)) + 0.5 * p.E_g;
    phi1 = arma::min(phi.data(p.g )) + 0.5 * p.E_g;
    phi2 = arma::max(phi.data(p.dg)) + 0.5 * p.E_g;
    limit = phi0 < phi2 ? phi0 : phi2;
    if (limit > phi1) {
        return get_bound_states(p, phi, phi1, limit);
    }

    // no bound states found
    return arma::vec(arma::uword(0));
}

arma::vec charge_density::get_bound_states(const device_params & p, const potential & phi, double E0, double E1) {
    using namespace arma;

    // absolute minimal precision of eigenvalue-finding (in eV)
    static constexpr double tol = 1e-10;

    // only look at the gated region
    span range { p.sg2.a, p.dg2.b };

    // for building the hamiltonian only in the gated region
    vec a  = p.t_vec(range);   // off-diagonal
    vec a2 = a % a;            // off-diagonal squared
    vec b  = phi.twice(range); // main diagonal

    // evaluate Sturm sequence to find number of eigenvalues smaller than E
    auto eval = [&] (double E) {
        int n = b.size();

        static const double eps = c::epsilon();

        // first iteration (i = 0)
        double q;
        double q0 = b[0] - E;
        int s = q0 < 0 ? 1 : 0;

        // start with i = 1
        for (int i = 1; i < n; ++i) {
            if (q0 == 0) {
                q = b[i] - E - a[i - 1] / eps;
            } else {
                q = b[i] - E - a2[i - 1] / q0;
            }

            q0 = q;
            if (q < 0) {
                ++s;
            }
        }

        return s;
    };

    double E2;
    int i0, i1;
    int s0, s1, s2;

    // s1 - s0 is the number of eigenvalues in the interval
    s0 = eval(E0);
    s1 = eval(E1);

    // check if no bound states in this interval
    if (s1 - s0 == 0) {
        return vec(uword(0));
    }

    // start iterative bisection method to find all eigenvalues in the interval
    unsigned n = 2;
    vec E(1025);
    ivec s(1025);
    E(0) = E0;
    E(1) = E1;
    s(0) = s0;
    s(1) = s1;

    unsigned n_bound = 0;
    vec E_bound(100);

    // stack for recursion
    std::stack<std::pair<int, int>> stack;

    // push first interval to stack
    stack.push(std::make_pair(0, 1));

    // repeat until all intervals inspected
    while (!stack.empty()) {
        const auto & i = stack.top();
        i0 = i.first;
        i1 = i.second;

        stack.pop();

        // load data
        E0 = E(i0);
        E1 = E(i1);
        s0 = s(i0);
        s1 = s(i1);

        // mid energy
        E2 = 0.5 * (E0 + E1);

        // if interval size sufficiently small enough, add new bound state (can be degenerate eigenvalue)
        if (E1 - E0 <= tol) {
            if (E_bound.size() <= n_bound) {
                E_bound.resize(n_bound * 2);
            }
            E_bound(n_bound++) = E2;
        } else {
            // evaluate s at mid energy
            s2 = eval(E2);

            // add intervals to stack if they contain bound states
            if (s1 - s2 > 0) {
                stack.push(std::make_pair(n, i1));
            }
            if (s2 - s0 > 0) {
                stack.push(std::make_pair(i0, n));
            }

            // save E2 and s2
            if (E.size() <= n) {
                E.resize(2 * n - 1);
                s.resize(2 * n - 1);
            }
            E(n) = E2;
            s(n) = s2;
            ++n;
        }
    }

    // shrink to fit and return bound states
    E_bound.resize(n_bound);
    return E_bound;
}

arma::vec charge_density::get_intervals(const arma::vec & E_bound, double E0, double E1) {
    using namespace arma;

    // create linear lattice
    vec lin = linspace(E0, E1, initial_waypoints);

    // merge bound-state positions and linear lattice together
    if ((E_bound.size() > 0) && (E_bound(0) < E1) && (E_bound(E_bound.size() - 1) > E0)) {
        vec ret = vec(E_bound.size() + lin.size());

        // get iterator to lin
        auto i0 = std::begin(lin);

        // get lowest bound state that is bigger than E0
        auto i1 = std::lower_bound(std::begin(E_bound), std::end(E_bound), E0);

        // index to ret vector
        unsigned j = 0;

        // merge lin and E_bound
        while ((i0 != std::end(lin)) && (i1 != std::end(E_bound))) {
            if (*i0 < *i1) {
                ret(j++) = *(i0++);
            } else {
                ret(j++) = *(i1++);
            }
        }

        // add rest of lin, discard rest of E_bound (out of range)
        while (i0 != std::end(lin)) {
            ret(j++) = *(i0++);
        }

        // shrink to fit and return
        ret.resize(j);
        return ret;
    }

    // no bound states in interval [E0, E1]
    return lin;
}

template<bool source>
arma::vec charge_density::get_A(const device_params & p, const potential & phi, double E) {
    using namespace arma;

    // calculate 1 column of green's function
    cx_double Sigma_s, Sigma_d;
    cx_vec G = green_col<source>(p, phi, E, Sigma_s, Sigma_d);

    // get spectral function for each orbital (2 values per unit cell)
    vec A_twice;
    if (source) {
        A_twice = std::abs(2 * Sigma_s.imag()) * real(G % conj(G)); // G .* conj(G) = abs(G).^2
    } else {
        A_twice = std::abs(2 * Sigma_d.imag()) * real(G % conj(G));
    }

    // reduce spectral function to 1 value per unit cell (simple addition of both values)
    vec A = vec(A_twice.size() / 2);
    for (unsigned i = 0; i < A.size(); ++i) {
        A(i) = A_twice(2 * i) + A_twice(2 * i + 1);
    }

    return A;
}

arma::vec charge_density::get_n0(const device_params & p) {
    using namespace arma;

    // check if n0 has already been calculated for this device
    static thread_local std::map<std::string, arma::vec> n0;
    auto it = n0.find(p.name);
    if (it != std::end(n0)) {
        return it->second;
    }

    // calculate new n0
    vec x0, x1, x2, x3, w0, w1, w2, w3;

    // valence band in contact region
    vec nvc = integral([&] (double E) {
        double dos = 4 * E / M_PI / sqrt(4*p.tc1*p.tc1*p.tc2*p.tc2 - (E*E - p.tc1*p.tc1 - p.tc2*p.tc2) * (E*E - p.tc1*p.tc1 - p.tc2*p.tc2));
        vec ret = arma::vec(2);
        ret(0) = (1 - fermi(E, p.F[S])) * dos;
        ret(1) = (1 - fermi(E, p.F[D])) * dos;
        return ret;
    }, 2, linspace(E_min, -0.5 * p.E_gc, 100), rel_tol, c::epsilon(), x0, w0);

    // conduction band in contact region
    vec ncc = integral([&] (double E) {
        double dos = 4 * E / M_PI / sqrt(4*p.tc1*p.tc1*p.tc2*p.tc2 - (E*E - p.tc1*p.tc1 - p.tc2*p.tc2) * (E*E - p.tc1*p.tc1 - p.tc2*p.tc2));
        vec ret = arma::vec(2);
        ret(0) = fermi(E, p.F[S]) * dos;
        ret(1) = fermi(E, p.F[D]) * dos;
        return ret;
    }, 2, linspace(0.5 * p.E_gc, E_max, 100), rel_tol, c::epsilon(), x1, w1);

    // valence band in central region
    vec nvsgd = integral([&] (double E) {
        double dos = 4 * E / M_PI / sqrt(4*p.t1*p.t1*p.t2*p.t2 - (E*E - p.t1*p.t1 - p.t2*p.t2) * (E*E - p.t1*p.t1 - p.t2*p.t2));
        vec ret = arma::vec(3);
        ret(0) = (1 - fermi(E, p.F[S])) * dos;
//        ret(1) = (1 - fermi(E, p.F[G])) * dos;
        ret(1) = (1 - fermi(E, 0)) * dos;
        ret(2) = (1 - fermi(E, p.F[D])) * dos;
        return ret;
    }, 3, linspace(E_min, - 0.5 * p.E_g, 100), rel_tol, c::epsilon(), x2, w2);

    // conduction band in central region
    vec ncsgd = integral([&] (double E) {
        double dos = 4 * E / M_PI / sqrt(4*p.t1*p.t1*p.t2*p.t2 - (E*E - p.t1*p.t1 - p.t2*p.t2) * (E*E - p.t1*p.t1 - p.t2*p.t2));
        vec ret = arma::vec(3);
        ret(0) = fermi(E, p.F[S]) * dos;
//        ret(1) = fermi(E, p.F[G]) * dos;
        ret(1) = fermi(E, 0) * dos;
        ret(2) = fermi(E, p.F[D]) * dos;
        return ret;
    }, 3, linspace(0.5 * p.E_g, E_max, 100), rel_tol, c::epsilon(), x3, w3);

    // total charge density in contact regions
    vec nc = nvc + ncc;
    // total charge density in central region
    vec nsgd = nvsgd + ncsgd;

    vec ret(p.N_x);
    ret.rows(0, p.sox.a).fill(nc(0));
    if (p.sg.a >= p.sox.a + 1) {
        ret.rows(p.sox.a + 1, p.sg.a).fill(nsgd(0));
    }
    ret.rows(p.sg.a + 1, p.sg.b).fill(0.0);
    ret.rows(p.g.a, p.dg.a).fill(nsgd(1));
    ret.rows(p.dg.a + 1, p.dg.b).fill(0.0);
    if (p.dox.b >= p.dox.a) {
        ret.rows(p.dox.a, p.dox.b).fill(nsgd(2));
    }
    ret.rows(p.dc.a, p.N_x - 1).fill(nc(1));

    ret *= c::e; // /dx*dx omitted

    // save and return n0
    n0[p.name] = ret;
    return n0[p.name];
}

#endif
