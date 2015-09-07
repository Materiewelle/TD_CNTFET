#ifndef CURRENT_HPP
#define CURRENT_HPP

#include <armadillo>

#include "device_params.hpp"
#include "potential.hpp"
#include "wave_packet.hpp"

class current {
public:
    arma::vec lv;    // current from left valence band
    arma::vec rv;    // current from right valence band
    arma::vec lc;    // current from left conduction band
    arma::vec rc;    // current from right conduction band
    arma::vec total; // total current

    inline current();
    inline current(const device_params & p, const potential & phi);
    inline current(const device_params & p, const potential & phi, const wave_packet psi[4]);

    inline double s() const;
    inline double d() const;
};

//----------------------------------------------------------------------------------------------------------------------

current::current() {
}

current::current(const device_params & p, const potential & phi)
    : lv(p.N_x), rv(p.N_x), lc(p.N_x), rc(p.N_x) {
    using namespace arma;

    // transmission probability
    auto transmission = [&] (double E) -> double {
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<false>(p, phi, E, Sigma_s, Sigma_d);
        return 4 * Sigma_s.imag() * Sigma_d.imag() * (std::norm(G(1)) + std::norm(G(2)));
    };

    static constexpr auto scale = 2.0 * c::e * c::e / c::h;

    auto i_v = linspace(phi.d() + charge_density::E_min, phi.d() - 0.5 * p.E_gc, 20);
    auto i_c = linspace(phi.d() + 0.5 * p.E_gc, phi.d() + charge_density::E_max, 20);

    auto I_s = [&] (double E) -> double {
        double ret = transmission(E) * scale;
        double f = fermi(E - phi.s(), p.F[S]);
        ret *= fermi<true>(f, E - phi.d());
        return ret;
    };

    auto I_d = [&] (double E) -> double {
        double ret = transmission(E) * scale;
        double f = fermi(E - phi.d(), p.F[D]);
        ret *= fermi<true>(f, E - phi.d());
        return ret;
    };

    vec E, W; // just dummy junk

    lv(0) = +integral(I_s, 1, i_v, charge_density::rel_tol, c::epsilon(1e-10), E, W)(0);
    lv.fill(lv(0));

    rv(0) = -integral(I_d, 1, i_v, charge_density::rel_tol, c::epsilon(1e-10), E, W)(0);
    rv.fill(rv(0));

    lc(0) = +integral(I_s, 1, i_c, charge_density::rel_tol, c::epsilon(1e-10), E, W)(0);
    lc.fill(lc(0));

    rc(0) = -integral(I_d, 1, i_c, charge_density::rel_tol, c::epsilon(1e-10), E, W)(0);
    rc.fill(rc(0));

    total = lv + rv + lc + rc;
}
current::current(const device_params & p, const potential & phi, const wave_packet psi[4]) {
    using namespace arma;

    // calculate current for specific wave_packet
    auto get_I = [&p, &phi] (const wave_packet & psi) -> vec {
        // initialize return vector
        vec I(p.N_x);
        I.fill(0.0);

        #pragma omp parallel
        {
            // each thread gets its own copy
            vec I_thread(I.size());
            I_thread.fill(0.0);

            // loop over all energies
            #pragma omp for schedule(static) nowait
            for (unsigned i = 0; i < psi.E0.size(); ++i) {
                // load fermi factor and weight
                double f = psi.F0(i);
                double W = psi.W(i);

                // a: current from cell j-1 to j; b: current from cell j to j+1
                double a;
                double b = p.t_vec(1) * (std::conj((*psi.data)(2, i)) * (*psi.data)(1, i)).imag();

                // first value: just take b (current from cell 0 to 1)
                I_thread(0) += b * W * fermi<true>(f, psi.E(0, i) - phi(0));

                // loop over all unit cells
                for (int j = 1; j < p.N_x - 1; ++j) {
                    // update a and b
                    a = b;
                    b = p.t_vec(j * 2 + 1) * (std::conj((*psi.data)(2 * j + 2, i)) * (*psi.data)(2 * j + 1, i)).imag();

                    // add average of a and b
                    I_thread(j) += 0.5 * (a + b) * W * fermi<true>(f, psi.E(j, i) - phi(j));
                }

                // last value: take old b (current from cell N_x - 2 to N_x - 1)
                I_thread(p.N_x - 1) += b * W * fermi<true>(f, psi.E(p.N_x - 1, i) - phi(p.N_x - 1));
            }
            // no implied barrier (nowait clause)

            // add everything together
            #pragma omp critical
            {
                I += I_thread;
            }
        }

        return I;
    };

    // scaling
    double scale = 4.0 * c::e * c::e / c::h_bar / M_PI;

    // calculate charge_density for each wave_packet
    lv = scale * get_I(psi[LV]);
    rv = scale * get_I(psi[RV]);
    lc = scale * get_I(psi[LC]);
    rc = scale * get_I(psi[RC]);

    // calculate total current
    total = lv + rv + lc + rc;
}

double current::s() const {
    return total[0];
}

double current::d() const {
    return total[total.size() - 1];
}

#endif

