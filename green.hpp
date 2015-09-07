#ifndef GREEN_HPP
#define GREEN_HPP

#include <armadillo>

#include "geometry.hpp"
#include "model.hpp"
#include "potential.hpp"
#include "util/gnuplot.hpp"
#include "util/inverse.hpp"

static inline void self_energy(const device_params & p, const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;
    using namespace std;

    // kinetic energy in source and drain
    auto E_s = E - phi.s();
    auto E_d = E - phi.d();

    // shortcuts
    double t12 = p.tc1 * p.tc1;
    double t22 = p.tc2 * p.tc2;

    // self energy
    Sigma_s = E_s * E_s - t12 - t22;
    Sigma_s = 0.5 * (E_s * E_s - t12 + t22 + sqrt(Sigma_s * Sigma_s + - 4 * t12 * t22)) / E_s;
    Sigma_d = E_d * E_d - t12 - t22;
    Sigma_d = 0.5 * (E_d * E_d - t12 + t22 + sqrt(Sigma_d * Sigma_d + - 4 * t12 * t22)) / E_d;

    // imaginary part must be negative
    Sigma_s.imag(-std::abs(Sigma_s.imag()));
    Sigma_d.imag(-std::abs(Sigma_d.imag()));
}

template<bool source>
static inline arma::cx_vec green_col(const device_params & p, const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;

    self_energy(p, phi, E, Sigma_s, Sigma_d);

    // build diagonal part of hamiltonian
    auto D = conv_to<cx_vec>::from(E - phi.twice);
    D(0)            -= Sigma_s;
    D(D.size() - 1) -= Sigma_d;

    return inverse_col<source>(vec(-p.t_vec), D);
}

static inline arma::mat get_lDOS(const device_params & p, const potential & phi, int N_grid, arma::vec & E) {
    using namespace arma;
    using namespace std::complex_literals;

    mat ret(N_grid, p.N_x);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_grid; ++i) {
        cx_double Sigma_s;
        cx_double Sigma_d;
        self_energy(p, phi, E(i), Sigma_s, Sigma_d);

        auto D = conv_to<cx_vec>::from(E(i) - phi.twice);
        D(0)            -= Sigma_s;
        D(D.size() - 1) -= Sigma_d;
        D += 0.001i;

        vec mixed = -arma::imag(inverse_diag(vec(-p.t_vec), D)) / M_PI;

        for (int j = 0; j < p.N_x; ++j) {
            ret(i, j) = mixed(2*j) + mixed(2*j+1);
        }
    }
    return ret;
}

static inline void plot_ldos(const device_params & p, const potential & phi, const unsigned N_grid = 500, double Emin = -1, double Emax = +1) {
    gnuplot gp;

    arma::vec E = arma::linspace(Emin, Emax, N_grid);

//    gp << "set title \"Logarithmic lDOS\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"E / eV\"\n";
    gp << "set zlabel \"log(lDOS)\"\n";
    gp << "unset key\n";
    gp << "unset colorbox\n";

    //gp << "set terminal pdf size 4,3 font 'arial,12'\nset output 'lDOS.pdf'\n";

    arma::mat lDOS = get_lDOS(p, phi, N_grid, E);
    gp.set_background(p.x, E, arma::log(lDOS));

    arma::vec vband = phi.data - 0.5 * p.E_g;
    vband.rows(0, p.N_sc) += 0.5 * (p.E_g - p.E_gc);
    vband.rows(p.N_x - p.N_dc - 2, p.N_x - 1) += 0.5 * (p.E_g - p.E_gc);

    arma::vec cband = phi.data + 0.5 * p.E_g;
    cband.rows(0, p.N_sc) -= 0.5 * (p.E_g - p.E_gc);
    cband.rows(p.N_x - p.N_dc - 2, p.N_x - 1) -= 0.5 * (p.E_g - p.E_gc);

    gp.add(p.x, vband);
    gp.add(p.x, cband);
    gp.add(p.x, phi.data);

    unsigned N_s = std::round(p.N_sc + 0.5 * p.N_sox);
    arma::vec fermi_l(N_s);
    fermi_l.fill(p.F[S] + phi.s());
    arma::vec x_l = p.x(arma::span(0, N_s-1));
    gp.add(x_l, fermi_l);

    unsigned N_d = std::round(p.N_dc + 0.5 * p.N_dox);
    arma::vec fermi_r(N_d);
    fermi_r.fill(p.F[D] + phi.d());
    arma::vec x_r = p.x(arma::span(p.N_x-N_d, p.N_x-1));
    gp.add(x_r, fermi_r);

    gp << "set style line 1 lt 1 lc rgb RWTH_Orange lw 2\n";
    gp << "set style line 2 lt 1 lc rgb RWTH_Orange lw 2\n";
    gp << "set style line 3 lt 1 lc rgb RWTH_Rot lw 2\n";
    gp << "set style line 4 lt 3 lc rgb RWTH_Schwarz lw 2\n";
    gp << "set style line 5 lt 3 lc rgb RWTH_Schwarz lw 2\n";

    double phi_naut = phi.data(p.g.a + p.N_g / 2); // middle of gate
    double indicator_max = phi_naut + .85 * p.E_g;
    double indicator_min = phi_naut - .85 * p.E_g;

    gp << "set arrow from " << p.x(p.g.a) << "," << indicator_min << " to " << p.x(p.g.a) << "," << indicator_max << " nohead front lt 2 lc rgb RWTH_Schwarz_75\n";
    gp << "set arrow from " << p.x(p.g.b) << "," << indicator_min << " to " << p.x(p.g.b) << "," << indicator_max << " nohead front lt 2 lc rgb RWTH_Schwarz_75\n";

    gp.plot();
}

#endif
