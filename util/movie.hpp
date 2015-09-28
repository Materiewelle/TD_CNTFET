#ifndef MOVIE_HPP
#define MOVIE_HPP

#include <iomanip>
#include <vector>
#include <algorithm>
//#include "device.hpp"
//#include "charge_density.hpp"


class movie {
public:
    // provide an initialized device object with solved steady-state
    inline movie(device & dev, const std::vector<std::pair<int, int>> & E_i, int skip);

    static inline std::vector<std::pair<int, int>> around_Ef(const device & d, double dist);

private:
    int frames; // the current number of frames that have been produced

    static constexpr double phimin = -1.5;
    static constexpr double phimax = +1.0;

    device & d;
    std::vector<std::pair<int, int>> E_ind;

    gnuplot gp, gp2D;
    arma::vec band_offset;

    int frame_skip = 1;

    // 2D frame stuff
    arma::mat l_real;
    arma::mat both_real;
    arma::mat density;
    static constexpr int nE = 350;
    double E_min, E_max;
    double dE;
    arma::vec E;

    inline void frame();
    inline void frame2D();

    inline std::string output_folder(int lattice, double E0);
    inline std::string output_file(int lattice, double E0, int frame_number);

    static inline const std::string & lattice_name(int i);
};

movie::movie(device & dev, const std::vector<std::pair<int, int> > &E_i, int skip)
    : frames(0), d(dev), E_ind(E_i), band_offset(d.p.N_x), frame_skip(skip) {

    // get energy range
    E_min = -1;//charge_density::E_min + std::min(d.phi[0].s(), d.phi[0].d());
    E_max = +1;//charge_density::E_max + std::max(d.phi[0].s(), d.phi[0].d());
    dE = (E_max - E_min) / nE;
    E = arma::linspace(E_min, E_max, nE);
    l_real = arma::mat(nE, d.p.N_x);
    both_real = arma::mat(nE, d.p.N_x);
    density = arma::mat(nE, d.p.N_x);


    // produce folder tree
    for (unsigned i = 0; i < E_ind.size(); ++i) {
        int lattice = E_ind[i].first;
        double E = d.psi[lattice].E0(E_ind[i].second);
        system("mkdir -p " + output_folder(lattice, E));
        system("mkdir -p " + save_folder() + "/" + d.name + "/2D_movie/dos");
        system("mkdir -p " + save_folder() + "/" + d.name + "/2D_movie/left");
        system("mkdir -p " + save_folder() + "/" + d.name + "/2D_movie/current");
    }

    // gnuplot setup
    gp << "set terminal pngcairo size 1024,768 font 'arial,16'\n";
    gp << "set style line 66 lc rgb RWTH_Schwarz_50 lt 1 lw 2\n";
    gp << "set border 3 ls 66\n";
    gp << "set tics nomirror\n";
    gp << "set style line 1 lc rgb RWTH_Gruen\n";
    gp << "set style line 2 lc rgb RWTH_Rot\n";
    gp << "set style line 3 lc rgb RWTH_Blau\n";
    gp << "set style line 4 lc rgb RWTH_Blau\n";

    gp2D << "set xlabel \"x / nm\"\n";
    gp2D << "set ylabel \"E / eV\"\n";
    gp2D << "unset key\n";
    gp2D << "unset colorbox\n";
    gp2D << "set terminal pngcairo size 600,450 font 'arial,16'\n";
    gp2D << "set yrange [-1:.7]\n";
    gp2D << "set style line 1 lc rgb RWTH_Schwarz\n";
    gp2D << "set style line 2 lc rgb RWTH_Schwarz\n";

    // band offsets for band drawing
    band_offset.fill(0.5 * d.p.E_g);
    band_offset.rows(0, d.p.N_sc) -= 0.5 * (d.p.E_g - d.p.E_gc);
    band_offset.rows(d.p.N_x - d.p.N_dc - 2, d.p.N_x - 1) -= 0.5 * (d.p.E_g - d.p.E_gc);

    d.add_callback([this] () {
        if (((d.m - 1) % frame_skip) == 0) {
            this->frame();
            this->frame2D();
            std::cout << "movie-frame in this step" << std::endl;
            ++frames;
        }
    });
}

std::vector<std::pair<int, int>> movie::around_Ef(const device & d, double dist) {
    std::vector<std::pair<int, int>> E_ind(1);

    // in which band does Ef lie for this device?
    E_ind[0].first = d.p.F[S] < 0 ? LV : LC;
    // find energy closest to Ef (the last below is taken)
    auto begin = d.E0[E_ind[0].first].begin();
    auto end   = d.E0[E_ind[0].first].end();
    E_ind[0].second = std::lower_bound(begin, end, d.phi[0].s() + d.p.F[S] + dist) - begin;

//    // waves comming from right:
//    E_ind[1].first = d.p.F[D] < 0 ? RV : RC;
//    begin = d.E0[E_ind[1].first].begin();
//    end   = d.E0[E_ind[1].first].end();
//    E_ind[1].second = std::lower_bound(begin, end, d.phi[0].d() + d.p.F[D] + dist) - begin;

    return E_ind;
}

void movie::frame() {
    using namespace arma;
    for (unsigned i = 0; i < E_ind.size(); ++i) {
        int lattice = E_ind[i].first;
        double E = d.psi[lattice].E0(E_ind[i].second);

        // this is a line that indicates the wave's injection energy
        vec E_line(d.p.N_x);
        //            E_line = d.psi[lattice].E.col(E_ind[i].second);
        E_line.fill(d.psi[lattice].E0(E_ind[i].second));

        // set correct output file
        gp << "set output \"" << output_file(lattice, E, frames) << "\"\n";
        gp << "set multiplot layout 1,2 title 't = " << std::setprecision(3) << std::fixed << (d.m - 1) * c::dt * 1e12 << " ps'\n";

        arma::vec data[7];

        arma::cx_vec wavefunction = d.psi[lattice].data->col(E_ind[i].second);
        data[0] = arma::real(wavefunction);
        data[1] = arma::imag(wavefunction);
        data[2] = +arma::abs(wavefunction);
        data[3] = -arma::abs(wavefunction);

        data[4] = d.phi[d.m - 1].data - band_offset;
        data[5] = d.phi[d.m - 1].data + band_offset;
        data[6] = E_line;

        // setup psi-plot:
        gp << "set xlabel 'x / nm'\n";
        gp << "set key top right\n";
        gp << "set ylabel '{/Symbol Y}'\n";
        gp << "set yrange [-3:3]\n";
        gp << "p "
              "'-' w l ls 1 lw 2 t 'real', "
              "'-' w l ls 2 lw 2 t 'imag', "
              "'-' w l ls 3 lw 2 t 'abs', "
              "'-' w l ls 3 lw 2 notitle\n";

        // pipe data to gnuplot
        for (unsigned p = 0; p < 7; ++p) {
            for(int k = 0; k < d.p.N_x; ++k) {
                if (p == 4) { // setup bands-plot
                    gp << "set ylabel 'E / eV'\n";
                    gp << "set yrange [" << phimin << ":" << phimax << "]\n";
                    gp << "p "
                          "'-' w l ls 3 lw 2 notitle, "
                          "'-' w l ls 3 lw 2 t 'band edges', "
                          //                              "'-' w l ls 2 lw 2 t '<E_{{/Symbol Y}}>(x)'\n";
                          "'-' w l ls 2 lw 2 t 'E_{inj}'\n";
                }
                gp << d.p.x(k) << " " << ((p < 4) ? data[p](2 * k) : data[p](k)) << std::endl;
                }
                gp << "e" << std::endl;
            }
            gp << "unset multiplot\n";
        }
    std::flush(gp);
}

void movie::frame2D() {
    using namespace arma;
//    gp2D << "set colorbox\n";

    l_real.zeros();
    both_real.zeros();
    density.zeros();

    // loop over energy grid in left valence band
    for (ulint iE = 0; iE < d.psi[LV].E0.size(); ++iE) {
        // get energy index in new grid
        int ind = std::floor((d.psi[LV].E0[iE] + d.phi[d.m - 1].s() - d.phi[0].s() - E_min) / dE);
        if (ind < 0 || ind >= nE) continue;
        for (int ix = 0; ix < d.p.N_x; ++ix) {
             // sort wavefunction into correct bin
            cx_double orb1 = (*d.psi[LV].data)(2*ix, iE);
            cx_double orb2 = (*d.psi[LV].data)(2*ix+1, iE);
            l_real(ind, ix) += real(orb1) * d.psi[LV].W(iE);
            both_real(ind, ix) += sqrt(real(orb1)*real(orb1)) * d.psi[LV].W(iE) * d.psi[LV].F0[iE];
            density(ind, ix) += (abs(orb1)*abs(orb1) + abs(orb2)*abs(orb2)) * d.psi[LV].W(iE);
        }
    }
    // loop over energy grid in left conduction band
    for (ulint iE = 0; iE < d.psi[LC].E0.size(); ++iE) {
        int ind = std::floor((d.psi[LC].E0[iE] + d.phi[d.m - 1].s() - d.phi[0].s() - E_min) / dE);
        if (ind < 0 || ind >= nE) continue;
        for (int ix = 0; ix < d.p.N_x; ++ix) {
            cx_double orb1 = (*d.psi[LC].data)(2*ix, iE);
            cx_double orb2 = (*d.psi[LC].data)(2*ix+1, iE);
            l_real(ind, ix) += real(orb1) * d.psi[LC].W(iE);
            both_real(ind, ix) += sqrt(real(orb1)*real(orb1)) * d.psi[LC].W(iE) * d.psi[LC].F0[iE];
            density(ind, ix) += (abs(orb1)*abs(orb1) + abs(orb2)*abs(orb2)) * d.psi[LC].W(iE);
        }
    }
    // loop over energy grid in right valence band
    for (ulint iE = 0; iE < d.psi[RV].E0.size(); ++iE) {
        int ind = std::floor((d.psi[RV].E0[iE] + d.phi[d.m - 1].d() - d.phi[0].d() - E_min) / dE);
        if (ind < 0 || ind >= nE) continue;
        for (int ix = 0; ix < d.p.N_x; ++ix) {
            cx_double orb1 = (*d.psi[RV].data)(2*ix, iE);
            cx_double orb2 = (*d.psi[RV].data)(2*ix+1, iE);
            both_real(ind, ix) -= sqrt(real(orb1)*real(orb1)) * d.psi[RV].W(iE) * d.psi[RV].F0[iE];
            density(ind, ix) += (abs(orb1)*abs(orb1) + abs(orb2)*abs(orb2)) * d.psi[RV].W(iE);
        }
    }
    // loop over energy grid in right conduction band
    for (ulint iE = 0; iE < d.psi[RC].E0.size(); ++iE) {
        int ind = std::floor((d.psi[RC].E0[iE] + d.phi[d.m - 1].d() - d.phi[0].d() - E_min) / dE);
        if (ind < 0 || ind >= nE) continue;
        for (int ix = 0; ix < d.p.N_x; ++ix) {
            cx_double orb1 = (*d.psi[RC].data)(2*ix, iE);
            cx_double orb2 = (*d.psi[RC].data)(2*ix+1, iE);
            both_real(ind, ix) -= sqrt(real(orb1)*real(orb1)) * d.psi[RC].W(iE) * d.psi[RC].F0[iE];
            density(ind, ix) += (abs(orb1)*abs(orb1) + abs(orb2)*abs(orb2)) * d.psi[RC].W(iE);
        }
    }

    // --------------- plot -----------------
    gp2D << "set title 't = " << std::setprecision(3) << std::fixed << (d.m - 1) * c::dt * 1e12 << " ps'\n";
    std::stringstream ss;

    gp2D << "set pal gray\n";

    // density
    gp2D.reset();
    ss << save_folder() << "/" << d.name << "/2D_movie/dos/" << std::setfill('0') << std::setw(4) << frames << ".png";
    gp2D << "set output \"" << ss.str() << "\"\n";
    gp2D.set_background(d.p.x, E, log(density + 1e-6));
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data - band_offset));
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data + band_offset));
    gp2D << "set cbrange [-6.5:-4]\n";
    gp2D.plot();
    gp2D.flush();

    gp2D << "set palette defined(0 RWTH_Blau, 1 '#FFFFFFFF', 2 RWTH_Rot)\n";

    // left
    gp2D.reset();
    ss.str("");
    ss << save_folder() << "/" << d.name << "/2D_movie/left/" << std::setfill('0') << std::setw(4) << frames << ".png";
    gp2D << "set output \"" << ss.str() << "\"\n";
    gp2D << "set cbrange [-.01:+.01]\n";
    gp2D.set_background(d.p.x, E, l_real);
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data - band_offset));
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data + band_offset));
    gp2D.plot();
    gp2D.flush();

    // both
    gp2D.reset();
    ss.str("");
    ss << save_folder() << "/" << d.name << "/2D_movie/current/" << std::setfill('0') << std::setw(4) << frames << ".png";
    gp2D << "set output \"" << ss.str() << "\"\n";
    gp2D << "set cbrange [-.01:+.01]\n";
    gp2D.set_background(d.p.x, E, both_real);
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data - band_offset));
    gp2D.add(std::make_pair(d.p.x, d.phi[d.m - 1].data + band_offset));
    gp2D.plot();
    gp2D.flush();

}

std::string movie::output_file(int lattice, double E0, int frame_number) {
    std::stringstream ss;
    ss << output_folder(lattice, E0) << "/" << std::setfill('0') << std::setw(4) << frame_number << ".png";
    return ss.str();
}

std::string movie::output_folder(int lattice, double E0) {
    std::stringstream ss;
    ss << save_folder() << "/" << d.name << "/" << lattice_name(lattice) << "/E0=" << std::setprecision(2) << E0 << "eV";
    return ss.str();
}

const std::string & movie::lattice_name(int i) {
    static const std::string names[6] = {
        "left_valence",
        "right_valence",
        "left_conduction",
        "right_conduction",
        "left_tunnel",
        "right_tunnel"
    };
    return names[i];
}

#endif

