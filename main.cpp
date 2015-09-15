#define ARMA_NO_DEBUG // no bound checks
//#define GNUPLOT_NOPLOTS

#include <armadillo>
#include <iostream>
#include <omp.h>
#include <xmmintrin.h>

#define CHARGE_DENSITY_HPP_BODY

#include "charge_density.hpp"
#include "circuit.hpp"
#include "constant.hpp"
#include "contact.hpp"
#include "current.hpp"
#include "device.hpp"
#include "device_params.hpp"
#include "geometry.hpp"
#include "green.hpp"
#include "inverter.hpp"
#include "model.hpp"
#include "potential.hpp"
#include "ring_oscillator.hpp"
#include "voltage.hpp"
#include "wave_packet.hpp"
#include "util/movie.hpp"

#undef CHARGE_DENSITY_HPP_BODY

#include "charge_density.hpp"

// -------- standard device params --------
static const geometry tfet_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox
    10.0, // l_sc
    20.0, // l_sox
     5.0, // l_sg
    30.0, // l_g
    30.0, // l_dg
     2.0, // l_dox
    10.0, // l_dc
     1.0, // r_cnt
     3.0, // d_ox
     2.0, // r_ext
     0.5, // dx
     0.1  // dr
};

static const model ntfetc_model { // most general model
    0.5,              // E_g
    0.04 * c::m_e,    // m_eff (Claus et al)
    0.30,             // E_gc  (Claus et al)
    0.10 * c::m_e,    // m_efc (Claus et al)
    {
        -0.5 / 2 - 0.02, // F[S] (p)
        +0.5 / 2 + 0.02, // F[D] (n)
         0.             // F[G]
    }
};

static const model ntfet_model {
    ntfetc_model.E_g,   // E_g
    ntfetc_model.m_eff, // m_eff
    ntfetc_model.E_g,   // same as E_g
    ntfetc_model.m_eff, // same as m_eff
    {
        ntfetc_model.F[S],
        ntfetc_model.F[D],
        ntfetc_model.F[G]
    }
};

static const model ptfet_model {
    ntfet_model.E_g,   // E_g
    ntfet_model.m_eff, // m_eff
    ntfet_model.E_gc,  // E_gc
    ntfet_model.m_efc, // m_efc
    { // reversed doping:
        -ntfet_model.F[S],
        -ntfet_model.F[D],
        -ntfet_model.F[G]
    }
};

static const geometry fet_geometry {
    tfet_geometry.eps_cnt,
    tfet_geometry.eps_ox,
    tfet_geometry.l_sc,
    tfet_geometry.l_sox,
    tfet_geometry.l_sg,
    tfet_geometry.l_g,
    tfet_geometry.l_sg,  //symmetrical
    tfet_geometry.l_sox, //symmetrical
    tfet_geometry.l_dc,
    tfet_geometry.r_cnt,
    tfet_geometry.d_ox,
    tfet_geometry.r_ext,
    tfet_geometry.dx,
    tfet_geometry.dr
};

static const model nfet_model {
    ntfet_model.E_g,
    ntfet_model.m_eff,
    ntfet_model.E_gc,
    ntfet_model.m_efc,
    {
        +ntfet_model.E_g / 2 + 0.02, // F[S] (n)
        +ntfet_model.E_g / 2 + 0.02, // F[D] (n)
         0.0             // F[G]
    }
};

static const device_params nfet("nfet", fet_geometry, nfet_model);
static const device_params ntfet("ntfet", tfet_geometry, ntfet_model);
static const device_params ntfetc("ntfetc", tfet_geometry, ntfetc_model);
static const device_params ptfet("ptfet", tfet_geometry, ptfet_model);

using namespace arma;
using namespace std;

double capacitance(const device_params & p) {
    double csg = c::eps_0 * M_PI * (p.R * p. R - (p.r_cnt + p.d_ox) * (p.r_cnt + p.d_ox)) / p.l_sg * 1e-9;
    double cdg = c::eps_0 * M_PI * (p.R * p. R - (p.r_cnt + p.d_ox) * (p.r_cnt + p.d_ox)) / p.l_dg * 1e-9;
    double czyl = 2 * M_PI * c::eps_0 * p.eps_ox * p.l_g / std::log((p.r_cnt + p.d_ox) / p.r_cnt) * 1e-9;

    cout << "C_sg  = " << csg << endl;
    cout << "C_dg  = " << cdg << endl;
    cout << "C_zyl = " << czyl << endl;
    double Cg = csg + cdg + czyl;
    cout << "C_g   = " << csg + cdg + czyl << endl;
    return Cg;
}

void to_CSV(const string infile) {
    mat in;
    in.load(infile);
    mat out = trans(in);
    out = out.col(out.n_cols - 1); // just I_d
    out.save(infile + ".csv", arma::csv_ascii);
}


//------- simulation routines ---------------

// global steady-state curve parameters:
static const double gvg0 = -.3;
static const double gvg1 = .5;
static const double gvgop = .4;
static const double gvd0 = 0.;
static const double gvd1 = .5;
static const double gvdop = .2;
static const int gN = 400;

void voltage_point(double vs, double vd, double vg) {
    device d("ntfet", ntfet, {vs, vd, vg});
    d.p.dx = .2; // for nicer figures
    d.p.update("small_dx");
    d.steady_state();
    cout << "I = " << d.I[0].total[0] << std::endl;
    plot(make_pair(d.p.x, d.phi[0].data));
    potential::plot2D(d.p, { 0, vd, vg }, d.n[0]);
    plot(make_pair(d.p.x, d.n[0].total));
    plot_ldos(d.p, d.phi[0], 1000, -1, .7);
}

void transfer_test(double vg0, double vg1, double vd, int N) {
    save_folder("tests");
    device d("ntfet", ntfet);
    transfer<true>(d.p, { { 0, vd, vg0 } }, vg1, N);
}

void output_test(double vd0, double vd1, double vg, int N) {
    save_folder("tests");
    device d("ntfet", ntfet);
    output<true>(d.p, { { 0, vd0, vg } }, vd1, N);
}

void scaling(double lg) {
    stringstream ss;
    ss << "scaling/lg=" << lg;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    d.p.l_g = lg;
    d.p.update("updated");
    transfer<true>(d.p, { { 0, gvdop, gvg0 } }, gvg1, gN);
}

void overlap(double gap) {
    stringstream ss;
    ss << "overlap/gap=" << gap;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    double l_total = d.p.l_sox + d.p.l_sg;
    d.p.l_sox = l_total - gap;
    d.p.l_sg = gap;
    d.p.update("updated");
    transfer<true>(d.p, { { 0, gvdop, gvg0 } }, gvg1, gN);
//    output<true>(d.p, { { 0, gvd0, gvgop } }, gvd1, gN);
}

void separation(double ldg) {
    stringstream ss;
    ss << "separation/ldg=" << ldg;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    d.p.l_dg = ldg;
    d.p.update("updated");
    transfer<true>(d.p, { { 0, gvdop, gvg0 } }, gvg1, gN);
}

void final_transfer(double vd) {
    stringstream ss;
    ss << "final/trans_vs=" << vd;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    transfer<true>(d.p, { { 0, vd, gvg0 } }, gvg1, gN);
}

void final_output(double vg) {
    stringstream ss;
    ss << "final/outp_vg=" << vg;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    output<true>(d.p, { { 0, gvd0, vg } }, gvd1, gN);
}

void contacts_transfer(double lsox) {
    stringstream ss;
    ss << "contacts/lsox=" << lsox;
    save_folder(ss.str());
    device d("ntfet", ntfetc);
    d.p.dx = .2; // we need broader bands in the contacts
    d.p.l_sox = lsox;
    d.p.update("contacts");
    transfer<true>(d.p, { { 0, gvdop, gvg0 } }, gvg1, gN);
}

void contacts_output(double lsox) {
    stringstream ss;
    ss << "contacts/lsox=" << lsox;
    save_folder(ss.str());
    device d("ntfet", ntfetc);
    d.p.dx = .2; // we need broader bands in the contacts
    d.p.l_sox = lsox;
    d.p.update("contacts");
    output<true>(d.p, { { 0, gvd0, gvgop } }, gvd1, gN);
}

void ntd_inverter(int part) {
    double N = 40;

    stringstream ss;
    ss << "ntd_inverter";
    save_folder(ss.str());

    // build inverter from matched devices
    device n("ntfet", ntfet);
    n.p.F[G] = .2; //match
    n.p.update("matched_n");
    device p("ptfet", ptfet);
    p.p.F[G] = -.2; //match
    p.p.update("matched_p");
    inverter inv(n.p, p.p);

    // points in this part
    vec V_in = linspace(part - 1, part - 1. / N, N) * .2 / 10.;

    // compute voltage points
    vec V_out(N);
    for (int i = 0; i < N; ++i) {
        cout << "\nstep " << i+1 << "/" << N << ": \n";
        inv.steady_state({ 0, gvdop, V_in(i) });
        V_out(i) = inv.get_output(0)->V;
    }

    // save
    mat data = join_horiz(V_in, V_out);
    stringstream file;
    file << "/part" << part << ".csv";
    data.save(save_folder() + file.str(), csv_ascii);
}

void gstep(double rise) {
    double beg = 1e-12;
    double cool = 3e-12;

    signal<3> pre   = linear_signal<3>(beg,  { 0, .2, 0. }, { 0, .2, 0. }); // before
    signal<3> slope = linear_signal<3>(rise,  { 0, .2, 0. }, { 0, .2, .2 }); // while
    signal<3> after = linear_signal<3>(cool, { 0, .2, .2 }, { 0, .2, .2 }); // after

    signal<3> sig = pre + slope + after; // complete signal

//    sigplot(sig); return;

    stringstream ss;
    ss << "gstep/rise=" << rise;
    save_folder(ss.str());

    device d("ntfet", ntfet, sig.V[0]);
    d.p.F[G] = .2; //match
    d.p.update("matched");
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind = movie::around_Ef(d, -0.05);
    movie argo(d, E_ind, 100); // not for actual movie, only for thesis

    // perform time-evolution
    for (int i = 1; i < sig.N_t; ++i) {
        d.contacts[G]->V = sig.V[i][G];
        d.time_step();
    }
    d.save();

    std::cout << "\nGetting quasi-static current curve...\n";
    quasi_static(sig, d);
}

inline void gsquare(double f) {
    double rise = 100e-15;
    double fall = rise;
    double len = 3.2 / f; // we want 3 periods

    signal<3> sig = square_signal<3>(len, { 0, .2, .0 }, { 0, .2, .2}, f, rise, fall);

//    sigplot(sig); return;

    stringstream ss;
    ss << "gate_square_signal/" << "f=" << f;
    save_folder(ss.str());

    device d("ntfet", ntfet, sig.V[0]);
    d.p.F[G] = .2; //match
    d.p.update("matched");
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // time-evolution:
    for (int i = 1; i < sig.N_t; ++i) {
        d.contacts[G]->V = sig.V[i][G];
        d.time_step();
    }
    d.save();

    std::cout << "\nGetting quasi-static current curve...\n";
    quasi_static(sig, d);
}

void gsine(double f) { // compile with smaller dt!!!
    double len = 3 / f;

    // swing around 0.1V with amplitude 0.1V -> vg between 0 and 0.2
    double phase = 1.5 * M_PI; // start from minimum
    signal<3> sig = sine_signal<3>(len,  { 0, .2, .1 }, { 0,  0, .1 }, f, phase);

//    sigplot(sig); return;

    stringstream ss;
    ss << "gsine/" << "f=" << f;
    save_folder(ss.str());

    device d("ntfet", ntfet, sig.V[0]);
    d.p.F[G] = .2; //match
    d.p.update("matched");
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind = movie::around_Ef(d, -0.05);
    movie argo(d, E_ind, 1);

    // time-evolution
    for (int i = 1; i < sig.N_t; ++i) {
        d.contacts[G]->V = sig.V[i][G];
        d.time_step();
    }
    d.save();

    std::cout << "\nGetting quasi-static current curve...\n";
//    std::string subfolder(save_folder() + "/" + d.name);
//    system("mkdir -p " + subfolder);
    quasi_static(sig, d);
}

void oscillator (double C) {
    stringstream ss;
    ss << "ring_oscillator";
    save_folder(ss.str());

    double T = 100e-12;

    device_params n(ntfet);
    n.F[G] = .2;
    n.update("n_matched");
    device_params p(ptfet);
    p.F[G] = -.2;
    p.update("p_matched");

    ring_oscillator<3> ro(n, p, C);
    ro.time_evolution(signal<2>(T, voltage<2>{ 0.0, 0.2 }));
    ro.save<true>();
}

void inverter_square (double f) {
    stringstream ss;
    ss << "inverter_square/f=" << f;
    save_folder(ss.str());

    device_params n(ntfet);
    n.F[G] = .2;
    n.update("n_matched");
    device_params p(ptfet);
    p.F[G] = -.2;
    p.update("p_matched");

    double C = capacitance(n);
    double rise = 300e-15;
    double fall = rise;
    double len = 3.2 / f; // we want 3 periods

    signal<3> sig = square_signal<3>(len, { 0, .2, .0 }, { 0, .2, .2}, f, rise, fall);

    inverter inv(n, p, C);
    inv.steady_state(sig.V[0]);
    inv.time_evolution(sig);
    inv.save();
}



int main(int argc, char ** argv) {
    // disable nested parallelism globally
    omp_set_nested(0);

    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // first argument is always the number of threads
    // (can not be more than the number specified when compiling openBLAS)
    omp_set_num_threads(stoi(argv[1]));


    // second argument chooses the type of simulation
    string stype(argv[2]);

    if (stype == "Cg" && argc == 3) {
        // estimate C_g
        capacitance(ntfet);
    } else if (stype == "to_CSV" && argc == 4) {
        // full path to .arma file
        to_CSV(string(argv[3]));

    // ------------- test functions -------------------------------------
    } else if (stype == "point" && argc == 6) {
        // Vs, Vd, Vg
        voltage_point(stod(argv[3]), stod(argv[4]), stod(argv[5]));
    } else if (stype == "transtest" && argc == 7) {
        // Vg0, Vg1, Vd, N
        transfer_test(stod(argv[3]), stod(argv[4]), stod(argv[5]), stoi(argv[6]));
    } else if (stype == "outptest" && argc == 7) {
        // Vd1, Vd2, Vg, N
        output_test(stod(argv[3]), stod(argv[4]), stod(argv[5]), stoi(argv[6]));

    // ------------- thesis steady-state simulations --------------------
    } else if (stype == "scaling" && argc == 4) {
        // lg for parallelization
        scaling(stod(argv[3]));
    } else if (stype == "overlap" && argc == 4) {
        // l_sg (called gap) for parallelization
        overlap(stod(argv[3]));
    } else if (stype == "separation" && argc == 4) {
        // l_dg for parallelization
        separation(stod(argv[3]));
    } else if (stype == "ftransfer" && argc == 4) {
        // vs for parallelization
        final_transfer(stod(argv[3]));
    } else if (stype == "foutput" && argc == 4) {
        // vg for parallelization
        final_output(stod(argv[3]));
    } else if (stype == "contacts_trans" && argc == 4) {
        contacts_transfer(stod(argv[3]));
    } else if (stype == "contacts_outp" && argc == 4) {
        contacts_output(stod(argv[3]));
    } else if (stype == "ntd_inverter" && argc == 4) {
        // part 1 to 10 for further parallelization
        ntd_inverter(stod(argv[3]));

    // ------------- thesis time-dependent simulations ------------------
    } else if (stype == "gstep" && argc == 4) {
        // vs for parallelization
        gstep(stod(argv[3]));
    } else if (stype == "gsquare" && argc == 4) {
        // square wave of certain frequency on gate
        gsquare(stod(argv[3]));
    } else if (stype == "gsine" && argc == 4) {
        // sine wave of certain frequency on gate
        gsine(stod(argv[3]));
    } else if (stype == "oscillator" && argc == 4) {
        // sine wave of certain frequency on gate
        oscillator(stod(argv[3]));
    } else if (stype == "inv_square" && argc == 4) {
        // sine wave of certain frequency on gate
        inverter_square(stod(argv[3]));


    } else {
        cout << "wrong number of arguments or unknown simulation type" << endl;
    }
    return 0;
}
