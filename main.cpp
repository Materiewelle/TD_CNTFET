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

using namespace arma;
using namespace std;

// set the type of device (mosfet or tfet)
static device_params ntype = ntfet;
static device_params ptype = ptfet;

static inline void setup() {
    // disable nested parallelism globally
    omp_set_nested(0);

    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

static inline void point(char ** argv) {
    // computes the current for a given voltage point

    double vs = stod(argv[3]);
    double vd = stod(argv[4]);
    double vg = stod(argv[5]);

    device d("ntype", ntype, {vs, vd, vg});
    d.steady_state();
    cout << "I = " << d.I[0].total[0] << std::endl;
}

static inline void trans(char ** argv) {
    // starts transfer-curve simulations with a certain gate-length

    double l_g = stod(argv[3]);
    double sox = stod(argv[4]);
    double Vg0 = stod(argv[5]);
    double Vg1 = stod(argv[6]);
    double Vd  = stod(argv[7]);
    int N      = stoi(argv[8]);

    stringstream ss;
    ss << "transfer/lg=" << l_g << "_lsox=" << sox << "_Vd=" << Vd;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    device d("prototype", ntype);
    d.p.l_g = l_g;

    // extract total contact length from prototype
    double tmp = d.p.l_sc + d.p.l_sox;
    d.p.l_sox = sox;
    d.p.l_sc = tmp - sox;
    d.p.update("updated");

    transfer<true>(d.p, { { 0, Vd, Vg0 } }, Vg1, N);

    ofstream s(save_folder() + "/device.ini");
    s << d.p.to_string();
    s.close();
}

static inline void outp(char ** argv) {
    // starts output-curve simulations with a certain gate-length

    double l_g = stod(argv[3]);
    double Vd0 = stod(argv[4]);
    double Vd1 = stod(argv[5]);
    double Vg  = stod(argv[6]);
    int N      = stoi(argv[7]);

    stringstream ss;
    ss << "output/lg=" << l_g << "_Vg=" << Vg;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    device d("prototype", ntype);
    d.p.l_g = l_g;
    d.p.update("updated");

    output<true>(d.p, { { 0, Vd0, Vg } }, Vd1, N);

    ofstream s(save_folder() + "/device.ini");
    s << d.p.to_string();
    s.close();
}

static void inv(char ** argv) {
    // starts a static inverter simulation

    double Vin0  = stod(argv[3]);
    double Vin1  = stod(argv[4]);
    double V_dd  = stod(argv[5]);
    int    N     = stoi(argv[6]);

    inverter inv(ntype, ptype);

    stringstream ss;
    ss << "ntd_inverter/Vdd=" << V_dd;

    cout << "saving results in " << save_folder(ss.str()) << endl;

    vec V_in = linspace(Vin0, Vin1, N);
    vec V_out(N);

    for (int i = 0; i < N; ++i) {
        cout << "\nstep " << i+1 << "/" << N << ": \n";
        inv.steady_state({ 0, V_dd, V_in(i) });
        V_out(i) = inv.get_output(0)->V;
    }

    mat ret = join_horiz(V_in, V_out);
    ret.save(save_folder() + "/inverter_curve.csv", csv_ascii);

    ofstream sn(save_folder() + "/ntype.ini");
    sn << ntype.to_string();
    sn.close();

    ofstream sp(save_folder() + "/ptype.ini");
    sp << ptype.to_string();
    sp.close();
}

static inline void ro(char ** argv) {
    // starts a transient ring-oscillator simulation

    double T = stod(argv[3]);
    double C = stod(argv[4]);
    double V_dd = stod(argv[5]);
    stringstream ss;
    ss << "ring_oscillator/" << "C=" << C << "_Vdd=" << V_dd;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    ring_oscillator<3> ro(ntype, ptype, C);
    ro.time_evolution(signal<2>(T, voltage<2>{ 0.0, V_dd }));
    ro.save<true>();

    ofstream sn(save_folder() + "/ntype.ini");
    sn << ntype.to_string();
    sn.close();

    ofstream sp(save_folder() + "/ptype.ini");
    sp << ptype.to_string();
    sp.close();
}

static inline void ldos(char ** argv) {
    // plots the ldos for a self-consistent static situation

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);
    double Emin = stod(argv[5]);
    double Emax = stod(argv[6]);
    int    N    = stoi(argv[7]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot_ldos(dev.p, dev.phi[0], N, Emin, Emax);
}

static inline void pot(char ** argv) {
    // plots a self-consisent potential

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot(make_pair(dev.p.x, dev.phi[0].data));
    potential::plot2D(dev.p, { 0, vd, vg }, dev.n[0]);
}

static inline void den(char ** argv) {

    double vd = stod(argv[3]);
    double vg = stod(argv[4]);

    device dev("test", ntype, voltage<3>{ 0, vd, vg });
    dev.steady_state();

    plot(make_pair(dev.p.x, dev.n[0].total));
}

static inline void gstep(char ** argv) {
    // time-dependent simulation with step-signal on the gate

    double V0    = stod(argv[3]);
    double V1    = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double beg   = stod(argv[6]);
    double len   = stod(argv[7]);
    double cool  = stod(argv[8]);

    stringstream ss;
    ss << "gate_step_signal/" << "V0=" << V0 << "V1=" << V1;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> pre   = linear_signal<3>(beg,  { 0, Vd, V0 }, { 0, Vd, V0 }); // before
    signal<3> slope = linear_signal<3>(len,  { 0, Vd, V0 }, { 0, Vd, V1 }); // while
    signal<3> after = linear_signal<3>(cool, { 0, Vd, V1 }, { 0, Vd, V1 }); // after

    signal<3> sig = pre + slope + after; // complete signal

    device d("nfet", ntype, { 0, Vd, V0 });
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, +0.1);
    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, +0.4);
    std::vector<std::pair<int, int>> E_ind;
    E_ind.reserve(E_ind1.size() + E_ind2.size());
    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

    //movie argo(d, E_ind);

    // set voltages
    // (note: only V_g is needed, but I wanted to try it for later...)
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void gsquare(char ** argv) {

    double V0    = stod(argv[3]);
    double V1    = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double rise  = stod(argv[6]);
    double fall  = stod(argv[7]); // after begin
    double f     = stod(argv[8]);
    double T     = stod(argv[9]);


    stringstream ss;
    ss << "gate_square_signal/" << "f=" << f;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> sig = square_signal<3>(T, { 0, V0, Vd }, { 0, V1, Vd}, f, rise, fall);

    vec vs(sig.N_t);
    vec vd(sig.N_t);
    vec vg(sig.N_t);
    for (int i = 0; i < sig.N_t; ++i) {
        vs(i) = sig.V[i][S];
        vd(i) = sig.V[i][D];
        vg(i) = sig.V[i][G];
    }
    plot(vs, vd, vg);

    device d("nfet", ntype, sig.V[0]);
    d.steady_state();
    d.init_time_evolution(sig.N_t);

//    // get energy indices around fermi energy and init movie
//    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, -0.05);
//    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, -0.1);
//    std::vector<std::pair<int, int>> E_ind;
//    E_ind.reserve(E_ind1.size() + E_ind2.size());
//    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
//    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

//    //movie argo(d, E_ind);

    // set voltages
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void gsine(char ** argv) {
    // time-dependent simulation with step-signal on the gate

    double V0    = stod(argv[3]);
    double Vamp  = stod(argv[4]);
    double Vd    = stod(argv[5]);
    double beg   = stod(argv[6]);
    double len   = stod(argv[7]); // after begin
    double f     = stod(argv[8]);
    double ph    = stod(argv[9]) * 2 * M_PI; // phase

    double Vstart = V0 + Vamp * sin(ph);

    stringstream ss;
    ss << "gate_sine_signal/" << "f=" << f;
    cout << "saving results in " << save_folder(ss.str()) << endl;

    signal<3> pre   = linear_signal<3>(beg,  { 0, Vd, Vstart }, { 0, Vd, Vstart }); // before
    signal<3> sine  =   sine_signal<3>(len,  { 0, Vd, V0 }, { 0,  0, Vamp }, f, ph); // oscillation

    signal<3> sig = pre + sine; // complete signal

//    vec vs(sig.N_t);
//    vec vd(sig.N_t);
//    vec vg(sig.N_t);
//    for (int i = 0; i < sig.N_t; ++i) {
//        vs(i) = sig.V[i][S];
//        vd(i) = sig.V[i][D];
//        vg(i) = sig.V[i][G];
//    }
//    plot(vs, vd, vg);

    device d("nfet", ntype, sig.V[0]);
    d.steady_state();
    d.init_time_evolution(sig.N_t);

    // get energy indices around fermi energy and init movie
    std::vector<std::pair<int, int>> E_ind1 = movie::around_Ef(d, -0.05);
    std::vector<std::pair<int, int>> E_ind2 = movie::around_Ef(d, -0.1);
    std::vector<std::pair<int, int>> E_ind;
    E_ind.reserve(E_ind1.size() + E_ind2.size());
    E_ind.insert(E_ind.end(), E_ind1.begin(), E_ind1.end());
    E_ind.insert(E_ind.end(), E_ind2.begin(), E_ind2.end());

    //movie argo(d, E_ind);

    // set voltages
    for (int i = 1; i < sig.N_t; ++i) {
        for (int term : {S, D, G}) {
            d.contacts[G]->V = sig.V[i][term];
        }
        d.time_step();
    }
    d.save();
}

static inline void test(char ** argv) {
    device d("nfet", ntype);
    auto t = transfer<false>(d.p, { { 0, .2, .2 } }, .2, 1);
    cout << t << endl;
}

int main(int argc, char ** argv) {
    setup();

    // first argument is always the number of threads
    // (can not be more than the number specified when compiling openBLAS)
    omp_set_num_threads(stoi(argv[1]));

    // second argument chooses the type of simulation
    string stype(argv[2]);
    if (stype == "point" && argc == 6) {
        point(argv);
    } else if (stype == "trans" && argc == 9) {
        trans(argv);
    } else if (stype == "outp" && argc == 8) {
        outp(argv);
    } else if (stype == "inv" && argc == 7) {
        inv(argv);
    } else if (stype == "ldos" && argc == 8) {
        ldos(argv);
    } else if (stype == "pot" && argc == 5) {
        pot(argv);
    } else if (stype == "den" && argc == 5) {
        den(argv);
    } else if (stype == "ro" && argc == 6) {
        ro(argv);
    } else if (stype == "gstep" && argc == 9) {
        gstep(argv);
    } else if (stype == "gsine" && argc == 10) {
        gsine(argv);
    } else if (stype == "gsquare" && argc == 10) {
        gsquare(argv);
    } else if (stype == "test") {
        test(argv);
    } else {
        cout << "wrong number of arguments or unknown simulation type" << endl;
    }

    return 0;
}
