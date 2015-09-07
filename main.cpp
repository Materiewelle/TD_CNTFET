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
     7.0, // l_sc
    15.0, // l_sox
     5.0, // l_sg
    20.0, // l_g
      20, // l_dg
       0, // l_dox
     7.0, // l_dc
     1.0, // r_cnt
     3.0, // d_ox
     2.0, // r_ext
     0.4, // dx
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

//------- simulation routines ---------------

using namespace arma;
using namespace std;

void voltage_point(double vs, double vd, double vg) {
    device d("ntfet", ntfet, {vs, vd, vg});
    d.steady_state();
    cout << "I = " << d.I[0].total[0] << std::endl;
    plot(make_pair(d.p.x, d.phi[0].data));
    potential::plot2D(d.p, { 0, vd, vg }, d.n[0]);
    plot(make_pair(d.p.x, d.n[0].total));
    plot_ldos(d.p, d.phi[0], 2000, -1, 1);
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
    transfer<true>(d.p, { { 0, .2, -.3 } }, .5, 800);
    output<true>(d.p,   { { 0, -.2, .2 } }, .5, 800);
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
    transfer<true>(d.p, { { 0, .2, -.3 } }, .5, 800);
    output<true>(d.p,   { { 0, -.2, .2 } }, .5, 800);
}

void separation(double ldg) {
    stringstream ss;
    ss << "separation/ldg=" << ldg;
    save_folder(ss.str());
    device d("ntfet", ntfet);
    d.p.l_dg = ldg;
    d.p.update("updated");
    transfer<true>(d.p, { { 0, .2, -.3 } }, .5, 800);
    output<true>(d.p,   { { 0, -.2, .2 } }, .5, 800);
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

    // ------------- test function calls --------------------------------
    if (stype == "point" && argc == 6) {
        // Vs, Vd, Vg
        voltage_point(stod(argv[3]), stod(argv[4]), stod(argv[5]));
    } else if (stype == "transfer" && argc == 7) {
        // Vg0, Vg1, Vd, N
        transfer_test(stod(argv[3]), stod(argv[4]), stod(argv[5]), stoi(argv[6]));
    } else if (stype == "output" && argc == 7) {
        // Vd1, Vd2, Vg, N
        output_test(stod(argv[3]), stod(argv[4]), stod(argv[5]), stoi(argv[6]));

    // ------------- thesis simulation types ------------------------------
    } else if (stype == "scaling" && argc == 4) {
        // lg for parallelization
        scaling(stod(argv[3]));
    } else if (stype == "overlap" && argc == 4) {
        // l_sg (called gap) for parallelization
        overlap(stod(argv[3]));
    } else if (stype == "separation" && argc == 4) {
        // l_dg for parallelization
        separation(stod(argv[3]));
    } else {
        cout << "wrong number of arguments or unknown simulation type" << endl;
    }
    return 0;
}
