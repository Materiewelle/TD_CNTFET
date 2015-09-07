#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <armadillo>
#include <map>

#include "charge_density.hpp"
#include "constant.hpp"
#include "device_params.hpp"
#include "voltage.hpp"
#include "util/anderson.hpp"
#include "util/gnuplot.hpp"

class potential {
public:
    arma::vec data;
    arma::vec twice;

    inline potential();
    inline potential(const device_params & p, const arma::vec & R);
    inline potential(const device_params & p, const arma::vec & R0, const charge_density & n);
    inline potential(const device_params & p, const voltage<3> & V);
    inline double update(const device_params & p, const arma::vec & R0, const charge_density & n, anderson & mr_neo);

    inline double & operator()(int index);
    inline const double & operator()(int index) const;
    inline double s() const;
    inline double d() const;

    inline void smooth(const device_params & p);

    static inline arma::vec get_R0(const device_params & p, const voltage<3> & V);

//private:

    inline void update_twice();

    template<bool minmax>
    inline void smooth(unsigned x0, unsigned x1);

    static inline arma::vec get_R(const device_params & p, const arma::vec & R0, const charge_density & n);
    static inline const arma::sp_mat & get_C(const device_params & p);

    static inline void plot2D(const device_params & p, const voltage<3> & V, const charge_density & n);
    static inline void plot2D(const device_params & p, const voltage<3> & V);
};

//----------------------------------------------------------------------------------------------------------------------

potential::potential() {
}
potential::potential(const device_params & p, const arma::vec & R) {
    using namespace arma;

    // superlu is apparantly not thread-safe
    vec phi2D;
    #pragma omp critical
    {
        phi2D = spsolve(get_C(p), R);
    }

    data = phi2D.rows((p.M_cnt - 1) * p.N_x, p.M_cnt * p.N_x - 1);

    update_twice();
}
potential::potential(const device_params & p, const arma::vec & R0, const charge_density & n)
    : potential(p, get_R(p, R0, n)) {
}
potential::potential(const device_params & p, const voltage<3> & V)
    : potential(p, get_R0(p, V)) {
}

double potential::update(const device_params & p, const arma::vec & R0, const charge_density & n, anderson & mr_neo) {
    // calculate undamped update
    potential phi_update(p, R0, n);

    // anderson mixing
    arma::vec f = phi_update.data - data;
    mr_neo.update(data, f);

    update_twice();

    // return dphi
    return max(abs(f));
}

double & potential::operator()(int index) {
    return data[index];
}
const double & potential::operator()(int index) const {
    return data[index];
}
double potential::s() const {
    return data[0];
}
double potential::d() const {
    return data[data.size() - 1];
}

void potential::smooth(const device_params & p) {
    // smooth source region
    if (p.F[S] > 0) {
        smooth<true>(0, p.N_sc + p.N_sox + p.N_sg + p.N_g * 0.05);
    } else {
        smooth<false>(0, p.N_sc + p.N_sox + p.N_sg + p.N_g * 0.05);
    }

    // smooth drain region
    if (p.F[D] > 0) {
        smooth<true>(p.N_sc + p.N_sox + p.N_sg + p.N_g * 0.95, p.N_x);
    } else {
        smooth<false>(p.N_sc + p.N_sox + p.N_sg + p.N_g * 0.95, p.N_x);
    }

    update_twice();
}

arma::vec potential::get_R0(const device_params & p, const voltage<3> & V0) {
    using namespace arma;

    // add built-in voltages
    voltage<3> V = - (V0 + p.F);

    // init return vector
    vec R0(p.N_x * p.M_r);
    R0.fill(0.0);

    for (int j = p.M_cnt + 1; j < p.M_cnt + p.M_ox; ++j) {
        for (int i = 0; i <= (int)p.sox.a; ++i) {
            R0(j * p.N_x + i) = V[S];
        }
        for (int i = p.dc.a; i < p.N_x; ++i) {
            R0(j * p.N_x + i) = V[D];
        }
    }
    for (int j = p.M_cnt + p.M_ox; j < p.M_r; ++j) {
        for (int i = 0; i <= (int)p.sg.a; ++i) {
            R0(j * p.N_x + i) = V[S];
        }
        for (int i = p.g.a; i <= (int)p.dg.a; ++i) {
            R0(j * p.N_x + i) = V[G];
        }
        for (int i = p.dox.a; i < p.N_x; ++i) {
            R0(j * p.N_x + i) = V[D];
        }
    }

    return R0;
}

void potential::update_twice() {
    twice.resize(data.size() * 2);

    // duplicate each entry
    for (unsigned i = 0; i < data.size(); ++i) {
        twice(2 * i    ) = data(i);
        twice(2 * i + 1) = data(i);
    }
}

template<bool minmax>
void potential::smooth(unsigned x0, unsigned x1) {
    using namespace arma;

    if (minmax) {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) >= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) >= data(i)) {
                    data.rows(i+1, j-1).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) >= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) >= data(i)) {
                    data.rows(j+1, i-1).fill(data(i));
                    break;
                }
            }
        }
    } else {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) <= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) <= data(i)) {
                    data.rows(i+1, j-1).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) <= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) <= data(i)) {
                    data.rows(j+1, i - 1).fill(data(i));
                    break;
                }
            }
        }
    }
}

arma::vec potential::get_R(const device_params & p, const arma::vec & R0, const charge_density & n) {
    using namespace arma;
    vec R = R0;

    // add non-constant part to R0 due to charge_density at the boundary of the cnt
    //R.rows((p.M_cnt - 1) * p.N_x, p.M_cnt * p.N_x - 1) += n.total * p.r_cnt * 1e9; // 10^9 because of m->nm in eps_0
    R.rows((p.M_cnt - 1) * p.N_x, p.M_cnt * p.N_x - 1) -= n.total;
    return R;
}
const arma::sp_mat & potential::get_C(const device_params & p) {
    using namespace arma;

    // check if C was already calculated for this device
    static thread_local std::map<std::string, arma::sp_mat> C;
    auto it = C.find(p.name);
    if (it != std::end(C)) {
        return it->second;
    }

    // get epsilon for indices i, j
    auto eps = [&p] (int i, int j) -> double {
        if (j < 0) {
        } else if (j < p.M_cnt) {
            if ((i >= (int)p.sc.a) && (i <= (int)p.dc.b)) {
                return p.eps_cnt * c::eps_0;
            }
        } else if (j < p.M_cnt + p.M_ox) {
            if ((i >= (int)p.sox.a) && (i <= (int)p.dox.b)) {
                return p.eps_ox * c::eps_0;
            } else {
                if ((i >= (int)p.sc.a) && (i <= (int)p.dc.b)) {
                    return c::eps_0;
                }
            }
        } else if (j < p.M_r - 1) {
            if ((i >= (int)p.sg.a) && ((i <= (int)p.sg.b) || (i >= (int)p.dg.a)) && (i <= (int)p.dg.b)) {
                return c::eps_0;
            }
        }
        return 0.0;
    };

    // check if (i, j) has dirichlet boundary conditions
    auto dirichlet = [&p] (int i, int j) -> bool {
        if ((j > p.M_cnt) && (j < p.M_cnt + p.M_ox)) {
            if ((i <= (int)p.sox.a) || (i > (int)p.dox.b)) {
                return true;
            }
        } else if (j >= p.M_cnt + p.M_ox) {
            if ((i <= (int)p.sg.a) || ((i > (int)p.sg.b) && (i <= (int)p.dg.a)) || (i > (int)p.dg.b)) {
                return true;
            }
        }
        return false;
    };

    // helper objects to create sparse matrix faster
    umat indices(2, p.M_r * p.N_x * 5);
    vec  values(    p.M_r * p.N_x * 5);
    uword N_v = 0;
    auto set_value = [&indices, &values, &N_v] (uword i, uword j, double val) {
        if (val != 0.0) {
            indices(0, N_v) = i;
            indices(1, N_v) = j;
            values(N_v++) = val;
        }
    };

    // shortcuts
    double dr = p.dr;
    double dx = p.dx;
    double dr_dx = dr / dx;
    double dx_dr = dx / dr;

    // start values
    int k       = 0;     // current main diagonal element index

    // main loop over rows
    for (int j = 0; j < p.M_r; ++j) {
        // radius
        double r   = j * dr;
        double rp  = r + 0.25 * dr;
        double rpp = r + 0.5  * dr;
        double rm  = r - 0.25 * dr;
        double rmm = r - 0.5  * dr;

        for (int i = 0; i < p.N_x; ++i) {
            if (dirichlet(i, j)) {
                set_value(k, k, 1.0);
            } else {
                double C_l = - M_PI * dr_dx * (eps(i - 1, j - 1) * rm  + eps(i - 1, j    ) * rp ) * 1e-9;
                double C_r = - M_PI * dr_dx * (eps(i    , j - 1) * rm  + eps(i    , j    ) * rp ) * 1e-9;
                double C_i = - M_PI * dx_dr * (eps(i - 1, j - 1) * rmm + eps(i    , j - 1) * rmm) * 1e-9;
                double C_o = - M_PI * dx_dr * (eps(i - 1, j    ) * rpp + eps(i    , j    ) * rpp) * 1e-9;
                double C_s = - (C_l + C_r + C_i + C_o);

                // store values
                set_value(k, k        , C_s);
                set_value(k, k - 1    , C_l);
                set_value(k, k + 1    , C_r);
                set_value(k, k - p.N_x, C_i);
                set_value(k, k + p.N_x, C_o);
            }
            ++k;
        }
    }

    // fit to size
    indices.resize(2, N_v);
    values.resize(N_v);

    // create, save and return sparse matrix C
    C[p.name] = sp_mat(indices, values);
    return C[p.name];
}

void potential::plot2D(const device_params & p, const voltage<3> & V, const charge_density & n) {

    arma::vec phi2Dv = spsolve(get_C(p), get_R(p, get_R0(p, V), n));

    arma::mat phi2D(p.N_x, p.M_r);

    // fill the matrix
    for (int j = 0; j < p.M_r; ++j) {
        for (int i = 0; i < p.N_x; ++i) {
            phi2D(i, j) = phi2Dv(j * p.N_x + i);
        }
    }

    // mirror
    phi2D = arma::join_horiz(arma::fliplr(phi2D), phi2D).t();

    gnuplot gp;

    // gnuplot setup

    gp << "set palette defined (0 '#B2182B', 1 '#D6604D', 2 '#F4A582', 3 '#FDDBC7', 4 '#D1E5F0', 5 '#92C5DE', 6 '#4393C3', 7 '#2166AC')\n";
//    gp << "set title \"Potential cross-section for V_{s} = " << V[S] << " V, V_{g} = " << V[G] << " V and V_{d} = " << V[D] << " V\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"r / nm\"\n";
    gp << "set zlabel \"Phi / V\"\n";
    gp << "unset key\n";

    gp << "set terminal pdf size 5,3\nset output 'pot.pdf'\n";

    // indicate cnt area
    gp << "set obj rect from " << 0 << "," << p.r_cnt << " to " << p.l << "," << -p.r_cnt << "front fillstyle empty\n";
    gp << "set label \"CNT\" at " << 0.5 * p.l << "," << 0 << " center front\n";

    // indicate oxide area
    double x0 = p.l_sc;
    double x1 = p.l - p.l_dc;
    double y0 = p.r_cnt + p.d_ox;
    double y1 = p.r_cnt;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    // indicate gate contact area
    x0 = p.l_sc + p.l_sox + p.l_sg;
    x1 = p.l - p.l_dc - p.l_dox - p.l_dg;
    y0 = p.R;
    y1 = p.r_cnt + p.d_ox;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"gate\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"gate\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    // indicate contact areas
    // --------------------------------------------------------------------------------------
    // top/bottom
    x0 = 0;
    x1 = p.l_sc + p.l_sox;
    y0 = p.R;
    y1 = y0;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n";
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror y
    x0 = p.l - p.l_dc - p.l_dox;
    x1 = p.l;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n"; //mirror x
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror both

    // outside
    x0 = 0;
    y0 = p.R;
    x1 = x0;
    y1 = p.r_cnt;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n";
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror
    x0 = p.l;
    x1 = x0;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n"; //mirror x
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror both

    // inside
    x0 = p.l_sc + p.l_sox;
    y0 = p.R;
    x1 = x0;
    y1 = p.r_cnt + p.d_ox;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n";
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror
    x0 = p.l - p.l_dc - p.l_dox;
    x1 = x0;
    gp << "set arrow from " << x0 << "," << y0 << " to "<< x1 << "," << y1 << " nohead front\n"; //mirror x
    gp << "set arrow from " << x0 << "," << -y0 << " to "<< x1 << "," << -y1 << " nohead front\n"; //mirror both

    // label
    x0 = 0;
    x1 = p.l_sc + p.l_sox;
    y0 = p.r_cnt + p.d_ox;
    y1 = p.R;
    gp << "set label \"source\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";
    gp << "set label \"source\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * +(y1 - y0) - y1 << " center front\n"; //mirror y
    x0 = p.l;
    x1 = p.l - p.l_dc - p.l_dox;
    gp << "set label \"drain\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n"; //mirror x
    gp << "set label \"drain\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * +(y1 - y0) - y1 << " center front\n"; //mirror both

    gp.set_background(p.x, arma::join_vert(arma::flipud(-p.r), p.r), phi2D);
    gp.plot();
}

void potential::plot2D(const device_params & p, const voltage<3> & V) {
    charge_density n;
    n.total.resize(p.N_x);
    n.total.fill(0.0);
    plot2D(p, V, n);
}

#endif
