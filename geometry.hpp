#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <armadillo>
#include <string>

class geometry {
public:
    // parameters
    double eps_cnt;
    double eps_ox;
    double l_sc;
    double l_sox;
    double l_sg;
    double l_g;
    double l_dg;
    double l_dox;
    double l_dc;
    double r_cnt;
    double d_ox;
    double r_ext;
    double dx;
    double dr;

    inline std::string to_string();
};

static const geometry fet_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox
     7.0, // l_sc
    15.0, // l_sox
     5.0, // l_sg
    20.0, // l_g
    20.0, // l_dg
     0.0, // l_dox
     7.0, // l_dc
     1.0, // r_cnt
     2.0, // d_ox
     2.0, // r_ext
     0.4, // dx
     0.1  // dr
};

static const geometry tfet_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox (Hf02)
     7.0, // l_sc
    15.0, // l_sox
     5.0, // l_sg
    20.0, // l_g
      20, // l_dg
       0, // l_dox
     7.0, // l_dc
     1.0, // r_cnt (Claus et. al.)
     3.0, // d_ox
     2.0, // r_ext
     0.4, // dx
     0.1  // dr
};

static const geometry tfetc_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox (Hf02)
     7.0, // l_sc
    15.0, // l_sox
     5.0, // l_sg
    20.0, // l_g
    20.0, // l_dg
     0.0, // l_dox
     7.0, // l_dc
.5 * 1.3, // r_cnt (16,0)
     1.0, // d_ox
     2.0, // r_ext
     0.2, // dx
     0.1  // dr
};

static const geometry tfetn_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox (Hf02)
    10.0, // l_sc
     0.0, // l_sox
    10.0, // l_sg
    20.0, // l_g
    10.0, // l_dg
     0.0, // l_dox
    10.0, // l_dc
.5 * 1.3, // r_cnt (16,0)
     1.0, // d_ox
     2.0, // r_ext
     0.1, // dx
     0.1  // dr
};

std::string geometry::to_string() {
    using namespace std;

    stringstream ss;

    ss << "eps_cnt = " << eps_cnt << endl;
    ss << "eps_ox  = " << eps_ox  << endl;
    ss << "l_sc    = " << l_sc    << endl;
    ss << "l_sox   = " << l_sox   << endl;
    ss << "l_sg    = " << l_sg    << endl;
    ss << "l_g     = " << l_g     << endl;
    ss << "l_dg    = " << l_dg    << endl;
    ss << "l_dox   = " << l_dox   << endl;
    ss << "l_dc    = " << l_dc    << endl;
    ss << "r_cnt   = " << r_cnt   << endl;
    ss << "d_ox    = " << d_ox    << endl;
    ss << "r_ext   = " << r_ext   << endl;
    ss << "dx      = " << dx      << endl;
    ss << "dr      = " << dr      << endl;

    return ss.str();
}

#endif // GEOMETRY_HPP

