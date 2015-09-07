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

