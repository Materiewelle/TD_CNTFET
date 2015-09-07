#ifndef MODEL_HPP
#define MODEL_HPP

#include <armadillo>
#include <array>
#include <string>

#include "constant.hpp"

class model {
public:
    // parameters
    double E_g;
    double m_eff;
    double E_gc;
    double m_efc;
    std::array<double, 3> F;

    inline std::string to_string();
};

std::string model::to_string() {
    using namespace std;

    stringstream ss;

    ss << "E_g     = " << E_g   << endl;
    ss << "m_eff   = " << m_eff << endl;
    ss << "E_gc    = " << E_gc  << endl;
    ss << "m_efc   = " << m_efc << endl;
    ss << "F_s     = " << F[S]  << endl;
    ss << "F_d     = " << F[D]  << endl;
    ss << "F_g     = " << F[G]  << endl;

    return ss.str();
}

#endif

