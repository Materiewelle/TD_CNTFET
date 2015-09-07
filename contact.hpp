#ifndef CONTACT_HPP
#define CONTACT_HPP

#include <memory>

class contact {
public:
    double V; // voltage
    double c; // capacitance
    inline contact(double V, double c);

    inline void update(double dI, double dt);
};

using contact_ptr = std::shared_ptr<contact>;

//----------------------------------------------------------------------------------------------------------------------

contact::contact(double V_, double c_)
    : V(V_), c(c_) {
}

void contact::update(double dI, double dt) {
    V -= dI * dt / c;
}

#endif

