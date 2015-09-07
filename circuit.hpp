#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include "device.hpp"
#include "voltage.hpp"
#include "signal.hpp"

template<ulint N_in, ulint N_out>
class circuit {
public:
    inline circuit();

    inline const device & operator[](int index) const;
    inline device & operator[](int index);

    inline const contact_ptr & get_input(int index) const;
    inline contact_ptr & get_input(int index);
    inline const contact_ptr & get_output(int index) const;
    inline contact_ptr & get_output(int index);

    inline int add_device(const std::string & n, const device_params & p);

    inline void link(int d1, int ct1, int d2, int ct2);
    inline void link_input(int d, int ct, int in);
    inline void link_output(int d, int ct, int out);
    inline void link_inout(int in, int out);
    inline void link_outin(int out, int in);

    virtual bool steady_state(const voltage<N_in> & V) = 0;
    inline bool time_step(const voltage<N_in> & V);

    inline void time_evolution(const signal<N_in> & s);

    template<bool plots>
    inline void save();

protected:
    std::vector<device> devices;
    std::array<contact_ptr, N_in> inputs;
    std::array<contact_ptr, N_out> outputs;
    std::vector<voltage<N_out>> V_out;
};

//----------------------------------------------------------------------------------------------------------------------

template<ulint N_in, ulint N_out>
circuit<N_in, N_out>::circuit() {
    for (ulint i = 0; i < N_in; ++i) {
        inputs[i] = std::make_shared<contact>(0.0, c::inf);
    }
    for (ulint i = 0; i < N_out; ++i) {
        outputs[i] = std::make_shared<contact>(0.0, c::inf);
    }
}

template<ulint N_in, ulint N_out>
const device & circuit<N_in, N_out>::operator[](int index) const {
    return devices[index];
}
template<ulint N_in, ulint N_out>
device & circuit<N_in, N_out>::operator[](int index) {
    return devices[index];
}

template<ulint N_in, ulint N_out>
const contact_ptr & circuit<N_in, N_out>::get_input(int index) const {
    return inputs[index];
}
template<ulint N_in, ulint N_out>
contact_ptr & circuit<N_in, N_out>::get_input(int index) {
    return inputs[index];
}
template<ulint N_in, ulint N_out>
const contact_ptr & circuit<N_in, N_out>::get_output(int index) const {
    return outputs[index];
}
template<ulint N_in, ulint N_out>
contact_ptr & circuit<N_in, N_out>::get_output(int index) {
    return outputs[index];
}

template<ulint N_in, ulint N_out>
int circuit<N_in, N_out>::add_device(const std::string & n, const device_params & p) {
    int s = devices.size();
    devices.emplace_back(n, p);
    return s;
}

template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::link(int d1, int ct1, int d2, int ct2) {
    devices[d1].contacts[ct1] = devices[d2].contacts[ct2];
}
template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::link_input(int d, int ct, int in) {
    devices[d].contacts[ct] = inputs[in];
}
template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::link_output(int d, int ct, int out) {
    devices[d].contacts[ct] = outputs[out];
}
template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::link_inout(int in, int out) {
    inputs[in] = outputs[out];
}
template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::link_outin(int out, int in) {
    outputs[out] = inputs[in];
}

template<ulint N_in, ulint N_out>
void circuit<N_in, N_out>::time_evolution(const signal<N_in> & s) {
    steady_state(s[0]);

    #pragma omp parallel for schedule(static)
    for (ulint i = 0; i < devices.size(); ++i) {
        devices[i].init_time_evolution(s.N_t);
    }
    for (int i = 1; i < s.N_t; ++i) {
        time_step(s[i]);
    }
}

template<ulint N_in, ulint N_out>
bool circuit<N_in, N_out>::time_step(const voltage<N_in> & V) {
    // set input voltages
    for (ulint i = 0; i < N_in; ++i) {
        inputs[i]->V = V[i];
    }

    // calculate time_step for each device
    bool converged = true;

    #pragma omp parallel for schedule(static)
    for (ulint i = 0; i < devices.size(); ++i) {
        converged &= devices[i].time_step();
    }

    // update device contacts
    for (ulint i = 0; i < devices.size(); ++i) {
        devices[i].update_contacts();
    }

    // save output voltages
    voltage<N_out> V_o;
    for (ulint i = 0; i < N_out; ++i) {
        V_o[i] = outputs[i]->V;
        std::cout << "V_out[" << i << "] = " << std::setprecision(12) << V_o[i] << std::endl;
    }
    V_out.push_back(V_o);

    return converged;
}

template<ulint N_in, ulint N_out>
template<bool plots>
void circuit<N_in, N_out>::save() {
    for (ulint i = 0; i < devices.size(); ++i) {
        devices[i].save<plots>();
    }

    for (ulint i = 0; i < N_out; ++i) {
        arma::vec V(V_out.size());
        for (ulint j = 0; j < V_out.size(); ++j) {
            V[j] = V_out[j][i];
        }
        std::stringstream ss;
        ss << save_folder() << "/V_out" << i;
        std::string file_name = ss.str();
        V.save(file_name + ".arma");

        if (plots) {
            // time vector
            arma::vec t = arma::linspace(0, V.size() * c::dt, V.size());

            // make a plot of V and save it as a PNG
            gnuplot gp;
            gp << "set terminal png\n";
            gp << "set title 'Output voltage'\n";
            gp << "set xlabel 't / ps'\n";
            gp << "set ylabel 'V_{out" << i << "} / V'\n";
            gp << "set format x '%1.2f'\n";
            gp << "set format y '%1.2f'\n";
            gp << "set output '" << file_name << ".png'\n";
            gp.add(std::make_pair(t * 1e12, V));
            gp.plot();
        }
    }
}

#endif
