#ifndef RING_OSCILLATOR_HPP
#define RING_OSCILLATOR_HPP

#include <cmath>

#include "circuit.hpp"

template<int N>
class ring_oscillator : private circuit<2, N> {
public:
    inline ring_oscillator(const device_params & n, const device_params & p, double capacitance);

    inline const device & n(int i) const;
    inline device & n(int i);
    inline const device & p(int i) const;
    inline device & p(int i);

    inline bool steady_state(const voltage<2> & V) override;
    using circuit<2, N>::time_step;
    using circuit<2, N>::time_evolution;
    using circuit<2, N>::save;

private:
    std::array<int, N> n_i;
    std::array<int, N> p_i;
};

//----------------------------------------------------------------------------------------------------------------------

template<int N>
ring_oscillator<N>::ring_oscillator(const device_params & n_, const device_params & p_, double capacitance) {
    // create devices
    for (int i = 0; i < N; ++i) {
        // give each device a distinct name
        std::stringstream ss;
        ss << "_" << i;
        std::string suffix = ss.str();

        // add devices and save indices
        n_i[i] = this->add_device(n_.name + suffix, n_);
        p_i[i] = this->add_device(p_.name + suffix, p_);
    }

    // link devices
    for (int i = 0; i < N; ++i) {
        // source contacts
        this->link_input(n_i[i], S, GND); // nfet source to ground
        this->link_input(p_i[i], S, VDD); // pfet source to to V_dd

        // gate contacts
        this->link_output(n_i[i], G, (i + N - 1) % N); // nfet gate to previous output
        this->link_output(p_i[i], G, (i + N - 1) % N); // pfet gate to previous output

        // drain contacts
        this->link_output(n_i[i], D, i); // nfet drain to output[i]
        this->link_output(p_i[i], D, i); // pfet drain to output[i]
    }

    // set capacitance (all the same)
    for (int i = 0; i < N; ++i) {
        this->outputs[i]->c = capacitance;
    }
}

template<int N>
const device & ring_oscillator<N>::n(int i) const {
    return this->devices[n_i[i]];
}
template<int N>
device & ring_oscillator<N>::n(int i) {
    return this->devices[n_i[i]];
}
template<int N>
const device & ring_oscillator<N>::p(int i) const {
    return this->devices[p_i[i]];
}
template<int N>
device & ring_oscillator<N>::p(int i) {
    return this->devices[p_i[i]];
}

template<int N>
bool ring_oscillator<N>::steady_state(const voltage<2> & V) {
    // NOTE: there is no steady state for a ring oscillator,
    //       but a self-consistent solution is needed as a starting point for time-evolutions!

    // set input voltages
    this->inputs[GND]->V = V[GND];
    this->inputs[VDD]->V = V[VDD];

    // strength of the kick (relative to Vdd)
    static constexpr double kick = 1e-2;

    // give the inverter inputs a small kick according to their phase
    for (int i = 0; i < N; ++i) {
        double phase_offset = sin(i * 2 * M_PI / N);
        std::cout << "rel kick to stage " << i << ": " << phase_offset << std::endl;
        this->outputs[i]->V = .5 * (V[VDD] + phase_offset * kick);
    }

    // solve devices' steady states
    bool ok = true;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * N; ++i) {
        ok &= (i % 2 == 0) ? n(i/2).steady_state() : p(i/2).steady_state();
    }

    // save first output voltage
    this->V_out.resize(1);
    for (int i = 0; i < N; ++i) {
        this->V_out[0][i] = this->outputs[i]->V;
    }

    return ok;
}

#endif

