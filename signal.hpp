#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <vector>

#include "constant.hpp"
#include "voltage.hpp"

template<ulint N>
class signal {
public:
    int N_t;
    double T;
    std::vector<voltage<N>> V;

    inline signal();
    inline signal(double T);
    inline signal(double T, const voltage<N> & V);

    inline voltage<N> & operator[](int index);
    inline const voltage<N> & operator[](int index) const;
};

// append two signals
template<ulint N>
static inline signal<N> operator+(const signal<N> & s1, const signal<N> & s2);

// create a linear signal: V[i] = V0 + (V1 - V0) * t_i / T
template<ulint N>
static inline signal<N> linear_signal(double T, const voltage<N> & V0, const voltage<N> & V1);

// create a sine signal: V[i] = V0 + VA * sin(f * t_i  + ph)
template<ulint N>
static inline signal<N> sine_signal(double T, const voltage<N> & V0, const voltage<N> & VA, const double f, const double ph = 0);

// create a square wave signal: V[i] = V0 + VA * sgn[sin(f * t_i  + ph)]
// NOTE: signal will be clipped to complete oscillations!
template<ulint N>
static inline signal<N> square_signal(double T, const voltage<N> & V0, const voltage<N> & VA, const double f, const double t_rise, const double t_fall);

//----------------------------------------------------------------------------------------------------------------------

template<ulint N>
signal<N>::signal() {
}
template<ulint N>
signal<N>::signal(double T_)
    :  N_t(std::round(T_ / c::dt)), T(N_t * c::dt), V(N_t) {
}
template<ulint N>
signal<N>::signal(double T_, const voltage<N> & V_)
    : signal(T_) {
    std::fill(V.begin(), V.end(), V_);
}

template<ulint N>
voltage<N> & signal<N>::operator[](int index) {
    return V[index];
}
template<ulint N>
const voltage<N> & signal<N>::operator[](int index) const {
    return V[index];
}

template<ulint N>
signal<N> operator+(const signal<N> & s1, const signal<N> & s2) {
    signal<N> s3(s1.T + s2.T);
    std::copy(s1.V.begin(), s1.V.end(), s3.V.begin());
    std::copy(s2.V.begin(), s2.V.end(), s3.V.begin() + s1.V.size());
    return s3;
}

template<ulint N>
signal<N> linear_signal(double T, const voltage<N> & V0, const voltage<N> & V1) {
    signal<N> s(T);

    for (int i = 0; i < s.N_t; ++i) {
        double r = ((double)i) / ((double)(s.N_t - 1));
        s[i] = V0 * (1.0 - r) + V1 * r;
    }

    return s;
}
template<ulint N>
signal<N> sine_signal(double T, const voltage<N> & V0, const voltage<N> & VA, const double f, const double ph) {
    signal<N> s(T);

    for (int i = 0; i < s.N_t; ++i) {
        double t = i * c::dt;
        s[i] = V0 + VA * std::sin(t * 2 * M_PI * f + ph);
    }

    return s;
}
template<ulint N>
signal<N> square_signal(double T, const voltage<N> & V0, const voltage<N> & V1, const double f, const double t_rise, const double t_fall) {
    signal<N> s(c::dt, V0);

    signal<N> rise = linear_signal(t_rise, V0, V1);
    signal<N> fall = linear_signal(t_fall, V1, V0);
    signal<N>  low(.5/f - t_rise, V0);
    signal<N> high(.5/f - t_fall, V1);

    signal<N> osci = low + rise + high + fall;

    int n_osci = std::floor(T * f);

    for (int i = 0; i < n_osci; ++i) {
        s = s + osci;
    }

    return s;
}

static void sigplot(signal<3> sig) {
    using namespace arma;
    vec vs(sig.N_t);
    vec vd(sig.N_t);
    vec vg(sig.N_t);
    for (int i = 0; i < sig.N_t; ++i) {
        vs(i) = sig.V[i][S];
        vd(i) = sig.V[i][D];
        vg(i) = sig.V[i][G];
    }
    plot(vs, vd, vg);
}

static void quasi_static(const signal<3> & sig, const device_params & p, int N = 400) {
    int step = sig.N_t / N;
    std::vector<voltage<3>> points(N);
    for (int i = 0; i < N; ++i) {
        points[i] = sig.V[i * step];
    }
    // get current points in parallel
    std::vector<current> i_quasi = curve(p, points);
    // save for plotting
    arma::mat CSV (N, 2);
    for (int j = 0; j < N; ++j) {
        CSV(j, 0) = c::dt * step;
        CSV(j, 1) = i_quasi[j].total(0);
    }
    CSV.save(save_folder() + "/I_quasi.csv", arma::csv_ascii);
}

#endif

