#ifndef VOLTAGE_HPP
#define VOLTAGE_HPP

#include <array>

using ulint = unsigned long int;

template<ulint N>
using voltage = std::array<double, N>;

template<ulint N>
static inline voltage<N> operator+(const voltage<N> & a, double b) {
    voltage<N> c;
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b;
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator+(double a, const voltage<N> & b) {
    return b + a;
}
template<ulint N>
static inline voltage<N> operator+(const voltage<N> & a, const voltage<N> & b) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator-(const voltage<N> & a) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = - a[i];
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator-(const voltage<N> & a, double b) {
    return a + (-b);
}
template<ulint N>
static inline voltage<N> operator-(double a, const voltage<N> & b) {
    return a + (-b);
}
template<ulint N>
static inline voltage<N> operator-(const voltage<N> & a, const voltage<N> & b) {
    return a + (-b);
}
template<ulint N>
static inline voltage<N> operator*(const voltage<N> & a, double b) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = a[i] * b;
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator*(double a, const voltage<N> & b) {
    return b * a;
}
template<ulint N>
static inline voltage<N> operator*(const voltage<N> & a, const voltage<N> & b) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = a[i] * b[i];
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator/(const voltage<N> & a, double b) {
    return a * (1.0 / b);
}
template<ulint N>
static inline voltage<N> operator/(double a, const voltage<N> & b) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = a / b[i];
    }
    return c;
}
template<ulint N>
static inline voltage<N> operator/(const voltage<N> & a, const voltage<N> & b) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = a[i] / b[i];
    }
    return c;
}
template<ulint N, class F>
static inline voltage<N> func(F && f, const voltage<N> & a) {
    voltage<N> c;
    for (ulint i = 0; i < N; ++i) {
        c[i] = f(a[i]);
    }
    return c;
}

#endif

