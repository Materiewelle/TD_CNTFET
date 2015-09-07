#ifndef BRENT_HPP
#define BRENT_HPP

#include <cmath>

template<class F>
static inline bool brent(F && f, double a, double b, double tol, double & x0) {
    using namespace std;

    double f_a = f(a);
    double f_b = f(b);

//    if (f_a * f_b >= 0) {
//        return false;
//    }

    if (abs(f_a) < abs(f_b)) {
        // swap a and b, f_a and f_b
        double t = a;
        a = b;
        b = t;
        t = f_a;
        f_a = f_b;
        f_b = t;
    }

    double c = a;
    double f_c = f_a;
    bool mflag = true;

    double d;

    double s = b;
    double f_s = f_b;

    while((f_b != 0) && (f_s != 0) && (abs(b - a) > tol)) {
        if ((f_a != f_c) && (f_b != f_c)) {
            // inverse quadratic interpolation
            s = a * f_b * f_c / ((f_a - f_b) * (f_a - f_c))
              + b * f_a * f_c / ((f_b - f_a) * (f_b - f_c))
              + c * f_a * f_b / ((f_c - f_a) * (f_c - f_b));
        } else {
            // secant method
            s = b - f_b * (b - a) / (f_b - f_a);
        }

        double t = 0.25 * (3 * a + b);


        if ((((s < t) && (s < b)) || ((s > t) && (s > b))) ||
            (mflag && (abs(s - b) >= 0.5 * abs(b - c))) ||
            (!mflag && (abs(s - b) >= 0.5 * abs(c - d))) ||
            (mflag && (abs(b - c) < tol)) ||
            (!mflag && (abs(c - d) < tol))) {
            s = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }

        f_s = f(s);
        d = c;
        c = b;
        f_c = f_b;
        if (f_a * f_s < 0) {
            b = s;
            f_b = f_s;
        } else {
            a = s;
            f_a = f_s;
        }
        if (abs(f_a) < abs(f_b)) {
            // swap a and b, f_a and f_b
            double t = a;
            a = b;
            b = t;
            t = f_a;
            f_a = f_b;
            f_b = t;
        }
    }

    x0 = s;
    return true;
}

#endif
