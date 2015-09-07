#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include <armadillo>
#include <queue>

template<class F>
inline auto integral(F && f, int N, const arma::vec & intervals, double tol, double tol_f, arma::vec & x, arma::vec & w);

namespace integral_impl {

    struct interval_data {
        unsigned index[5];
        arma::vec I;

        inline unsigned operator()(int i) const {
            return index[i];
        }
    };

    template<class F>
    inline auto integral_interval(F && f, int N, arma::vec & x, arma::mat & y, arma::vec & w, double tol, double tol_f);

}

//----------------------------------------------------------------------------------------------------------------------

template<class F>
auto integral(F && f, int N, const arma::vec & intervals, double tol, double tol_f, arma::vec & x, arma::vec & w) {
    using namespace arma;
    using namespace integral_impl;

    // initial x values
    vec x0 = vec(4 * intervals.size() - 3);
    x0(0) = intervals(0);
    for (unsigned i = 1; i < intervals.size(); ++i) {
        x0(4 * i - 3) = 0.75 * intervals(i - 1) + 0.25 * intervals(i);
        x0(4 * i - 2) = 0.50 * intervals(i - 1) + 0.50 * intervals(i);
        x0(4 * i - 1) = 0.25 * intervals(i - 1) + 0.75 * intervals(i);
        x0(4 * i - 0) = 0.00 * intervals(i - 1) + 1.00 * intervals(i);
    }

    // matrix to hold initial y values;
    mat y0(N, x0.size());
    vec I(N);
    I.fill(0);

    // x_i and w_i vectors
    vec x_i[intervals.size() - 1];
    vec w_i[intervals.size() - 1];

    // total number of x points
    unsigned x_n = 0;

    // enter multithreaded region
    #pragma omp parallel reduction(+ : x_n)
    {
        // local variable for intermediate integral value
        vec I_thread(N);
        I_thread.fill(0);

        // calculate y values for initial x points
        #pragma omp for schedule(static)
        for (unsigned i = 0; i < x0.size(); ++i) {
            y0.col(i) = f(x0(i));
            while(!y0.col(i).is_finite()) {
                x0(i) = std::nextafter(x0(i), x0(i) + std::copysign(1.0, x0(i)));
                y0.col(i) = f(x0(i));
            }
        }
        // implied omp barrier

        // integrate each interval; schedule(dynamic) since each interval takes a different amount of time to compute
        #pragma omp for schedule(dynamic)
        for (unsigned i = 0; i < intervals.size() - 1; ++i) {
            x_i[i] = x0.rows(4 * i, 4 * i + 4);
            w_i[i] = vec(5);
            mat y1 = y0.cols(4 * i, 4 * i + 4);

            I_thread += integral_interval(f, N, x_i[i], y1, w_i[i], tol, tol_f);
            x_n += x_i[i].size();
        }
        // implied omp barrier

        // perform reduction of intermediate integral values
        #pragma omp critical
        {
            I += I_thread;
        }
    }

    // merge x_i and w_i vectors
    x.resize(x_n - (intervals.size() - 2));
    w.resize(x_n - (intervals.size() - 2));
    unsigned i0 = 0;
    unsigned i1 = x_i[0].size();
    x.rows(i0, i1 - 1) = x_i[0];
    w.rows(i0, i1 - 1) = w_i[0];
    for (unsigned i = 1; i < intervals.size() - 1; ++i) {
        i0 = i1;
        i1 += x_i[i].size() - 1;
        x.rows(i0, i1 - 1) = x_i[i].rows(1, x_i[i].size() - 1);
        w.rows(i0, i1 - 1) = w_i[i].rows(1, w_i[i].size() - 1);

        // add overlapping weights
        w(i0 - 1) += w_i[i](0);
    }

    return I;
}

template<class F>
auto integral_impl::integral_interval(F && f, int N, arma::vec & x, arma::mat & y, arma::vec & w, double tol, double tol_f) {
    using namespace arma;

    std::queue<interval_data> queue;
    double min_h = c::epsilon(x(4) - x(0)) / 1024.0;
    bool stop_it = false;

    // return value
    vec I(N);
    I.fill(0);

    // preallocation
    unsigned n = 5;
    x.resize(25);
    y.resize(N, 25);
    w.resize(25);

    // weights
    w.fill(0.0);

    // initial simpson estimate on rough grid
    vec I1 = (x(4) - x(0)) * (y.col(0) + 4 * y.col(2) + y.col(4)) / 6;

    // put initial data on stack
    queue.push({{0, 1, 2, 3, 4}, I1});

    // loop until no more data left
    while (!queue.empty()) {
        // too many points ?
        if (x.size() > 5000) {
            stop_it = true;
        }

        // select interval
        interval_data & i = queue.front();

        // interval size
        double h = x(i(4)) - x(i(0));

        // simpson estimate on fine grid
        vec I2l = 0.5 * h * (y.col(i(0)) + 4 * y.col(i(1)) + y.col(i(2))) / 6;
        vec I2r = 0.5 * h * (y.col(i(2)) + 4 * y.col(i(3)) + y.col(i(4))) / 6;
        auto I2  = I2l + I2r;
        vec I2_abs = abs(I2);

        // difference between two estimates
        vec delta_I = I2 - i.I;

        // convergence condition
        if ((all(abs(delta_I) <= tol * I2_abs)) || (max(I2_abs) <= tol_f) || (h <= min_h) || stop_it) {
            I += I2 + delta_I / 15;

            // update weights
            w(i(0)) += h *  7 / 90;
            w(i(1)) += h * 16 / 45;
            w(i(2)) += h *  2 / 15;
            w(i(3)) += h * 16 / 45;
            w(i(4)) += h *  7 / 90;

        } else {
            // check if capacity of vectors is sufficient
            if (x.size() < n + 4) {
                unsigned capacity = x.size() * 2;
                x.resize(capacity);
                y.resize(N, capacity);
                w.resize(capacity);
            }

            // indices
            unsigned j0 = n;
            unsigned j1 = n + 4;

            // update size
            n += 4;

            // eval function at new points (4 evals)
            for (unsigned j = j0; j < j1; ++j) {
                x(j) = h / 8 + x(i(j - j0));
                y.col(j) = f(x(j));
                while(!y.col(j).is_finite()) {
                    x(j) = std::nextafter(x(j), x(j) + std::copysign(1.0, x(j)));
                    y.col(j) = f(x(j));
                }
                w(j) = 0;
            }

            // put two new intervals on the queue
            queue.push({{i(0), j1 - 4, i(1), j1 - 3, i(2)}, I2l});
            queue.push({{i(2), j1 - 2, i(3), j1 - 1, i(4)}, I2r});
        }

        // remove first element from queue
        queue.pop();
    }

    // cut off excess space
    x.resize(n);
    w.resize(n);

    // sort x and w
    // save sort_index as uvec not auto since armadillo treats it as glue => sorting of w breaks after x is sorted
    uvec s = sort_index(x);
    x = x(s);
    w = w(s);

    return I;
}

#endif
