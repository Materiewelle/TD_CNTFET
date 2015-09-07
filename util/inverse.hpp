#ifndef INVERSE_HPP
#define INVERSE_HPP

#include <armadillo>

namespace inverse_impl {

    using arma::vec;
    using arma::cx_vec;
    using arma::mat;
    using arma::cx_mat;

    template<class T, class U>
    class ret_type;

    template<>
    class ret_type<vec, vec> {
    public:
        using type = vec;
    };
    template<>
    class ret_type<vec, cx_vec> {
    public:
        using type = cx_vec;
    };
    template<>
    class ret_type<cx_vec, vec> {
    public:
        using type = cx_vec;
    };
    template<>
    class ret_type<cx_vec, cx_vec> {
    public:
        using type = cx_vec;
    };

    template<class T, class U>
    class ret_type_n;

    template<>
    class ret_type_n<vec, vec> {
    public:
        using type = mat;
    };
    template<>
    class ret_type_n<vec, cx_vec> {
    public:
        using type = cx_mat;
    };
    template<>
    class ret_type_n<cx_vec, vec> {
    public:
        using type = cx_mat;
    };
    template<>
    class ret_type_n<cx_vec, cx_vec> {
    public:
        using type = cx_mat;
    };

}

template<bool first, class T, class U>
inline auto inverse_col(const T & A, const U & B) {
    int n = B.size();

    auto I = typename inverse_impl::ret_type<T, U>::type(n);

    auto a = A.memptr();
    auto b = B.memptr();
    auto c = I.memptr();

    if (!first) {
        c[0] = a[0] / b[0];

        for (int i = 1; i < n - 1; ++i) {
            c[i] = a[i] / (b[i] - a[i - 1] * c[i - 1]);
        }
        c[n - 1] = 1.0 / (b[n - 1] - a[n - 2] * c[n - 2]);

        for (int i = n - 2; i >= 0; --i) {
            c[i] *= - c[i + 1];
        }
    } else {
        c[n - 1] = a[n - 2] / b[n - 1];

        for (int i = n - 2; i >= 1; --i) {
            c[i] = a[i - 1] / (b[i] - a[i] * c[i + 1]);
        }
        c[0] = 1.0 / (b[0] - a[0] * c[1]);

        for (int i = 1; i < n; ++i) {
            c[i] *= - c[i - 1];
        }
    }

    return I;
}

template<class T, class U>
inline auto inverse_diag(const T & A, const U & B) {
    int n = B.size();

    auto I = typename inverse_impl::ret_type<T, U>::type(n);

    auto a = A.memptr();
    auto b = B.memptr();
    auto c = I.memptr();

    c[0] = a[0] / b[0];

    for (int i = 1; i < n - 1; ++i) {
        c[i] = a[i] / (b[i] - a[i - 1] * c[i - 1]);
    }
    c[n - 1] = 1.0 / (b[n - 1] - a[n - 2] * c[n - 2]);

    for (int i = n - 2; i >= 0; --i) {
        c[i] = c[i] / a[i] * (1.0 + c[i] * c[i + 1] * a[i]);
    }

    return I;
}

template<bool first, int N, class T, class U>
inline auto inverse_cols(const T & A, const U & B) {
    int n = B.size();

    auto I = typename inverse_impl::ret_type_n<T, U>::type(n, N);

    auto a = A.memptr();
    auto b = B.memptr();

    if (!first) {
        auto c = I.colptr(N - 1);

        c[0] = a[0] / b[0];

        for (int i = 1; i < n - 1; ++i) {
            c[i] = a[i] / (b[i] - a[i - 1] * c[i - 1]);
        }
        c[n - 1] = 1.0 / (b[n - 1] - a[n - 2] * c[n - 2]);

        for (int i = n - 2; i >= n - N; --i) {
            I(i, i-n+N) = c[i] / a[i] * (1.0 + c[i] * I(i+1, i-n+N+1) * a[i]);
            for (int j = i + 1; j < n; ++j) {
                I(j, i-n+N) = - I(j, i-n+N+1) * c[i];
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = n - N - 1 + i; j >= 0; --j) {
                I(j, i) = - I(j + 1, i) * c[j];
            }
        }

    } else {
        auto c = I.colptr(0);

        c[n - 1] = a[n - 2] / b[n - 1];

        for (int i = n - 2; i >= 1; --i) {
            c[i] = a[i - 1] / (b[i] - a[i] * c[i + 1]);
        }
        c[0] = 1.0 / (b[0] - a[0] * c[1]);

        for (int i = 1; i < N; ++i) {
            I(i, i) = c[i] / a[i-1] * (1.0 + c[i] * I(i-1, i-1) * a[i-1]);
            for (int j = 0; j < i; ++j) {
                I(j, i) = - I(j, i - 1) * c[i];
            }
        }

        for (int i = N - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                I(j, i) = - I(j - 1, i) * c[j];
            }
        }

    }

    return I;
}

#endif
