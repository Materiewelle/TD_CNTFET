#ifndef WAVE_PACKET_HPP
#define WAVE_PACKET_HPP

#include <armadillo>

#include "constant.hpp"
#include "device_params.hpp"

class wave_packet {
public:
    arma::vec E0;
    arma::vec F0;
    arma::vec W;
    arma::cx_mat * data;
    arma::mat E;

    inline wave_packet();
    inline wave_packet(const device_params & p, int src, unsigned mem, const arma::vec & E, const arma::vec & W, const potential & phi);
    inline wave_packet(const wave_packet & psi);
    inline wave_packet(wave_packet && psi);

    inline wave_packet & operator=(const wave_packet & psi);
    inline wave_packet & operator=(wave_packet && psi);

    inline void memory_init();
    inline void memory_update(const arma::cx_mat & affe, unsigned m);

    inline void source_init(const device_params & p, const arma::cx_mat & u, const arma::cx_mat & q);
    inline void source_update(const arma::cx_mat & u, const arma::cx_mat & L, const arma::cx_mat & qsum, unsigned m);

    inline void propagate(const arma::cx_mat & U_eff, const arma::cx_mat & inv);
//    inline void propagate(const arma::cx_vec & U_eff, const arma::cx_mat & inv);

    inline void remember();

    inline void update_sum(unsigned m);
    inline void update_E(const device_params & p, const potential & phi, const potential & phi0);

private:
    bool left;
    unsigned mem;

    arma::cx_mat in;
    arma::cx_mat out;

    arma::cx_cube sum;

    arma::cx_mat source;
    arma::cx_mat memory;

    // from previous time-step
    arma::cx_mat old_source;
    arma::cx_mat * old_data;

    arma::cx_mat data1;
    arma::cx_mat data2;
};

//----------------------------------------------------------------------------------------------------------------------

wave_packet::wave_packet() {
}
wave_packet::wave_packet(const device_params & p, int src, unsigned mem_, const arma::vec & E_, const arma::vec & W_, const potential & phi)
    : E0(E_), F0(fermi(E0 - ((src == S) ? phi.s() : phi.d()), p.F[src])), W(W_), left(src == S), mem(mem_) {
    using namespace arma;

    data1 = cx_mat(p.N_x * 2, E0.size());
    data2 = cx_mat(p.N_x * 2, E0.size());
    data = &data1;
    E = mat(p.N_x, E0.size());
    in = cx_mat(E0.size(), 2);
    out = cx_mat(E0.size(), 2);
    sum = cx_cube(mem, E0.size(), 2);
    source = cx_mat(E0.size(), 2);
    memory = cx_mat(E0.size(), 2);

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < E0.size(); ++i) {
        // calculate 1 column of green's function
        cx_double Sigma_s, Sigma_d;
        cx_vec G;

        // calculate wave function
        if (left) {
            G = green_col<true>(p, phi, E0(i), Sigma_s, Sigma_d);
            G *= std::sqrt(cx_double(- 2 * Sigma_s.imag()));
        } else {
            G = green_col<false>(p, phi, E0(i), Sigma_s, Sigma_d);
            G *= std::sqrt(cx_double(- 2 * Sigma_d.imag()));
        }

        // extract data
        data->col(i) = G;
        in.col(S)(i) = G(0);
        in.col(D)(i) = G(G.size() - 1);

        // calculate first layer in the leads analytically
        out.col(S)(i) = ((E0(i) - phi.s()) * in.col(S)(i) - p.tc1 * G(           1)) / p.tc2;
        out.col(D)(i) = ((E0(i) - phi.d()) * in.col(D)(i) - p.tc1 * G(G.size() - 2)) / p.tc2;
    }

    update_E(p, phi, phi);
}
wave_packet::wave_packet(const wave_packet & psi) :
    E0(psi.E0),
    F0(psi.F0),
    W(psi.W),
    E(psi.E),
    left(psi.left),
    mem(psi.mem),
    in(psi.in),
    out(psi.out),
    sum(psi.sum),
    source(psi.source),
    memory(psi.memory),
    old_source(psi.old_source),
    data1(psi.data1),
    data2(psi.data2) {
    data = (psi.data == &psi.data1) ? &data1 : ((psi.data == &psi.data2) ? &data2 : nullptr);
    old_data = (psi.old_data == &psi.data1) ? &data1 : ((psi.old_data == &psi.data2) ? &data2 : nullptr);
}

wave_packet::wave_packet(wave_packet && psi) :
    E0(std::move(psi.E0)),
    F0(std::move(psi.F0)),
    W(std::move(psi.W)),
    E(std::move(psi.E)),
    left(psi.left),
    mem(psi.mem),
    in(std::move(psi.in)),
    out(std::move(psi.out)),
    sum(std::move(psi.sum)),
    source(std::move(psi.source)),
    memory(std::move(psi.memory)),
    old_source(std::move(psi.old_source)),
    data1(std::move(psi.data1)),
    data2(std::move(psi.data2)) {
    data = (psi.data == &psi.data1) ? &data1 : ((psi.data == &psi.data2) ? &data2 : nullptr);
    old_data = (psi.old_data == &psi.data1) ? &data1 : ((psi.old_data == &psi.data2) ? &data2 : nullptr);
}

wave_packet & wave_packet::operator=(const wave_packet & psi) {
    // "Don't do it in practice. The whole thing is ugly beyond description."
    new(this) wave_packet(psi);
    return *this;
}
wave_packet & wave_packet::operator=(wave_packet && psi) {
    new(this) wave_packet(psi);
    return *this;
}

void wave_packet::memory_init() {
    memory.fill(0.0);
}
void wave_packet::memory_update(const arma::cx_mat & affe, unsigned m) {
    if (m <= mem) {
        memory.col(S) = (affe.col(S).st() * sum.slice(S).rows(0, m - 2)).st();
        memory.col(D) = (affe.col(D).st() * sum.slice(D).rows(0, m - 2)).st();
    } else {
        unsigned n = (m - 1) % mem;
        memory.col(S) = (affe.submat(0, S, mem - n - 1, S).st() * sum.slice(S).rows(n, mem - 1)).st();
        memory.col(D) = (affe.submat(0, D, mem - n - 1, D).st() * sum.slice(D).rows(n, mem - 1)).st();
        if (n > 0) {
            memory.col(S) += (affe.submat(mem - n, S, mem - 1, S).st() * sum.slice(S).rows(0, n - 1)).st();
            memory.col(D) += (affe.submat(mem - n, D, mem - 1, D).st() * sum.slice(D).rows(0, n - 1)).st();
        }
    }
}

void wave_packet::source_init(const device_params & p, const arma::cx_mat & u, const arma::cx_mat & q) {
    using namespace std::complex_literals;

    static constexpr double g = c::g;

    source.col(S) = - 2i * g * u(0, S) * (p.tc2 * out.col(S) + 1i * g * q(0, S) * in.col(S)) / (1.0 + 1i * g * E0);
    source.col(D) = - 2i * g * u(0, D) * (p.tc2 * out.col(D) + 1i * g * q(0, D) * in.col(D)) / (1.0 + 1i * g * E0);
}
void wave_packet::source_update(const arma::cx_mat & u, const arma::cx_mat & L, const arma::cx_mat & qsum, unsigned m) {
    using namespace std::complex_literals;

    static constexpr double g = c::g;
    static constexpr double g2 = g * g;

    int n = mem + 1 - m;
    auto qss = (n >= 0) ? qsum(n, S) : 0.0;
    auto qsd = (n >= 0) ? qsum(n, D) : 0.0;

    auto E0p = 1 + 1i * g * E0;
    auto E0m = 1 - 1i * g * E0;

    source.col(S) = (old_source.col(S) % E0m * u(m - 1, S) * u(m - 2, S) + 2 * g2 * L(0, S) / u(m - 1, S) * qss * in.col(S)) / E0p;
    source.col(D) = (old_source.col(D) % E0m * u(m - 1, D) * u(m - 2, D) + 2 * g2 * L(0, D) / u(m - 1, D) * qsd * in.col(D)) / E0p;
}

void wave_packet::propagate(const arma::cx_mat & U_eff, const arma::cx_mat & inv) {
    *data = U_eff * (*old_data)
          + arma::kron(source.col(S).st() + memory.col(S).st(), inv.col(S))
          + arma::kron(source.col(D).st() + memory.col(D).st(), inv.col(D));
}

//void wave_packet::propagate(const arma::cx_vec & U_eff, const arma::cx_mat & inv) {
//    using namespace arma;
//    using namespace simd;

//    const cx_double * U = U_eff.memptr();
//    const cx_double * A = old_data->memptr();
//    int N = data->n_rows;
//    int M = data->n_cols;
//    cx_double * B = data->memptr();

//    // multiply old_data by U_eff and save result in data
//    for (int i = 0; i < M; ++i) {
//        for (int j = 0; j < N; ++j) {
//            int k  = (j < c::bw) ? 0 : (j - c::bw + 1);
//            int k1 = (j > N - c::bw) ? N : (j + c::bw);


//#if defined(__AVX__) || defined(__SSE2__)
//            m128d S1 = zero_m128d();
//            m128d S2 = zero_m128d();

//            for (; k <= j; ++k) {
//                m128d Uk = load_m128d((double *)(U + k * (c::bw - 1) + j));
//                m128d Ak = load_m128d((double *)(A + k));
//                m128d Al = shufpd<1, 0>(Ak, Ak);
//                m128d P1 = mulpd(Uk, Ak);
//                m128d P2 = mulpd(Uk, Al);
//                S1 = addpd(S1, P1);
//                S2 = addpd(S2, P2);
//            }

//# if defined(__AVX__)
//            m256d S1_256 = m128d_to_m256d(S1);
//            m256d S2_256 = m128d_to_m256d(S2);
//            for (; k < k1 - 1; k += 2) {
//                m256d Uk = load_m256d((double *)(U + j * (c::bw - 1) + k));
//                m256d Ak = load_m256d((double *)(A + k));
//                m256d Al = vshufpd<1, 0, 1, 0>(Ak, Ak);
//                m256d P1 = vmulpd(Uk, Ak);
//                m256d P2 = vmulpd(Uk, Al);
//                S1_256 = vaddpd(S1_256, P1);
//                S2_256 = vaddpd(S2_256, P2);
//            }
//            S1 = addpd(vextractf128<0>(S1_256), vextractf128<1>(S1_256));
//            S2 = addpd(vextractf128<0>(S2_256), vextractf128<1>(S2_256));
//# endif
//            for (; k < k1; ++k) {
//                m128d Uk = load_m128d((double *)(U + j * (c::bw - 1) + k));
//                m128d Ak = load_m128d((double *)(A + k));
//                m128d Al = shufpd<1, 0>(Ak, Ak);
//                m128d P1 = mulpd(Uk, Ak);
//                m128d P2 = mulpd(Uk, Al);
//                S1 = addpd(S1, P1);
//                S2 = addpd(S2, P2);
//            }

//            // reduction
//            m128d R = hsubpd(S1, S1);
//            m128d I = haddpd(S2, S2);
//            m128d T = shufpd<0, 0>(R, I);

//            // save
//            store_m128d((double *)(B + j), T);
//#else
//            B[j] = 0.0;
//            for (; k <= j; ++k) {
//                B[j] += U[k * (c::bw - 1) + j] * A[k];
//            }
//            for (; k < k1; ++k) {
//                B[j] += U[j * (c::bw - 1) + k] * A[k];
//            }
//#endif
//        }
//        A += N;
//        B += N;
//    }

//    // add source and memory terms
//    *data += kron(source.col(S).st() + memory.col(S).st(), inv.col(S))
//           + kron(source.col(D).st() + memory.col(D).st(), inv.col(D));
//}

void wave_packet::remember() {
    if (data == &data1) {
        data     = &data2;
        old_data = &data1;
    } else {
        data     = &data1;
        old_data = &data2;
    }
    old_source = source;
}

void wave_packet::update_sum(unsigned m) {
    int n = (m - 1) % mem;
    int end = old_data->n_rows - 1;

    sum.slice(S).row(n) = old_data->row(  0) + data->row(  0);
    sum.slice(D).row(n) = old_data->row(end) + data->row(end);
}
void wave_packet::update_E(const device_params & p, const potential & phi, const potential & phi0) {
    using namespace std;

    for (unsigned i = 0; i < E.n_cols; ++i) {
        for (unsigned j = 1; j < E.n_rows - 1; ++j) {
            double n = 1.0 / (norm((*data)(2 * j, i)) + norm((*data)(2 * j + 1, i)));
            double m1 = 2 * (real((*data)(2 * j, i)) * real((*data)(2 * j + 1, i)) + imag((*data)(2 * j, i)) * imag((*data)(2 * j + 1, i)));
            arma::cx_double m2 = conj((*data)(2 * j, i)) * (*data)(2 * j - 1, i);
            arma::cx_double m3 = conj((*data)(2 * j + 1, i)) * (*data)(2 * j + 2, i);
            E(j, i) = phi(j) + n * (p.t_vec(2 * j) * m1 + real(p.t_vec(2 * j - 1) * m2 + p.t_vec(2 * j + 1) * m3));
        }
    }

    if (left) {
        for (unsigned i = 0; i < E.n_cols; ++i) {
            E(0, i) = E0(i) + phi.s() - phi0.s();
            E(E.n_rows - 1, i) = E(E.n_rows - 2, i);
        }
    } else {
        for (unsigned i = 0; i < E.n_cols; ++i) {
            E(E.n_rows - 1, i) = E0(i) + phi.d() - phi0.d();
            E(0, i) = E(1, i);
        }
    }
}

#endif

