#ifndef GNUPLOT_HPP
#define GNUPLOT_HPP

#include <armadillo>
#include <tuple>
#include <ext/stdio_filebuf.h>

#include "rwth.hpp"

class gnuplot_file {
public:
    std::FILE * f;
    __gnu_cxx::stdio_filebuf<char> fbuf;
    inline gnuplot_file();
    inline ~gnuplot_file();
};

class gnuplot : private gnuplot_file, public std::ostream {
public:
    inline gnuplot();

    inline void add(const std::pair<arma::vec, arma::vec> & xy);
    inline void add(const arma::vec & x, const arma::vec & y);
    inline void add(const arma::vec & y);
    inline void add(const std::pair<arma::vec, arma::cx_vec> & xy);
    inline void add(const arma::vec & x, const arma::cx_vec & y);
    inline void add(const arma::cx_vec & y);

    inline void add(const std::string title);

    inline void set_background(const std::tuple<arma::vec, arma::vec, arma::mat> & xyz);
    inline void set_background(const arma::vec & x, const arma::vec & y, const arma::mat & z);
    inline void set_background(const arma::mat & z);

    inline void plot();

    inline void reset();

private:
    std::tuple<arma::vec, arma::vec, arma::mat> background;
    bool plot_background;
    std::vector<std::pair<arma::vec, arma::vec>> data;
};

template<class ... T>
static inline void plot(const std::pair<arma::vec, T> & ... xy);

template<class T, class ... U>
static inline void plot(gnuplot & gp, const std::pair<arma::vec, T> & xy0, const std::pair<arma::vec, U>  & ... xy);

template<class T>
static inline void plot(gnuplot & gp, const std::pair<arma::vec, T> & xy);

template<class ... T>
static inline void plot(const T & ... y);

template<class T, class ... U>
static inline void plot(gnuplot & gp, const T & y0, const U & ... y);

template<class T>
static inline void plot(gnuplot & gp, const T & y);

static inline void image(std::tuple<arma::vec, arma::vec, arma::mat> & xyz);

static inline void image(const arma::vec & x, const arma::vec & y, const arma::mat & z);

static inline void image(const arma::mat & z);

//----------------------------------------------------------------------------------------------------------------------

gnuplot_file::gnuplot_file()
    : f(::popen("gnuplot -persist", "w")), fbuf(fileno(f), std::ios_base::out) {
}

gnuplot_file::~gnuplot_file() {
    ::pclose(f);
}

gnuplot::gnuplot()
    : gnuplot_file(), std::ostream(&fbuf), plot_background(false) {
    *this << rwth_plt;
}

void gnuplot::add(const std::pair<arma::vec, arma::vec> & xy) {
    data.push_back(xy);
}

void gnuplot::add(const arma::vec & x, const arma::vec & y) {
    data.push_back(std::make_pair(x, y));
}

void gnuplot::add(const arma::vec & y) {
    add(arma::linspace(0, y.size()-1, y.size()), y);
}

void gnuplot::add(const std::pair<arma::vec, arma::cx_vec> & xy) {
    add(xy.first, xy.second);
}

void gnuplot::add(const arma::vec & x, const arma::cx_vec & y) {
    add(x, arma::real(y));
    add(x, arma::imag(y));
}

void gnuplot::add(const arma::cx_vec & y) {
    add(arma::linspace(0, y.size()-1, y.size()), arma::real(y));
    add(arma::linspace(0, y.size()-1, y.size()), arma::imag(y));
}

void gnuplot::add(const std::string title) {
    *this << "set title \"" << title << "\"\n";
}

void gnuplot::set_background(const std::tuple<arma::vec, arma::vec, arma::mat> & xyz) {
    background = xyz;
    plot_background = true;
}

void gnuplot::set_background(const arma::vec & x, const arma::vec & y, const arma::mat & z) {
    set_background(std::make_tuple(x, y, z));
}

void gnuplot::set_background(const arma::mat & z) {
    auto x = arma::linspace(0, z.n_cols-1, z.n_cols);
    auto y = arma::linspace(0, z.n_rows-1, z.n_rows);
    set_background(x, y, z);
}

void gnuplot::plot() {
    #ifndef GNUPLOT_NOPLOTS
    if (plot_background) {
        auto & x = std::get<0>(background);
        auto & y = std::get<1>(background);
        auto & z = std::get<2>(background);

        *this << "set xrange[" << arma::min(x) << ":" << arma::max(x) << "]\n";
        *this << "set yrange[" << arma::min(y) << ":" << arma::max(y) << "]\n";
        *this << "p \"-\" w image notitle";
        for (unsigned i = 0; i < data.size(); ++i) *this << ", \"-\" w l ls " << i+1 << " notitle";
        *this << std::endl;
        for (unsigned i = 0; i < x.size(); ++i) {
            for (unsigned j = 0; j < y.size(); ++j) {
                *this << x(i) << " " << y(j) << " " << z(j, i) << "\n";
            }
            *this << std::endl;
        }
        *this << "e\n";
        for(unsigned i = 0; i < data.size(); ++i) {
            for(unsigned j = 0; j < data[i].first.size(); ++j) {
                *this << data[i].first[j] << " " << data[i].second[j] << std::endl;
            }
            *this << "e" << std::endl;
        }
    } else {
        if (data.size() > 0) {
            *this << "p ";
            for(unsigned i = 0; i < data.size(); ++i) {
                *this << "\"-\" w l ls " << i+1 << " notitle";
                if (i < data.size() - 1) {
                    *this << ", ";
                }
            }
            *this << std::endl;
            for(unsigned i = 0; i < data.size(); ++i) {
                for(unsigned j = 0; j < data[i].first.size(); ++j) {
                    *this << data[i].first[j] << " " << data[i].second[j] << std::endl;
                }
                *this << "e" << std::endl;
            }
        }
    }
    #endif
}

void gnuplot::reset() {
    data.clear();
    plot_background = false;
}

template<class ... T>
void plot(const std::pair<arma::vec, T> & ... xy) {
    gnuplot gp;
    plot(gp, xy ...);
}

template<class T, class ... U>
void plot(gnuplot & gp, const std::pair<arma::vec, T> & xy0, const std::pair<arma::vec, U> & ... xy) {
    gp.add(xy0);
    plot(gp, xy ...);
}

template<class T>
void plot(gnuplot & gp, const std::pair<arma::vec, T> & xy) {
    gp.add(xy);
    gp.plot();
}

template<class ... T>
void plot(const T & ... y) {
    gnuplot gp;
    plot(gp, y ...);
}

template<class T, class ... U>
void plot(gnuplot & gp, const T & y0, const U & ... y) {
    gp.add(y0);
    plot(gp, y ...);
}

template<class T>
void plot(gnuplot & gp, const T & y) {
    gp.add(y);
    gp.plot();
}

void image(std::tuple<arma::vec, arma::vec, arma::mat> & xyz) {
    gnuplot gp;
    gp.set_background(xyz);
    gp.plot();
}

void image(const arma::vec & x, const arma::vec & y, const arma::mat & z) {
    gnuplot gp;
    gp.set_background(x, y, z);
    gp.plot();
}

void image(const arma::mat & z) {
    gnuplot gp;
    gp.set_background(z);
    gp.plot();
}

#endif
