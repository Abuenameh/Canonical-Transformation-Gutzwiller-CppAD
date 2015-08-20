/* 
 * File:   cppad.hpp
 * Author: Abuenameh
 *
 * Created on August 20, 2015, 4:06 PM
 */

#ifndef CPPAD_HPP
#define	CPPAD_HPP

#include <vector>
#include <string>
using std::vector;
using std::string;
using std::copy;

#include <boost/date_time.hpp>
using namespace boost::posix_time;

#include <cppad/cppad.hpp>
using CppAD::AD;
using CppAD::ADFun;
using CppAD::Independent;

inline double g(int n, int m) {
    return sqrt(1.0 * (n + 1) * m);
}

inline double eps(vector<double>& U, int i, int j, int n, int m) {
    return n * U[i] - (m - 1) * U[j];
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}

inline double eps(vector<double>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n * U[i] - (m - 1) * U[j] + (q - 1) * U[k] - p * U[l];
}

class GroundStateProblem {
public:
    GroundStateProblem() {}

    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu, double theta);
//    void setTheta(double theta);

    ADFun<double>* E() {
        return Efunc;
    }

    string& getStatus() {
        return status;
    }
    string getRuntime();

    void start() {
        start_time = microsec_clock::local_time();
    }

    void stop() {
        stop_time = microsec_clock::local_time();
    }

//    static void setup();

private:

    ptime start_time;
    ptime stop_time;

    template<class T> static T energy(CppAD::vector<T>& fin, vector<double>& J, double U0, vector<double>& dU, double mu, double theta);

    string status;
    double runtime;

    ADFun<double>* Efunc;
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CPPAD_HPP */

