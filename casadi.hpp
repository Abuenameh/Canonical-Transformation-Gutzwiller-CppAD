/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <casadi/casadi.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include "gutzwiller.hpp"

inline double g2(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline double eps(vector<double>& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

inline double eps(double U, int n, int m) {
    return (n - m + 1) * U;
}

inline SX g(int n, int m) {
    return sqrt(1.0*(n + 1) * m);
}

inline SX eps(vector<SX>& U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

inline SX eps(SX& U, int n, int m) {
    return (n - m + 1) * U;
}

inline SX eps(vector<SX>& U, int i, int j, int n, int m, int k, int l, int p, int q) {
    return n*U[i] - (m-1)*U[j] + (q-1)*U[k] - p*U[l];
}

class GroundStateProblem {
public:
    GroundStateProblem();
    
    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
    void setTheta(double theta);
    
    double E(const vector<double>& f, vector<double>& grad);
    
    string& getStatus() { return status; }
    string getRuntime();
    
    void start() { start_time = microsec_clock::local_time(); }
    void stop() { stop_time = microsec_clock::local_time(); }
    
    static void setup();
    
    static double energy2(vector<double>& fin, vector<double>& J, double U0, vector<double>& dU, double mu, double theta);
    static double energy2(int i, int n, vector<double>& fin, vector<double>& J, double U0, vector<double>& dU, double mu, double theta);
    
private:
    
    ptime start_time;
    ptime stop_time;
    
    static SX energy(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta);
    static SX energy(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu, SX& theta);
    
    vector<SX> fin;
    SX U0;
    vector<SX> dU;
    vector<SX> J;
    SX mu;
    SX theta;
    
    SX x;
    SX p;
    
    vector<double> params;
    
    vector<Function> Ef;
    vector<Function> Egradf;
    
    string status;
    double runtime;
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

