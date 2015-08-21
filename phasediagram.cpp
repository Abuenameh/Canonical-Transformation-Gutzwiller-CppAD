#include <queue>
#include <tuple>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include <cppad/cppad.hpp>
using CppAD::thread_alloc;
using CppAD::parallel_ad;
using CppAD::CheckSimpleVector;
using CppAD::one_element_std_set;
using CppAD::two_element_std_set;


//#include "casadi.hpp"
#include "cppad.hpp"
#include "gutzwiller.hpp"
#include "mathematica.hpp"

typedef boost::array<double, L> Parameter;

typedef std::tuple<double, double, int, int> Sample;

double M = 1000;
double g13 = 2.5e9;
double g24 = 2.5e9;
double delta = 1.0e12;
double Delta = -2.0e10;
double alpha = 1.1e7;

double Ng = sqrt(M) * g13;

double JW(double W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

double JWij(double Wi, double Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

Parameter JW(Parameter W) {
    Parameter v;
    for (int i = 0; i < L; i++) {
        v[i] = W[i] / sqrt(Ng * Ng + W[i] * W[i]);
    }
    Parameter J;
    for (int i = 0; i < L - 1; i++) {
        J[i] = alpha * v[i] * v[i + 1];
    }
    J[L - 1] = alpha * v[L - 1] * v[0];
    return J;
}

double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

Parameter UW(Parameter W) {
    Parameter U;
    for (int i = 0; i < L; i++) {
        U[i] = -2 * (g24 * g24) / Delta * (Ng * Ng * W[i] * W[i]) / ((Ng * Ng + W[i] * W[i]) * (Ng * Ng + W[i] * W[i]));
    }
    return U;
}

boost::mutex progress_mutex;
boost::mutex points_mutex;
boost::mutex problem_mutex;

struct Point {
    double x;
    double mu;
};

struct PointResults {
    double W;
    double mu;
    double E0;
    double Eth;
    double E2th;
    double fs;
    double Jx;
    double Ux;
    vector<double> J;
    vector<double> U;
    double fmin;
    vector<double> fn0;
    vector<double> fmax;
    vector<double> f0;
    vector<double> fth;
    vector<double> f2th;
    string status0;
    string statusth;
    string status2th;
    string runtime0;
    string runtimeth;
    string runtime2th;
    double theta;
    //    double runtime0;
    //    double runtimeth;
    //    double runtime2th;
};

vector<double> norm(vector<double>& x) {
    vector<const complex<double>*> f(L);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const complex<double>*> (&x[2 * i * dim]);
    }

    vector<double> norms(L);

    for (int i = 0; i < L; i++) {
        double normi = 0;
        for (int n = 0; n <= nmax; n++) {
            normi += norm(f[i][n]);
        }
        norms[i] = sqrt(normi);
    }
    return norms;
}

//boost::random::mt19937 xrng;
//boost::random::uniform_real_distribution<> xuni(0, 1);
//
//double randx() {
//    return xuni(xrng);
//}

//class energyprob : public base {
//public:
//    energyprob(GroundStateProblem& prob_) : prob(prob_) {}
//    base_ptr clone() const;
//    
//    void objfun_impl(fitness_vector& f, const decision_vector& x) {
//        
//    }
//    
//private:
//    GroundStateProblem& prob;
//};

bool parallel = false;

bool in_parallel() {
    return parallel;
}

class thread_id {
public:
    int id;

    thread_id(int i) : id(i) {
    }
};

boost::thread_specific_ptr<thread_id> tls;

size_t thread_num() {
    return tls->id;
}

void phasepoints(int thread, Parameter& xi, double theta, queue<Point>& points, vector<PointResults>& pres, progress_display& progress) {
    tls.reset(new thread_id(thread));

    int ndim = 2 * L * dim;

    boost::random::mt19937 xrng;
    xrng.seed(time(NULL));
    boost::random::uniform_real_distribution<> xuni(0, 1);
    vector<double> xrand(ndim);
    for (int i = 0; i < ndim; i++) {
        xrand[i] = xuni(xrng);
    }

    vector<double> x(ndim);
    vector<complex<double>*> f(L);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<double>*> (&x[2 * i * dim]);
    }

    vector<double> U(L), J(L), dU(L);

    vector<double> x0(ndim), xth(ndim), x2th(ndim);
    vector<complex<double>*> f0(L);
    for (int i = 0; i < L; i++) {
        f0[i] = reinterpret_cast<complex<double>*> (&x0[2 * i * dim]);
    }

    vector<vector<double> > fabs(L, vector<double>(dim));

    vector<double> fn0(L);
    vector<double> fmax(L);

    vector<double> norms(L);

    double scale = 1;

//    GroundStateProblem* prob;
    GroundStateProblem prob;
    opt lopt(LD_LBFGS, ndim);
//    opt lopt(LN_SBPLX, ndim);
    //    opt lopt(LD_CCSAQ, ndim);
    opt gopt(GN_DIRECT, ndim);
    //    energyprob eprob(ndim);
    //    pagmo::algorithm::de_1220 algo(100);
    //    int npop = 20;
    {
        boost::mutex::scoped_lock lock(problem_mutex);
//        prob = new GroundStateProblem();

        lopt.set_lower_bounds(-1);
        lopt.set_upper_bounds(1);
        lopt.set_min_objective(energyfunc, &prob);
//        lopt.set_ftol_abs(1e-15);
        lopt.set_ftol_rel(1e-15);
//        lopt.set_xtol_abs(1e-30);
//        lopt.set_xtol_rel(1e-16);
        gopt.set_lower_bounds(-1);
        gopt.set_upper_bounds(1.1);
        gopt.set_min_objective(energyfunc, &prob);
        gopt.set_maxtime(120);
        //                lopt.set_maxtime(120);
        //                lopt.set_ftol_abs(1e-17);
        //                lopt.set_ftol_rel(1e-17);
        //                lopt.set_xtol_abs(1e-17);
        //                lopt.set_xtol_rel(1e-17);

        //                eprob.setProblem(prob);
    }

    for (;;) {
        Point point;
        {
            boost::mutex::scoped_lock lock(points_mutex);
            if (points.empty()) {
                break;
            }
            point = points.front();
            points.pop();
        }

        PointResults pointRes;
        pointRes.W = point.x;
        pointRes.mu = point.mu;

        vector<double> W(L);
        for (int i = 0; i < L; i++) {
            W[i] = xi[i] * point.x;
        }
        double U0 = 1 / scale;
        //        for (int i = 0; i < L; i++) {
        //            U[i] = UW(W[i]) / UW(point.x) / scale;
        //            U0 += U[i] / L;
        //        }
        for (int i = 0; i < L; i++) {
            U[i] = UW(W[i]) / UW(point.x) / scale;
            //            U[i] = 1 / scale;
            dU[i] = U[i] - U0;
            J[i] = JWij(W[i], W[mod(i + 1)]) / UW(point.x) / scale;
            //            J[i] = JWij(point.x, point.x) / UW(point.x) / scale;
        }
//        cout << endl << ::math(dU) << endl << endl;
//        cout << endl << ::math(J) << endl << endl;
//        cout << endl << point.mu << endl << endl;
        pointRes.Ux = UW(point.x);
        pointRes.Jx = JWij(point.x, point.x);
        pointRes.J = J;
        pointRes.U = U;

        //        fill(x0.begin(), x0.end(), 0.5);
        //        fill(xth.begin(), xth.end(), 0.5);
        //        fill(x2th.begin(), x2th.end(), 0.5);
        //        generate(x0.begin(), x0.end(), randx);
        //        generate(xth.begin(), xth.end(), randx);
        //        generate(x2th.begin(), x2th.end(), randx);
        x0 = xrand;
        xth = xrand;
        x2th = xrand;

//        prob->setParameters(U0, dU, J, point.mu / scale);
        prob.setParameters(U0, dU, J, point.mu / scale, 0);

        //        generate(x0.begin(), x0.end(), randx);
        //        generate(xth.begin(), xth.end(), randx);
        //        generate(x2th.begin(), x2th.end(), randx);

//        prob->setTheta(0);

        double E0;
        string result0;
        try {
            prob.start();
//            prob->start();
            //            population pop0(eprob, npop);
            //            algo.evolve(pop0);
            //            E0 = pop0.champion().f[0];
            //            x0 = pop0.champion().x;
//                        result gres = gopt.optimize(x0, E0);
            result res = lopt.optimize(x0, E0);
//            prob->stop();
            prob.stop();
            result0 = to_string(res);
            //            E0 = prob->solve(x0);
        }
        catch (std::exception& e) {
//            prob->stop();
            prob.stop();
            result res = lopt.last_optimize_result();
            result0 = to_string(res) + ": " + e.what();
            printf("nlopt failed for E0 at %f, %f\n", point.x, point.mu);
            cout << e.what() << endl;
            E0 = numeric_limits<double>::quiet_NaN();
        }
        //        cout << ::math(x0) << endl;
        pointRes.status0 = result0;
        //        pointRes.status0 = prob->getStatus();
//        pointRes.runtime0 = prob->getRuntime();
        pointRes.runtime0 = prob.getRuntime();

        norms = norm(x0);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * dim + n)] /= norms[i];
                x0[2 * (i * dim + n) + 1] /= norms[i];
            }
            transform(f0[i], f0[i] + dim, fabs[i].begin(), std::ptr_fun<const complex<double>&, double>(abs));
            fmax[i] = *max_element(fabs[i].begin(), fabs[i].end());
            fn0[i] = fabs[i][1];
        }
//        cout << endl << ::math(x0) << endl << endl;

        pointRes.fmin = *min_element(fn0.begin(), fn0.end());
        pointRes.fn0 = fn0;
        pointRes.fmax = fmax;
        pointRes.f0 = x0;
        pointRes.E0 = E0;
        
//        pointRes.Eth = numeric_limits<double>::infinity();

//        double count = 0;
//        for (int j = 0; j < 1; j++) {
//            count = j;

            //        for (int thi = 0; thi < 10; thi++) {

            //                    generate(xth.begin(), xth.end(), randx);
            //                    generate(x2th.begin(), x2th.end(), randx);

//            if (j == 1) {
//                copy(x0.begin(), x0.end(), xth.begin());
//                copy(x0.begin(), x0.end(), x2th.begin());
//            }

//            prob->setTheta(theta);
        prob.setParameters(U0, dU, J, point.mu / scale, theta);

//    for (int i = 0; i < ndim; i++) {
//        xth[i] = xuni(xrng);
//    }
            double Eth;
            string resultth;
            try {
//                prob->start();
                prob.start();
                //            population popth(eprob, npop);
                //            algo.evolve(popth);
                //            Eth = popth.champion().f[0];
                //            xth = popth.champion().x;
                //                result gres = gopt.optimize(xth, Eth);
                result res = lopt.optimize(xth, Eth);
//                prob->stop();
                prob.stop();
                resultth = to_string(res);
                //            Eth = prob->solve(xth);
            }
            catch (std::exception& e) {
//                prob->stop();
                prob.stop();
                result res = lopt.last_optimize_result();
                resultth = to_string(res) + ": " + e.what();
                printf("nlopt failed for Eth at %f, %f\n", point.x, point.mu);
                cout << e.what() << endl;
                Eth = numeric_limits<double>::quiet_NaN();
            }
            pointRes.statusth = resultth;
            //        pointRes.statusth = prob->getStatus();
//            pointRes.runtimeth = prob->getRuntime();
            pointRes.runtimeth = prob.getRuntime();

            norms = norm(xth);
            for (int i = 0; i < L; i++) {
                for (int n = 0; n <= nmax; n++) {
                    xth[2 * (i * dim + n)] /= norms[i];
                    xth[2 * (i * dim + n) + 1] /= norms[i];
                }
            }
//        cout << endl << ::math(xth) << endl << endl;
//            vector<double> grad(2*L*dim);
//            prob->E(xth,grad);
//            double maxg = 0;
//            for (int i = 0; i < 2*L*dim; i++) {
//                maxg = max(maxg, abs(grad[i]));
//            }
//            cout << ::math(maxg) << endl;
//            prob->setTheta(0);
//            prob->E(x0,grad);
//            maxg = 0;
//            for (int i = 0; i < 2*L*dim; i++) {
//                maxg = max(maxg, abs(grad[i]));
//            }
//            cout << ::math(maxg) << endl;
//            cout << endl;

            pointRes.fth = xth;
            pointRes.Eth = Eth;
//            pointRes.Eth = min(pointRes.Eth, Eth);
//            pointRes.Eth = GroundStateProblem::energy2(x0, J, U0, dU, point.mu, 0);

            //            prob->setTheta(2 * theta);
            //
            //            double E2th;
            //            string result2th;
            //            try {
            //                prob->start();
            ////            population pop2th(eprob, npop);
            ////            algo.evolve(pop2th);
            ////            E2th = pop2th.champion().f[0];
            ////            x2th = pop2th.champion().x;
            ////                result gres = gopt.optimize(x2th, E2th);
            //                result res = lopt.optimize(x2th, E2th);
            //                prob->stop();
            //                result2th = to_string(res);
            //                //            E2th = prob->solve(x2th);
            //            } catch (std::exception& e) {
            //                prob->stop();
            //                result res = lopt.last_optimize_result();
            //                result2th = to_string(res) + ": " + e.what();
            //                printf("Ipopt failed for E2th at %f, %f\n", point.x, point.mu);
            //                cout << e.what() << endl;
            //                E2th = numeric_limits<double>::quiet_NaN();
            //            }
            //            pointRes.status2th = result2th;
            //            //        pointRes.status2th = prob->getStatus();
            //            pointRes.runtime2th = prob->getRuntime();
            //
            //            norms = norm(x2th);
            //            for (int i = 0; i < L; i++) {
            //                for (int n = 0; n <= nmax; n++) {
            //                    x2th[2 * (i * dim + n)] /= norms[i];
            //                    x2th[2 * (i * dim + n) + 1] /= norms[i];
            //                }
            //            }
            //
            //            pointRes.f2th = x2th;
            //            pointRes.E2th = E2th;
            //
            //            pointRes.fs = (E2th - 2 * Eth + E0) / (L * theta * theta);

            pointRes.fs = (pointRes.Eth - E0) / (L * theta * theta);

//            if (pointRes.fs > -1e-5) {
//                break;
//            }
//            else {
//                //                theta *= 0.4641588833612779;
//            }
//        }
//        pointRes.theta = count; //theta;

        {
            boost::mutex::scoped_lock lock(points_mutex);
            pres.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

    {
        boost::mutex::scoped_lock lock(problem_mutex);
//        delete prob;
    }

}

int BW(double x, double a, double d) {
    return fabs(x - a) < d ? 1 : 0;
}

int BWfs(double fs) {
    return BW(fs, 0, 1e-4);
}

int BWfmin(double fmin) {
    return BW(fmin, 1, 1e-6);
}

double mufunc1(double x) {
    return 0.04750757147094086 - 9.163521595283873e-14 * x + 5.1156708283229015e-24 * x * x - 5.913212341351232e-36 * x * x*x;
}

double mufunc2(double x) {
    return 0.5426347180060362 - 4.734951434161236e-13 * x + 9.809069459521505e-23 * x * x - 5.644594041343443e-34 * x * x*x;
}

double mufunc3(double x) {
    return 1.77820606358231 - 1.9795818263926455e-11 * x + 1.1897718315201328e-22 * x * x - 2.9480343588099163e-34 * x * x*x;
}

double mufunc015l(double x) {
    return 0.032913659749522636 - 2.9822328051812337e-13*x + 8.053722708617216e-24*x*x - 1.8763641134601787e-35*x*x*x;
}

double mufunc015u(double x) {
    return 0.9681686436831983 - 8.658141185587507e-13*x - 1.101464387746557e-23*x*x + 1.1101188794879753e-35*x*x*x;
}

void getPoints(double xmin, double xmax, int nx, double (*mufunc)(double), int nmu, double muwidth, queue<Point>& points) {
    deque<double> x(nx);
    double dx = (xmax - xmin) / (nx - 1);
    for (int ix = 0; ix < nx; ix++) {
        x[ix] = xmin + ix * dx;
    }

    for (int ix = 0; ix < nx; ix++) {
        double mu0 = mufunc(x[ix]);
        double mui = mu0 - muwidth;
        double muf = mu0 + muwidth;
        if (mui < 0 || muf > 1)
            continue;
        deque<double> mu(nmu);
        if (nmu == 1) {
            mu[0] = mui;
        }
        else {
            double dmu = (muf - mui) / (nmu - 1);
            for (int imu = 0; imu < nmu; imu++) {
                mu[imu] = mui + imu * dmu;
            }
        }
        for (int imu = 0; imu < nmu; imu++) {
            Point point;
            point.x = x[ix];
            point.mu = mu[imu];
            points.push(point);
        }
    }
}

int main(int argc, char** argv) {
    //    GroundStateProblem prob;
    //
    ////    cout << prob.getE() << endl;
    ////    cout << prob.subst() << endl;
    //        vector<double> dU(L, 0);
    //        vector<double> J(L, 0.01);
    //        prob.setParameters(1, dU, J, 0.5);
    //        prob.setTheta(0);
    //        vector<double> f_(2*L*dim, 1);
    //        vector<double> grad(2*L*dim);
    //        cout << ::math(prob.E(f_, grad)) << endl;
    //        cout << ::math(grad) << endl;
    //    cout << ::math(prob.call(f_)) << endl;
    //        return 0;
    ////    vector<double> f;
    //    vector<double> f;
    //    double E = prob.solve(f);
    ////    prob.solve();
    //    cout << E << endl;
    //    cout << str(f) << endl;
    //    cout << prob.getEtheta() << endl;

    //    Ipopt::IpoptApplication app;
    //    app.PrintCopyrightMessage();

    cout << setprecision(20);

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    int nseed = lexical_cast<int>(argv[2]);

    double xmin = lexical_cast<double>(argv[3]);
    double xmax = lexical_cast<double>(argv[4]);
    int nx = lexical_cast<int>(argv[5]);

    deque<double> x(nx);
    if (nx == 1) {
        x[0] = xmin;
    }
    else {
        double dx = (xmax - xmin) / (nx - 1);
        for (int ix = 0; ix < nx; ix++) {
            x[ix] = xmin + ix * dx;
        }
    }

    int nlsampx = lexical_cast<int>(argv[6]);
    deque<double> lsampx(nlsampx);
    double dlsampx = (xmax - xmin) / (nlsampx - 1);
    for (int isampx = 0; isampx < nlsampx; isampx++) {
        lsampx[isampx] = xmin + isampx * dlsampx;
    }
    int nusampx = lexical_cast<int>(argv[7]);
    deque<double> usampx(nusampx);
    double xumin = 9e10;
    double xumax = xmax; //2.2e11;
    double dusampx = (xumax - xumin) / (nusampx - 1);
    for (int isampx = 0; isampx < nusampx; isampx++) {
        usampx[isampx] = xumin + isampx * dusampx;
    }
    //    int nntx = 6;
    //    double ntxmin = 1.8e11;
    //    double ntxmax = 2.2e11;
    //    for (int ix = 0; ix < nntx; ix++) {
    //        double ntx = ntxmin + ix * (ntxmax - ntxmin) / (nntx - 1);
    //        usampx.push_back(ntx);
    //    }
    //    nusampx = usampx.size();

    double mumin = lexical_cast<double>(argv[8]);
    double mumax = lexical_cast<double>(argv[9]);
    int nmu = lexical_cast<int>(argv[10]);

    deque<double> mu(nmu);
    if (nmu == 1) {
        mu[0] = mumin;
    }
    else {
        double dmu = (mumax - mumin) / (nmu - 1);
        for (int imu = 0; imu < nmu; imu++) {
            mu[imu] = mumin + imu * dmu;
        }
    }

    int nlsampmu = lexical_cast<int>(argv[11]);
    int nusampmu = lexical_cast<int>(argv[12]);

    int nxtip = lexical_cast<int>(argv[13]);
    int nmutip = lexical_cast<int>(argv[14]);

    double D = lexical_cast<double>(argv[15]);
    double theta = lexical_cast<double>(argv[16]);

    int numthreads = lexical_cast<int>(argv[17]);

    int resi = lexical_cast<int>(argv[18]);

    //    bool sample = lexical_cast<bool>(argv[18]);

    tls.reset(new thread_id(0));
    thread_alloc::parallel_setup(numthreads+1, in_parallel, thread_num);
    thread_alloc::hold_memory(true);
    parallel_ad<double>();
    CheckSimpleVector<size_t, CppAD::vector<size_t>>();
    CheckSimpleVector<set<size_t>, CppAD::vector<set<size_t>>>(CppAD::one_element_std_set<size_t>(), CppAD::two_element_std_set<size_t>());
    parallel = true;

#ifdef AMAZON
    //    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Gutzwiller");
    //    path resdir("/media/ubuntu/Results/CTG");
#else
    //    path resdir("/Users/Abuenameh/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Gutzwiller");
    //    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Gutzwiller Amazon");
    //    path resdir("/User/Abuenameh/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Gutzwiller");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    for (int iseed = 0; iseed < nseed; iseed++, seed++) {
        ptime begin = microsec_clock::local_time();


        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }

        Parameter xi;
        xi.fill(1);

        rng.seed(seed);

        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        boost::filesystem::ofstream os(resfile);
        printMath(os, "Lres", resi, L);
        printMath(os, "nmaxres", resi, nmax);
        printMath(os, "seed", resi, seed);
        printMath(os, "theta", resi, theta);
        printMath(os, "Delta", resi, D);
        printMath(os, "xres", resi, x);
        printMath(os, "mures", resi, mu);
        printMath(os, "xires", resi, xi);
        os << flush;

        cout << "Res: " << resi << endl;

//        GroundStateProblem::setup();

        queue<Point> points;
        queue<Point> points2;
        //            if (false)
        {
            double muwidth = 0.05;
            //            queue<Point> points;
            
                /*queue<Point> lpoints;
            double mulsampwidth = 0.02;
            for (int ix = 0; ix < nlsampx; ix++) {
                //                double mu0 = 0.03615582350346575 - 5.005273114442404e-14*x[ix] + 6.275817853250553e-24*x[ix]*x[ix] - 1.4195907309128102e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.25
                //                double mu0 = 0.025470163481530313 - 2.2719398923789667e-13*x[ix] + 8.92045173286913e-24*x[ix]*x[ix] - 2.4033506846113224e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.1
                //                double mu0 = 0.028572248841708368 - 4.1318226651330257e-13*x[ix] + 1.1199528880961205e-23*x[ix]*x[ix] - 3.0330199477565917e-35*x[ix]*x[ix]*x[ix]; // Delta = 0
//                double mu0 = 0.030969306517268605 + 1.9188880181335529e-13 * lsampx[ix] + 2.5616067018411045e-24 * lsampx[ix] * lsampx[ix] + 1.0173988468289905e-36 * lsampx[ix] * lsampx[ix] * lsampx[ix]; // Delta = 0.25 Lower
                double mu0 = mufunc015l(lsampx[ix]);
                double mui = max(mumin, mu0 - mulsampwidth);
                double muf = min(mumax, mu0 + mulsampwidth);
                deque<double> mu(nlsampmu);
                if (nlsampmu == 1) {
                    mu[0] = mui;
                }
                else {
                    double dmu = (muf - mui) / (nlsampmu - 1);
                    for (int imu = 0; imu < nlsampmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                }
                for (int imu = 0; imu < nlsampmu; imu++) {
                    Point point;
                    point.x = lsampx[ix];
                    point.mu = mu[imu];
                    lpoints.push(point);
//                    points.push(point);
                }
            }

            progress_display lprogress(lpoints.size());

            vector<PointResults> lpointRes;

            thread_group lthreads;
            for (int i = 0; i < numthreads; i++) {
                lthreads.create_thread(bind(&phasepoints, boost::ref(xi), theta, boost::ref(lpoints), boost::ref(lpointRes), boost::ref(lprogress)));
            }
            lthreads.join_all();

            vector<Sample> lWmuBWfsfmin;

            for (PointResults pres : lpointRes) {
                lWmuBWfsfmin.push_back(make_tuple(pres.W, pres.mu, BWfs(pres.fs), BWfmin(pres.fmin)));
            }
            sort(lWmuBWfsfmin.begin(), lWmuBWfsfmin.end(), [](const Sample& a, const Sample & b) {
                return get<0>(a) < get<0>(b);
            });
            stable_sort(lWmuBWfsfmin.begin(), lWmuBWfsfmin.end(), [](const Sample& a, const Sample & b) {
                return get<1>(a) < get<1>(b);
            });
            vector<Sample> lsampbound;
            for (int ix = 0; ix < nlsampx; ix++) {
                auto boundary = find_if(lWmuBWfsfmin.begin(), lWmuBWfsfmin.end(), [&](const Sample & a) {
                    return get<0>(a) == lsampx[ix] && get<2>(a) == 1;
                });
                if (boundary != lWmuBWfsfmin.end()) {
                    lsampbound.push_back(*boundary);
                }
            }
            for (int bix = 0; bix < lsampbound.size() - 1; bix++) {
                double x1 = get<0>(lsampbound[bix]);
                double x2 = get<0>(lsampbound[bix + 1]);
                double mu1 = get<1>(lsampbound[bix]);
                double mu2 = get<1>(lsampbound[bix + 1]);
                double dx = (x2 - x1) / (nx - 1);
                for (int ix = 0; ix < nx; ix++) {
                    if (ix < nx - 1 || (bix == lsampbound.size() - 2)) {
                        double mu0 = ix * dx * (mu2 - mu1) / (x2 - x1) + mu1;
                        double mui = max(mumin, mu0 - muwidth);
                        double muf = min(mumax, mu0 + muwidth);
                        deque<double> mu(nmu);
                        if (nmu == 1) {
                            mu[0] = mui;
                        }
                        else {
                            double dmu = (muf - mui) / (nmu - 1);
                            for (int imu = 0; imu < nmu; imu++) {
                                mu[imu] = mui + imu * dmu;
                            }
                        }
                        for (int imu = 0; imu < nmu; imu++) {
                            Point point;
                            point.x = x1 + ix * dx;
                            point.mu = mu[imu];
//                            points.push(point);
                        }
                    }
                }
            }*/

            /*int nldx = 5;
            for (int ix = 0; ix < nldx*(nlsampx - 1); ix++) {
                double sx = xmin + dlsampx * ix / nldx;
                if (sx > get<0>(lsampbound.back()))
                    continue;
                //                double mu0 = 0.03615582350346575 - 5.005273114442404e-14*x[ix] + 6.275817853250553e-24*x[ix]*x[ix] - 1.4195907309128102e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.25
                //                double mu0 = 0.025470163481530313 - 2.2719398923789667e-13*x[ix] + 8.92045173286913e-24*x[ix]*x[ix] - 2.4033506846113224e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.1
                //                double mu0 = 0.028572248841708368 - 4.1318226651330257e-13*x[ix] + 1.1199528880961205e-23*x[ix]*x[ix] - 3.0330199477565917e-35*x[ix]*x[ix]*x[ix]; // Delta = 0
                double mu0 = 0.030969306517268605 + 1.9188880181335529e-13 * sx + 2.5616067018411045e-24 * sx * sx + 1.0173988468289905e-36 * sx * sx * sx; // Delta = 0.25 Lower
                double mui = max(mumin, mu0 + mulsampwidth);
//                double muf = 0.5;
                double muf = max(mumin, mu0 + 2*mulsampwidth);
                int nmu = 5;
                deque<double> mu(nmu);
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = xmin + dlsampx * ix / nldx;
                    point.mu = mu[imu];
                    points.push(point);
                }
            }
            for (int ix = 0; ix < nldx*(nlsampx - 1); ix++) {
                double sx = xmin + dlsampx * ix / nldx;
                if (sx > get<0>(lsampbound.back()))
                    continue;
                //                double mu0 = 0.03615582350346575 - 5.005273114442404e-14*x[ix] + 6.275817853250553e-24*x[ix]*x[ix] - 1.4195907309128102e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.25
                //                double mu0 = 0.025470163481530313 - 2.2719398923789667e-13*x[ix] + 8.92045173286913e-24*x[ix]*x[ix] - 2.4033506846113224e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.1
                //                double mu0 = 0.028572248841708368 - 4.1318226651330257e-13*x[ix] + 1.1199528880961205e-23*x[ix]*x[ix] - 3.0330199477565917e-35*x[ix]*x[ix]*x[ix]; // Delta = 0
                double mu0 = 0.030969306517268605 + 1.9188880181335529e-13 * sx + 2.5616067018411045e-24 * sx * sx + 1.0173988468289905e-36 * sx * sx * sx; // Delta = 0.25 Lower
                double mui = max(mumin, mu0 - 2*mulsampwidth);
//                double muf = 0.5;
                double muf = max(mumin, mu0 - mulsampwidth);
                int nmu = 5;
                deque<double> mu(nmu);
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = xmin + dlsampx * ix / nldx;
                    point.mu = mu[imu];
                    points.push(point);
                }
            }*/

            /*queue<Point> upoints;
            double muusampwidth = 0.05;
            for (int ix = 0; ix < nusampx; ix++) {
//                                    double mu0 = 1.0275844755940469 - 1.3286603408812447e-12*usampx[ix] - 1.9177090288512203e-23*usampx[ix]*usampx[ix] + 9.572518996956652e-35*usampx[ix]*usampx[ix]*usampx[ix] - 2.095759744296641e-46*usampx[ix]*usampx[ix]*usampx[ix]*usampx[ix]; // Delta 0.25
                double mu0 = mufunc015u(usampx[ix]);
                double mui = max(mumin, mu0 - muusampwidth);
                double muf = min(mumax, mu0 + 2*muusampwidth);
                deque<double> mu(nusampmu);
                if (nusampmu == 1) {
                    mu[0] = mui;
                }
                else {
                    double dmu = (muf - mui) / (nusampmu - 1);
                    for (int imu = 0; imu < nusampmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                }
                for (int imu = 0; imu < nusampmu; imu++) {
                    Point point;
                    point.x = usampx[ix];
                    point.mu = mu[imu];
                    upoints.push(point);
//                    points.push(point);
//                    points2.push(point);
                }
            }

            progress_display uprogress(upoints.size());

            vector<PointResults> upointRes;

            thread_group uthreads;
            for (int i = 0; i < numthreads; i++) {
                uthreads.create_thread(bind(&phasepoints, boost::ref(xi), theta, boost::ref(upoints), boost::ref(upointRes), boost::ref(uprogress)));
            }
            uthreads.join_all();

            vector<Sample> uWmuBWfsfmin;

            for (PointResults pres : upointRes) {
                uWmuBWfsfmin.push_back(make_tuple(pres.W, pres.mu, BWfs(pres.fs), BWfmin(pres.fmin)));
            }
            sort(uWmuBWfsfmin.begin(), uWmuBWfsfmin.end(), [](const Sample& a, const Sample & b) {
                return get<0>(a) < get<0>(b);
            });
            stable_sort(uWmuBWfsfmin.begin(), uWmuBWfsfmin.end(), [](const Sample& a, const Sample & b) {
                return get<1>(a) < get<1>(b);
            });
            vector<Sample> usampbound1;
            for (int ix = 0; ix < nusampx; ix++) {
                auto inner = find_if(uWmuBWfsfmin.begin(), uWmuBWfsfmin.end(), [&](const Sample& a) {
                    return get<0>(a) == usampx[ix] && get<3>(a) == 1;
                });
                auto boundary = find_if(inner, uWmuBWfsfmin.end(), [&](const Sample & a) {
                    return get<0>(a) == usampx[ix] && get<3>(a) == 0;
                });
                if (boundary != uWmuBWfsfmin.end()) {
                    usampbound1.push_back(*boundary);
                }
            }
            for (int bix = 0; bix < usampbound1.size() - 1; bix++) {
                double x1 = get<0>(usampbound1[bix]);
                double x2 = get<0>(usampbound1[bix + 1]);
                double mu1 = get<1>(usampbound1[bix]);
                double mu2 = get<1>(usampbound1[bix + 1]);
                double dx = (x2 - x1) / (nx - 1);
                for (int ix = 0; ix < nx; ix++) {
                    if (ix < nx - 1 || (bix == usampbound1.size() - 2)) {
                        double mu0 = ix * dx * (mu2 - mu1) / (x2 - x1) + mu1;
                        double mui = max(mumin, mu0 - muwidth);
                        double muf = min(mumax, mu0 + muwidth);
                        deque<double> mu(nmu);
                        if (nmu == 1) {
                            mu[0] = mui;
                        }
                        else {
                            double dmu = (muf - mui) / (nmu - 1);
                            for (int imu = 0; imu < nmu; imu++) {
                                mu[imu] = mui + imu * dmu;
                            }
                        }
                        for (int imu = 0; imu < nmu; imu++) {
                            Point point;
                            point.x = x1 + ix * dx;
                            point.mu = mu[imu];
                            points.push(point);
//                            points2.push(point);
                        }
                    }
                }
            }
            vector<Sample> usampbound2;
            for (int ix = 0; ix < nusampx; ix++) {
                auto boundary = find_if(uWmuBWfsfmin.rbegin(), uWmuBWfsfmin.rend(), [&](const Sample & a) {
                    return get<0>(a) == usampx[ix] && get<2>(a) != 0;
                });
                if (boundary != uWmuBWfsfmin.rend()) {
                    usampbound2.push_back(*boundary);
                }
            }
            for (int bix = 0; bix < usampbound2.size() - 1; bix++) {
                double x1 = get<0>(usampbound2[bix]);
                double x2 = get<0>(usampbound2[bix + 1]);
                double mu1 = get<1>(usampbound2[bix]);
                double mu2 = get<1>(usampbound2[bix + 1]);
                double dx = (x2 - x1) / (nx - 1);
                for (int ix = 0; ix < nx; ix++) {
                    if (ix < nx - 1 || (bix == usampbound2.size() - 2)) {
                        double mu0 = ix * dx * (mu2 - mu1) / (x2 - x1) + mu1;
                        double mui = max(mumin, mu0 - muwidth);
                        double muf = min(mumax, mu0 + muwidth);
                        deque<double> mu(nmu);
                        if (nmu == 1) {
                            mu[0] = mui;
                        }
                        else {
                            double dmu = (muf - mui) / (nmu - 1);
                            for (int imu = 0; imu < nmu; imu++) {
                                mu[imu] = mui + imu * dmu;
                            }
                        }
                        for (int imu = 0; imu < nmu; imu++) {
                            Point point;
                            point.x = x1 + ix * dx;
                            point.mu = mu[imu];
                            points.push(point);
//                            points2.push(point);
                        }
                    }
                }
            }*/

            /*int nudx = 5;
            for (int ix = 0; ix < nudx*(nusampx-1); ix++) {
                double sx = xumin + dusampx * ix / nudx;
                if (sx > get<0>(usampbound1.back()))
                    continue;
                                    double mu0 = 1.0275844755940469 - 1.3286603408812447e-12*sx - 1.9177090288512203e-23*sx*sx + 9.572518996956652e-35*sx*sx*sx - 2.095759744296641e-46*sx*sx*sx*sx; // Delta 0.25
//                double mui = 0.5;
                double mui = min(mumax, mu0 - 2*muusampwidth);
                double muf = min(mumax, mu0 - muusampwidth);
                int nmu = 5;
                deque<double> mu(nmu);
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = xumin + dusampx * ix / nudx;
                    point.mu = mu[imu];
//                    points.push(point);
                }
            }
            for (int ix = 0; ix < nudx*(nusampx-1); ix++) {
                double sx = xumin + dusampx * ix / nudx;
                if (sx > get<0>(usampbound1.back()))
                    continue;
                                    double mu0 = 1.0275844755940469 - 1.3286603408812447e-12*sx - 1.9177090288512203e-23*sx*sx + 9.572518996956652e-35*sx*sx*sx - 2.095759744296641e-46*sx*sx*sx*sx; // Delta 0.25
//                double mui = 0.5;
                double mui = min(mumax, mu0 + muusampwidth);
                double muf = min(mumax, mu0 + 2*muusampwidth);
                int nmu = 5;
                deque<double> mu(nmu);
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = xumin + dusampx * ix / nudx;
                    point.mu = mu[imu];
//                    points.push(point);
                }
            }*/

        }

        double corxmin = 2e10;
        double corxmax = 9e10;
        int ncorx = 61;
        double dcorx = (corxmax - corxmin) / (ncorx - 1);
        double cormumin = 0.5;
        double cormumax = 1;
        int ncormu = 51;
        double dcormu = (cormumax - cormumin) / (ncormu - 1);
        for (int ix = 0; ix < ncorx; ix++) {
            double corx = corxmin + ix * dcorx;
            for (int imu = 0; imu < ncormu; imu++) {
                double cormu = cormumin + imu * dcormu;
                Point point;
                point.x = corx;
                point.mu = cormu;
                //                    points.push(point);
            }
        }

        double lxmin = 2e10;
        double lxmax = 2.5e10;
        int nlx = 4;
        double dlx = (lxmax - lxmin) / (nlx - 1);
        double lmumin = 0;
        double lmumax = 0.5;
        int nlmu = 10;
        double dlmu = (lmumax - lmumin) / (nlmu - 1);
        for (int ix = 0; ix < nlx; ix++) {
            double lx = lxmin + ix * dlx;
            for (int imu = 0; imu < nlmu; imu++) {
                double lmu = lmumin + imu * dlmu;
                Point point;
                point.x = lx;
                point.mu = lmu;
                //                    points.push(point);
            }
        }

        double xtip = 2.4e11;
        double xtipwidth = 2.8e10;
        //            double xtipwidth = 3e10;
        //            double xtip = 2.57e11;
        //            double xtipwidth = 1e10;
        double xtipmin = xtip - xtipwidth;
        double xtipmax = xtip + xtipwidth;
        ;
        double dxtip = (xtipmax - xtipmin) / (nxtip - 1);
        double mutip = 0.27;
        double mutipwidth = 0.15;
        double mutipmin = mutip - mutipwidth;
        double mutipmax = mutip + mutipwidth;
        //            double mutipmin = mutip - mutipwidth;
        //            double mutipmax = mutip + 2*mutipwidth;
        double dmutip = (mutipmax - mutipmin) / (nmutip - 1);
        for (int ix = 0; ix < nxtip; ix++) {
            double tx = xtipmin + ix * dxtip;
            for (int imu = 0; imu < nmutip; imu++) {
                double tmu = mutipmin + imu * dmutip;
                Point point;
                point.x = tx;
                point.mu = tmu;
                //                    points.push(point);
                //                    points2.push(point);
            }
        }

        double muuwidth = 0.2;
        for (int ix = 0; ix < nx; ix++) {
            double mu0 = 1.0275844755940469 - 1.3286603408812447e-12 * x[ix] - 1.9177090288512203e-23 * x[ix] * x[ix] + 9.572518996956652e-35 * x[ix] * x[ix] * x[ix] - 2.095759744296641e-46 * x[ix] * x[ix] * x[ix] * x[ix]; // Delta 0.25
            double mui = max(mumin, mu0 - muuwidth);
            double muf = mu0 + muuwidth; //min(mumax, mu0 + muuwidth);
            deque<double> mu(nmu);
            if (nmu == 1) {
                mu[0] = mui;
            }
            else {
                double dmu = (muf - mui) / (nmu - 1);
                for (int imu = 0; imu < nmu; imu++) {
                    mu[imu] = mui + imu * dmu;
                }
            }
            for (int imu = 0; imu < nmu; imu++) {
                if (mu[imu] > mumax)
                    continue;
                Point point;
                point.x = x[ix];
                point.mu = mu[imu];
                //                    points.push(point);
                //                    points2.push(point);
            }
        }

        //        getPoints(2.05e10, 2.12e11, 140, mufunc1, nmu, 0.01, points);
        //        getPoints(2.22499e10, 7.43281e10, 40, mufunc2, nmu, 0.01, points);
        //        getPoints(7.7128e10, 1.97524e11, 100, mufunc3, nmu, 0.01, points);

        //        vector<pair<double, double>> ps({{50000000000, 0.903265}, {159500000000, 0.666948}, {234500000000, 
        //  0.166431}, {303500000000, 0.121481}, {309500000000, 0.0942961}});
        vector<pair<double, double>> ps({
            {2.60100166944908e10, 0.0366959735767002}});
        for (pair<double, double> p : ps) {
            Point point;
            point.x = p.first;
            point.mu = p.second;
//            points.push(point);
        }

        double muwidth = 0.02;
        for (int ix = 0; ix < nx; ix++) {
            double mu0 = 0.9411179399500129 - 3.5751626448519524e-13 * x[ix] - 7.407324226206937e-24 * x[ix] * x[ix] - 1.376619100837241e-35 * x[ix] * x[ix] * x[ix] +
                    4.1960731262022256e-47 * x[ix] * x[ix] * x[ix] * x[ix];
            double mui = max(mumin, mu0 - muwidth);
            double muf = min(mumax, mu0 + muwidth);
            deque<double> mu(nmu);
            if (nmu == 1) {
                mu[0] = mui;
            }
            else {
                double dmu = (muf - mui) / (nmu - 1);
                for (int imu = 0; imu < nmu; imu++) {
                    mu[imu] = mui + imu * dmu;
                }
            }
            for (int imu = 0; imu < nmu; imu++) {
                Point point;
                point.x = x[ix];
                point.mu = mu[imu];
                //                                        points.push(point);
            }
        }
        for (int ix = 0; ix < nx; ix++) {
            double mu0 = 0.03557224884170837 - 4.1318226651330257e-13 * x[ix] + 1.1199528880961205e-23 * x[ix] * x[ix] - 3.0330199477565917e-35 * x[ix] * x[ix] * x[ix];
            double mui = max(mumin, mu0 - muwidth);
            double muf = min(mumax, mu0 + muwidth);
            deque<double> mu(nmu);
            if (nmu == 1) {
                mu[0] = mui;
            }
            else {
                double dmu = (muf - mui) / (nmu - 1);
                for (int imu = 0; imu < nmu; imu++) {
                    mu[imu] = mui + imu * dmu;
                }
            }
            for (int imu = 0; imu < nmu; imu++) {
                Point point;
                point.x = x[ix];
                point.mu = mu[imu];
                //                            points.push(point);
            }
        }
        
        int nmu2 = 20;
        int nx2 = 20;
        for (int ix = 0; ix < nx2; ix++) {
            double x = 2e10 + ix*(3e11 - 2e10)/(nx2-1);
            for (int imu = 0; imu < nmu2; imu++) {
                double mu = imu/(nmu2 - 1.);
                Point point;
                point.x = x;
                point.mu = mu;
//                points.push(point);
            }
        }
        
        double muwidth2 = 0.1;
//        int nmu2 = 6;
//        int nx2 = 100;
        for (int ix = 0; ix < nx2; ix++) {
            double x = 2e10 + ix*(2.6e11 - 2e10)/(nx2-1);
//                double mu0 = -0.018989311717356086 + 6.87667461054985e-13*x + 7.7264998850342525e-25*x*x - 2.069564731044878e-36*x*x*x;
                double mu0 = 0.032913659749522636 - 2.9822328051812337e-13*x + 8.053722708617216e-24*x*x - 1.8763641134601787e-35*x*x*x;
                double mui = mu0 - muwidth2;
                double muf = mu0 + muwidth2;
            for (int imu = 0; imu < nmu2; imu++) {
                double mu = mui + imu*(muf-mui)/(nmu2 - 1);
                Point point;
                point.x = x;
                point.mu = mu;
//                points.push(point);
            }
        }
        for (int ix = 0; ix < nx2; ix++) {
            double x = 2e10 + ix*(2.6e11 - 2e10)/(nx2-1);
//                double mu0 = 0.9464941207678484 - 2.5363733791190035e-13*x - 1.961773720477146e-23*x*x + 3.7097027455669513e-35*x*x*x;
                double mu0 = 0.9681686436831983 - 8.658141185587507e-13*x - 1.101464387746557e-23*x*x + 1.1101188794879753e-35*x*x*x;
                double mui = mu0 - muwidth2;
                double muf = mu0 + muwidth2;
            for (int imu = 0; imu < nmu2; imu++) {
                double mu = mui + imu*(muf-mui)/(nmu2 - 1);
                Point point;
                point.x = x;
                point.mu = mu;
//                points.push(point);
            }
        }
        
        
        for (int ix = 0; ix < 50; ix++) {
            double x = 2e10 + ix*(2.6e11 - 2e10)/(50-1);
            for (int imu = 0; imu < 50; imu++) {
                double mu = imu/(50.-1);
//                points.push({x, mu});
            }
        }
        
//        points.push({2.27878787879e11, 0.2667});
        int nW = 40;
        for (int i = 0; i < nW; i++) {
            double Wi = 2.2424242424e11;
            double Wf = 2.4e11;
            double W = Wi + i*(Wf - Wi)/(nW-1);
            points.push({W, 0.2667});
        }
//        points.push({2e10,0.9});

        /*{
                  double x1min = 2.05e10;
                  double x1max = 2.12e11;
                  double nx1 = 10;
                  deque<double> x1(nx1);
                  double dx1 = (x1max - x1min) / (nx1 - 1);
                  for (int ix = 0; ix < nx1; ix++) {
                      x1[ix] = x1min + ix * dx1;
                  }

                  double mu1width = 0.01;
                  for (int ix = 0; ix < nx1; ix++) {
                      double mu0 = mufunc1(x1[ix]);
                      double mui = max(mumin, mu0 - mu1width);
                      double muf = min(mumax, mu0 + mu1width);
                      deque<double> mu(nmu);
                      if (nmu == 1) {
                          mu[0] = mui;
                      }
                      else {
                          double dmu = (muf - mui) / (nmu - 1);
                          for (int imu = 0; imu < nmu; imu++) {
                              mu[imu] = mui + imu * dmu;
                          }
                      }
                      for (int imu = 0; imu < nmu; imu++) {
                          Point point;
                          point.x = x1[ix];
                          point.mu = mu[imu];
                          points.push(point);
                      }
                  }
              }*/

        //        double muwidth = 0.05;
        //        double muwidth = 0.01;
        /*
                    queue<Point> points;
                //        bool sample = true;
                if (sample) {
                    //            for (int ix = 0; ix < nx; ix++) {
                    //                //            double mu0 = x[ix] / 1e12 + 0.05;
                    ////                double mu0 = 7.142857142857143e-13 * x[ix] + 0.08571428571428572;
                    ////                double mu0 = -6.333293551338674e-24 * x[ix] * x[ix] - 8.967458328360531e-13 * x[ix] + 0.9514478259139914; // Delta = 0
                    ////                double mu0 = -1.0374437419130666e-23 * x[ix] * x[ix] - 5.901199487215756e-13 * x[ix] + 0.8982308684507191; // Delta = 0.25
                    ////                double mu0 = -1.0374437419130666e-23 * x[ix] * x[ix] - 4.901199487215756e-13 * x[ix] + 0.8982308684507191; // Delta = 0.25
                    //                double mu0 = 1.0275844755940469 - 1.3286603408812447e-12*x[ix] - 1.9177090288512203e-23*x[ix]*x[ix] + 9.572518996956652e-35*x[ix]*x[ix]*x[ix] - 2.095759744296641e-46*x[ix]*x[ix]*x[ix]*x[ix]; // Delta 0.25
                    ////                double mu0 = 0.9617950685857694 - 7.84998396963284e-13*x[ix] - 9.165384267382779e-24*x[ix]*x[ix] + 3.646236061739209e-36*x[ix]*x[ix]*x[ix] + 4.290137652003345e-48*x[ix]*x[ix]*x[ix]*x[ix]; // Delta = 0.1
                    ////                double mu0 = 0.9311179399500129 - 3.5751626448519524e-13*x[ix] - 7.407324226206937e-24*x[ix]*x[ix] - 1.376619100837241e-35*x[ix]*x[ix]*x[ix] + 4.1960731262022256e-47*x[ix]*x[ix]*x[ix]*x[ix]; // Delta = 0
                    //                double mui = max(mumin, mu0 - muwidth);
                    //                double muf = min(mumax, mu0 + muwidth);
                    //                deque<double> mu(nmu);
                    //                if (nmu == 1) {
                    //                    mu[0] = mui;
                    //                } else {
                    //                    double dmu = (muf - mui) / (nmu - 1);
                    //                    for (int imu = 0; imu < nmu; imu++) {
                    //                        mu[imu] = mui + imu * dmu;
                    //                    }
                    //                }
                    //                for (int imu = 0; imu < nmu; imu++) {
                    //                    Point point;
                    //                    point.x = x[ix];
                    //                    point.mu = mu[imu];
                    //                    points.push(point);
                    //                }
                    //            }
                    for (int ix = 0; ix < nsampx; ix++) {
                        //            double mu0 = -3*x[ix] / 1e12 + 0.96;
                        //                double mu0 = -2.142857142857143e-12 * x[ix] + 0.942857142857143;
                        //                double mu0 = -3.301221096348316e-35 * x[ix] * x[ix] * x[ix] + 1.3058538719558353e-23 * x[ix] * x[ix] - 7.882264201707455e-13 * x[ix] + 0.0413527624303548; // Delta = 0
                        //                double mu0 = 8.938048153734245e-36 * x[ix] * x[ix] * x[ix] - 2.202590437883966e-25 * x[ix] * x[ix] + 4.3412578706695816e-13 * x[ix] + 0.023602991053971553; // Delta = 0.25
                        //                double mu0 = 0.03615582350346575 - 5.005273114442404e-14*x[ix] + 6.275817853250553e-24*x[ix]*x[ix] - 1.4195907309128102e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.25
                        //                double mu0 = 0.025470163481530313 - 2.2719398923789667e-13*x[ix] + 8.92045173286913e-24*x[ix]*x[ix] - 2.4033506846113224e-35*x[ix]*x[ix]*x[ix]; // Delta = 0.1
                        //                double mu0 = 0.028572248841708368 - 4.1318226651330257e-13*x[ix] + 1.1199528880961205e-23*x[ix]*x[ix] - 3.0330199477565917e-35*x[ix]*x[ix]*x[ix]; // Delta = 0
                        double mu0 = 0.030969306517268605 + 1.9188880181335529e-13 * x[ix] + 2.5616067018411045e-24 * x[ix] * x[ix] + 1.0173988468289905e-36 * x[ix] * x[ix] * x[ix]; // Delta = 0.25 Lower
                        double mui = max(mumin, mu0 - muwidth);
                        double muf = min(mumax, mu0 + muwidth);
                        deque<double> mu(nmu);
                        if (nmu == 1) {
                            mu[0] = mui;
                        }
                        else {
                            double dmu = (muf - mui) / (nmu - 1);
                            for (int imu = 0; imu < nmu; imu++) {
                                mu[imu] = mui + imu * dmu;
                            }
                        }
                        for (int imu = 0; imu < nmu; imu++) {
                            Point point;
                            point.x = x[ix];
                            point.mu = mu[imu];
                            points.push(point);
                        }
                    }
                }
                else {
                    for (int imu = 0; imu < nmu; imu++) {
                        for (int ix = 0; ix < nx; ix++) {
                            Point point;
                            point.x = x[ix];
                            point.mu = mu[imu];
                            points.push(point);
                        }
                    }
                }*/
        progress_display progress(points.size());

        vector<PointResults> pointRes;

//        GroundStateProblem::setup();
        thread_group threads;
        for (int i = 0; i < numthreads; i++) {
            //                        threads.emplace_back(phasepoints, std::ref(xi), theta, std::ref(points), std::ref(f0res), std::ref(E0res), std::ref(Ethres), std::ref(fsres), std::ref(progress));
            threads.create_thread(bind(&phasepoints, i+1, boost::ref(xi), theta, boost::ref(points), boost::ref(pointRes), boost::ref(progress)));
        }
        threads.join_all();

        vector<pair<double, double> > Wmu;
        vector<double> Jxs;
        vector<double> Uxs;
        vector<vector<double> > Js;
        vector<vector<double> > Us;
        vector<double> fs;
        vector<double> fmin;
        vector<vector<double> > fn0;
        vector<vector<double> > fmax;
        vector<vector<double> > f0;
        vector<vector<double> > fth;
        vector<vector<double> > f2th;
        vector<double> E0;
        vector<double> Eth;
        vector<double> E2th;
        vector<string> status0;
        vector<string> statusth;
        vector<string> status2th;
        vector<string> runtime0;
        vector<string> runtimeth;
        vector<string> runtime2th;
        vector<double> thetas;

        for (PointResults pres : pointRes) {
            Wmu.push_back(make_pair(pres.W, pres.mu));
            Jxs.push_back(pres.Jx);
            Uxs.push_back(pres.Ux);
            Js.push_back(pres.J);
            Us.push_back(pres.U);
            fs.push_back(pres.fs);
            fmin.push_back(pres.fmin);
            fn0.push_back(pres.fn0);
            fmax.push_back(pres.fmax);
            f0.push_back(pres.f0);
            fth.push_back(pres.fth);
            f2th.push_back(pres.f2th);
            E0.push_back(pres.E0);
            Eth.push_back(pres.Eth);
            //            E2th.push_back(pres.E2th);
            status0.push_back(pres.status0);
            statusth.push_back(pres.statusth);
            //            status2th.push_back(pres.status2th);
            runtime0.push_back(pres.runtime0);
            runtimeth.push_back(pres.runtimeth);
            //            runtime2th.push_back(pres.runtime2th);
            thetas.push_back(pres.theta);
        }

        printMath(os, "Wmu", resi, Wmu);
        printMath(os, "Jxs", resi, Jxs);
        printMath(os, "Uxs", resi, Uxs);
        printMath(os, "Js", resi, Js);
        printMath(os, "Us", resi, Us);
        printMath(os, "fs", resi, fs);
        //        printMath(os, "fn0", resi, fn0);
        printMath(os, "fmin", resi, fmin);
        //        printMath(os, "fmax", resi, fmax);
                printMath(os, "f0", resi, f0);
                printMath(os, "fth", resi, fth);
        //        printMath(os, "f2th", resi, f2th);
        printMath(os, "E0", resi, E0);
        printMath(os, "Eth", resi, Eth);
        //        printMath(os, "E2th", resi, E2th);
                printMath(os, "status0", resi, status0);
                printMath(os, "statusth", resi, statusth);
        //        printMath(os, "status2th", resi, status2th);
        //        printMath(os, "runtime0", resi, runtime0);
        //        printMath(os, "runtimeth", resi, runtimeth);
        //        printMath(os, "runtime2th", resi, runtime2th);
        //        printMath(os, "thetas", resi, thetas);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;
    }

    return 0;
}
