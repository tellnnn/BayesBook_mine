#include <stan/math.hpp>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace stan;
using namespace math;
using namespace Eigen;

void generate_data(int N, int K, VectorXd lambda, VectorXd pi, int seed) {
    // function to generate random data
    // inputs:
    //   N: the number of data points
    //   K: the number of clusters
    //   lambda: the rate parameter in poisson distribution
    //   pi: the mixing parameter
    //   seed: the random seed value

    // set random engine with the random seed value
    std::default_random_engine engine(seed);

    // set variables
    int s; // the latent variable
    int X; // the data
    
    // set the output file
    std::ofstream data("data.csv");
    // set the header in the file
    data << "s,X" << std::endl;
    for (int n = 0; n < N; n++) {
        // sample s and X
        s = categorical_rng(pi,engine);
        X = poisson_rng(lambda(s-1),engine);
        // output s and X
        data << s << "," << X << std::endl;
    }
}

std::vector<std::string> split(std::string& input, char delimiter) {
    // function to get a line and split it
    // based on https://cvtech.cc/readcsv/
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}

double calc_ELBO(int N, int K, VectorXi X,
                 VectorXd a_pri, VectorXd b_pri, VectorXd alpha_pri,
                 VectorXd a_pos, VectorXd b_pos, VectorXd alpha_pos) {
    // function to calculate ELBO
    // inputs:
    //   N: the number of data points
    //   K: the number of clusters
    //   X: the data
    //   a_pri: the shape parameter before the update
    //   b_pri: the rate parameter before the update
    //   alpha_pri: the concentration parameter before the update
    //   a_pos: the shape parameter after the update
    //   b_pos: the rate parameter after the update
    //   alpha_pos: the concentration parameter after the update

    // calc E[lambda], E[ln lambda], and E[ln pi]
    VectorXd expt_lambda = a_pos.array() / b_pos.array();
    VectorXd expt_ln_lambda = stan::math::digamma(a_pos.array()) - stan::math::log(b_pos.array());
    VectorXd expt_ln_pi = stan::math::digamma(alpha_pos.array()) - stan::math::digamma(stan::math::sum(alpha_pos.array()));

    // calc E[ln eta], E[S], E[ln S]
    // S is a variable translated from s with one-hot-labeling
    MatrixXd expt_ln_eta(N,K);
    MatrixXd expt_ln_S(N,K);
    MatrixXd expt_S(N,K);
    for (int n = 0; n < N; n++) {
        expt_ln_eta.row(n) = X(n) * expt_ln_lambda - expt_lambda + expt_ln_pi;
        expt_ln_eta.row(n) = expt_ln_eta.row(n) - rep_matrix(stan::math::log_sum_exp(expt_ln_eta.row(n)),1,K);
        expt_ln_S.row(n) = expt_ln_eta.row(n);
    }
    expt_S = exp(expt_ln_S);

    // calc log-likelihood
    double expt_ln_lkh = 0;
    for (int n = 0; n < N; n++) {
        expt_ln_lkh += expt_S.row(n) * (X(n) * expt_ln_lambda - expt_lambda - rep_matrix(stan::math::lgamma(X(n)+1),K,1));
    }
    
    // calc E[ln p(S)] and E[ln q(S)]
    double expt_ln_pS = sum(expt_S * expt_ln_pi);
    double expt_ln_qS = sum(expt_S.array() * expt_ln_S.array());

    // calc KL[q(lambda) || p(lambda)]
    double KL_lambda = stan::math::sum(
        a_pos.array() * log(b_pos.array()) - a_pri.array() * log(b_pri.array()) -
        stan::math::lgamma(a_pos.array()) + stan::math::lgamma(a_pri.array()) +
        (a_pos.array() - a_pri.array()) * expt_ln_lambda.array() +
        (b_pri.array() - b_pos.array()) * expt_lambda.array()
    );

    // calc KL[q(pi) || p(pi)]
    double KL_pi = 
        stan::math::lgamma(sum(alpha_pos)) - stan::math::lgamma(sum(alpha_pri)) -
        sum(stan::math::lgamma(alpha_pos)) + sum(stan::math::lgamma(alpha_pri)) +
        (alpha_pos - alpha_pri).transpose() * expt_ln_pi;
    
    return expt_ln_lkh + expt_ln_pS - expt_ln_qS - (KL_lambda + KL_pi);
}

void GibbsSampling(int N, int K, VectorXi X, int MAXITER, int seed) {
    // function to do Gibbs Sampling
    // inputs:
    //   N: the number of data points
    //   K: the number of clusters
    //   X: the data
    //   MAXITER: the maximum number of iterations
    //   seed: the random seed value
    
    // set random engine with the random seed value
    std::default_random_engine engine(seed);

    // set the output file
    std::ofstream samples("GS-samples.csv");
    // set the headers in the file
    for (int k = 0; k < K; k++) samples << "a." << k << ",";
    for (int k = 0; k < K; k++) samples << "b." << k << ",";
    for (int k = 0; k < K; k++) samples << "lambda." << k << ",";
    for (int k = 0; k < K; k++) samples << "alpha." << k << ",";
    for (int k = 0; k < K; k++) samples << "pi." << k << ",";
    for (int n = 0; n < N; n++) samples << "s." << n << ",";
    samples << "ELBO" << std::endl;

    // set variables
    VectorXd a; // the shape parameter
    VectorXd a_hat; // the modified shape parameter
    VectorXd b; // the rate parameter
    VectorXd b_hat; // the modified rate parameter
    VectorXd alpha; // the concentration parameter
    VectorXd alpha_hat; // the modified concentration parameter
    VectorXd lambda; // the rate parameter
    VectorXd pi; // the mixing parameter
    MatrixXd eta(N,K); // the temporal parameter to sample s
    VectorXi s; // the latent variable
    MatrixXi S(N,K); // the latent variable (one-hot-labeling)
    double ELBO; // ELBO

    // set initial values
    a = VectorXd::Constant(K,1,uniform_rng(0.1,2.0,engine));
    b = VectorXd::Constant(K,1,uniform_rng(0.005,0.05,engine)); 
    lambda = to_vector(gamma_rng(a,b,engine));
    alpha = VectorXd::Constant(K,1,uniform_rng(10.0,200.0,engine));
    pi = dirichlet_rng(alpha,engine);
    
    // sampling
    for (int i = 1; i <= MAXITER; i++) {
        // sample s and S
        s = VectorXi::Zero(N,1); // initialize s with zeros
        S = MatrixXi::Zero(N,K); // initialize S with zeros
        for (int n = 0; n < N; n++) {
            eta.row(n) = X(n) * log(lambda) - lambda + log(pi);
            eta.row(n) -= rep_row_vector(log_sum_exp(eta.row(n)),K);
            s(n) = categorical_rng(exp(eta.row(n)), engine);
            S(n,s(n)-1) = 1;
        }
        
        // sample lambda
        a_hat = a + S.transpose() * X;
        b_hat = b + S.colwise().sum().transpose();
        lambda = to_vector(gamma_rng(a_hat,b_hat,engine));
        
        // sample pi
        alpha_hat = alpha + S.colwise().sum().transpose();
        pi = dirichlet_rng(alpha_hat,engine);
                    
        // calc ELBO
        ELBO = calc_ELBO(N, K, X, a, b, alpha, a_hat, b_hat, alpha_hat);

        // output
        for (int k = 0; k < K; k++) samples << a_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << b_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << lambda(k) << ",";
        for (int k = 0; k < K; k++) samples << alpha_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << pi(k) << ",";
        for (int n = 0; n < N; n++) samples << s(n) << ",";
        samples << ELBO << std::endl;
    }
}

void VariationalInference(int N, int K, VectorXi X, int MAXITER, int seed) {
    // function to do Variational Inference
    // inputs:
    //   N: the number of data points
    //   K: the number of clusters
    //   X: the data
    //   MAXITER: the maximum number of iterations
    //   seed: the random seed value

    // set random engine with the random seed value
    std::default_random_engine engine(seed);
    
    // set the output file
    std::ofstream samples("VI-samples.csv");
    // set the headers in the file
    for (int k = 0; k < K; k++) samples << "a." << k << ",";
    for (int k = 0; k < K; k++) samples << "b." << k << ",";
    for (int k = 0; k < K; k++) samples << "lambda." << k << ",";
    for (int k = 0; k < K; k++) samples << "alpha." << k << ",";
    for (int k = 0; k < K; k++) samples << "pi." << k << ",";
    for (int n = 0; n < N; n++) samples << "s." << n << ",";
    samples << "ELBO" << std::endl;
    
    // set variables
    VectorXd a; // the shape parameter
    VectorXd a_hat; // the modified shape parameter
    VectorXd b; // the rate parameter
    VectorXd b_hat; // the modified rate parameter
    VectorXd alpha; // the concentration parameter
    VectorXd alpha_hat; // the modified concentration parameter
    VectorXd lambda; // the rate parameter
    VectorXd pi; // the mixing parameter
    VectorXd s; // the latent variable
    MatrixXd expt_S(N,K); // E[S]
    MatrixXd ln_expt_S(N,K); // ln E[S]
    double ELBO; // ELBO
    
    // set initial values
    a = VectorXd::Constant(K,1,uniform_rng(0.1,2.0,engine));
    b = VectorXd::Constant(K,1,uniform_rng(0.005,0.05,engine));
    alpha = VectorXd::Constant(K,1,uniform_rng(10.0,200.0,engine));
    pi = dirichlet_rng(alpha,engine);
    s = VectorXd::Zero(N,1); // initialize s with zeros
    expt_S = MatrixXd::Zero(N,K); // initialize expt_S with zeros
    for (int n = 0; n < N; n++) {
        s(n) = categorical_rng(pi,engine);
        expt_S(n,s(n)-1) = 1;
    }
    a_hat = a + expt_S.transpose() * X;
    b_hat = b + expt_S.colwise().sum().transpose();
    alpha_hat = alpha + expt_S.colwise().sum().transpose();
    
    // sampling
    for (int i = 1; i <= MAXITER; i++) {
        // sample s
        VectorXd expt_lambda = a_hat.array() / b_hat.array();
        VectorXd expt_ln_lambda = stan::math::digamma(a_hat.array()) - stan::math::log(b_hat.array());
        VectorXd expt_ln_pi = stan::math::digamma(alpha_hat.array()) - stan::math::digamma(stan::math::sum(alpha_hat.array()));
        for (int n = 0; n < N; n++) {
            ln_expt_S.row(n) = X(n) * expt_ln_lambda - expt_lambda + expt_ln_pi;
            ln_expt_S.row(n) -= rep_row_vector(log_sum_exp(ln_expt_S.row(n)), K);
            expt_S.row(n) = exp(ln_expt_S.row(n));
            s(n) = categorical_rng(expt_S.row(n), engine);
        }
        
        // sample lambda
        a_hat = a + expt_S.transpose() * X;
        b_hat = b + expt_S.colwise().sum().transpose();
        lambda = to_vector(gamma_rng(a_hat,b_hat,engine));
        
        // sample pi
        alpha_hat = alpha + expt_S.colwise().sum().transpose();
        pi = dirichlet_rng(alpha_hat,engine);

        // calc ELBO
        ELBO = calc_ELBO(N, K, X, a, b, alpha, a_hat, b_hat, alpha_hat);
        
        // output
        for (int k = 0; k < K; k++) samples << a_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << b_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << lambda(k) << ",";
        for (int k = 0; k < K; k++) samples << alpha_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << pi(k) << ",";
        for (int n = 0; n < N; n++) samples << s(n) << ",";
        samples << ELBO << std::endl;
    }
}

void CollapsedGibbsSampling(int N, int K, VectorXi X, int MAXITER, int seed) {
    // function to do Collapsed Gibbs Sampling
    // inputs:
    //   N: the number of data points
    //   K: the number of clusters
    //   X: the data
    //   MAXITER: the maximum number of iterations
    //   seed: the random seed value

    // set random engine with the random seed value
    std::default_random_engine engine(seed);
    
    // set the output file
    std::ofstream samples("CGS-samples.csv");
    // set the headers in the file
    for (int k = 0; k < K; k++) samples << "a." << k << ",";
    for (int k = 0; k < K; k++) samples << "b." << k << ",";
    for (int k = 0; k < K; k++) samples << "alpha." << k << ",";
    for (int n = 0; n < N; n++) samples << "s." << n << ",";
    samples << "ELBO" << std::endl;
    
    // set variables
    VectorXd a; // the shape parameter
    VectorXd a_hat; // the modified shape parameter
    VectorXd b; // the rate parameter
    VectorXd b_hat; // the modified rate parameter
    VectorXd alpha; // the concentration parameter
    VectorXd alpha_hat; // the modified concentration parameter
    VectorXd s; // the latent variable
    MatrixXd S(N,K); // the latent variable (one-hot-labeling)
    VectorXd sX_csum; // represents the sum_{n}^{N}(s_{n,k} * x_{n})
    VectorXd S_csum; // represents the sum_{n}^{n}(s_{n,k})
    VectorXd ln_lkh; // represents (log-) likelihood at Eq. (4.76)
    double ELBO; // ELBO
    
    // set initial values
    a = VectorXd::Constant(K,1,uniform_rng(0.1,2.0,engine));
    b = VectorXd::Constant(K,1,uniform_rng(0.005,0.05,engine));
    alpha = VectorXd::Constant(K,1,uniform_rng(10.0,200.0,engine));
    s = VectorXd::Zero(N,1); // initialize s with zeros
    S = MatrixXi::Zero(N,K); // initialize S with zeros
    for (int n = 0; n < N; n++) {
        s(n) = categorical_rng(rep_vector(1.0/K,K),engine);
        S(n,s(n)-1) = 1;
    }
    // calc a_hat, b_hat, and alpha_hat
    sX_csum = (S.array() * X.rowwise().replicate(K).array()).matrix().colwise().sum();
    S_csum = S.colwise().sum();
    a_hat = a + sX_csum;
    b_hat = b + S_csum;
    alpha_hat = alpha + S_csum;
    
    // sampling
    for (int i = 1; i <= MAXITER; i++) {
        for (int n = 0; n < N; n++) {
            // remove components related to x_{n}
            a_hat -= S.row(n) * X(n);
            b_hat -= S.row(n);
            alpha_hat -= S.row(n);

            // calc ln_lkh
            ln_lkh = VectorXd::Zero(K); // initialize ln_lkh with zeros
            for (int k = 0; k < K; k++) {
                ln_lkh(k) = neg_binomial_lpmf(X(n),a_hat(k),b_hat(k)) + stan::math::log(alpha_hat(k));
            }

            // sample s
            S = MatrixXi::Zero(N,K); // initialize S with zeros
            ln_lkh -= rep_vector(log_sum_exp(ln_lkh),K);
            s(n) = categorical_rng(stan::math::exp(ln_lkh),engine);
            S(n,s(n)-1) = 1;

            // add components related to x_{n}
            a_hat += S.row(n) * X(n);
            b_hat += S.row(n);
            alpha_hat += S.row(n);
        }

        // calc ELBO
        ELBO = calc_ELBO(N, K, X, a, b, alpha, a_hat, b_hat, alpha_hat);
        
        // output
        for (int k = 0; k < K; k++) samples << a_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << b_hat(k) << ",";
        for (int k = 0; k < K; k++) samples << alpha_hat(k) << ",";
        for (int n = 0; n < N; n++) samples << s(n) << ",";
        samples << ELBO << std::endl;
    }
}

int main(int argc, char *argv[]) {
    
    // get inputs 1 ~ 4
    std::string method = argv[1];
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int seed = atoi(argv[4]);

    if (method == "data") {
        
        // get parameters
        VectorXd lambda; // the rate parameter in poisson distribution
        VectorXd pi; // the mixing parameter
        lambda = VectorXd::Zero(K); // initialize with zeros
        pi = VectorXd::Zero(K); // initialize with zeros
        
        for (int k = 0; k < K; k++) lambda(k) = atof(argv[5+k]);
        for (int k = 0; k < K; k++)     pi(k) = atof(argv[5+K+k]);
        
        std::cout << "Random Data Generation" << std::endl;
        generate_data(N, K, lambda, pi, seed);

    } else {
        
        int MAXITER = atoi(argv[5]);

        // read data.csv
        VectorXi X;
        X = VectorXi::Zero(N);
        std::ifstream ifs("data.csv");
        std::string line;
        int idx = -1;
        while (getline(ifs, line)) {
            std::vector<std::string> strvec = split(line, ',');
            if (idx == -1) {
                idx++;
                continue;
            }
            X(idx) = stoi(strvec.at(1));
            idx++;
        }
        
        if (method == "GS") {

            std::cout << "Gibbs Sampling" << std::endl;
            GibbsSampling(N, K, X, MAXITER, seed);

        } else if (method == "VI") {

            std::cout << "Variational Inference" << std::endl;
            VariationalInference(N, K, X, MAXITER, seed);

        } else if (method == "CGS") {

            std::cout << "Collapsed Gibbs Sampling" << std::endl;
            CollapsedGibbsSampling(N, K, X, MAXITER, seed);

        } else {

            std::cout << "No valid samling method was selected." << std::endl;

        }
    }
}