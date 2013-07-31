
import sys
import pymc
import numpy as np
import matplotlib.pyplot as plt

VERBOSE = False

def load_data(filename):

    f = np.load(filename)
    inter = f['inter']
    intra = f['intra']
    
    return inter, intra


def main():

    print "usage: python %s <file>" % sys.argv[0]

    # get the data from disk
    inter, intra = load_data(sys.argv[1])
    combined = np.concatenate((inter, intra))
    print "loaded: %s" % sys.argv[1]

    # manual parameters
    samples = int(2e4)
    mu_min    = np.mean( combined ) - np.std( combined )
    mu_max    = np.mean( combined ) + np.std( combined )
    sigma_min = np.std( combined ) / 2
    sigma_max = np.std( combined ) * 2

    # convert sigmas to taus (note max/min switch)
    tau_max = np.power( sigma_min, -2 )
    tau_min = np.power( sigma_max, -2 )

    # we will use uniform priors on mu, sigma, and they will be
    # the same for all models
    
    prior_mu  = pymc.Uniform('prior_mu', lower=mu_min, upper=mu_max,
                             doc='PriorOnMu', verbose=VERBOSE)
    prior_tau = pymc.Uniform('prior_tau', lower=tau_min, upper=tau_max,
                             doc='PriorOnTau', verbose=VERBOSE)
    prior_mu2  = pymc.Uniform('prior_mu', lower=mu_min, upper=mu_max,
                             doc='PriorOnMu', verbose=VERBOSE)
    prior_tau2 = pymc.Uniform('prior_tau', lower=tau_min, upper=tau_max,
                             doc='PriorOnTau', verbose=VERBOSE)

    # the double-gaussian model will just be two single gaussians
    # corresponding to the two different datasets

    split_ts = pymc.Normal('split_ts', mu=prior_mu, tau=prior_tau,
                           value=inter, observed=True, verbose=VERBOSE)
    split_ms = pymc.Normal('split_ms', mu=prior_mu2, tau=prior_tau2,
                           value=intra, observed=True, verbose=VERBOSE)

    # the single-gaussian model will be a single gaussian for all
    # the data combined

    combined = pymc.Normal('combined', mu=prior_mu, tau=prior_tau,
                           value=combined, observed=True, verbose=VERBOSE)

    # now, calculate the Bayes factor based on the log-likelihood of
    # each model by sampling from the prior (MC integration)

    logP_trace_split = np.zeros(samples)
    logP_trace_combined = np.zeros(samples)
    print " --- Running %d Samples ---" % samples

    for i in range(samples):

        if i % 1000 == 0:
            print "Sample: %d/%d" % (i, samples)

        # sample from the prior
        prior_mu.random()
        prior_tau.random()
        prior_mu2.random()
        prior_tau2.random()
        
        # store log-probs
        logP_trace_split[i] = split_ts.logp + split_ms.logp

        # repeat for combined model
        prior_mu.random()
        prior_tau.random()
        logP_trace_combined[i] = combined.logp

    # calculate the bayes factor
    # to prevent extra-big/small numbers, center the data around a mean `alpha`
    alpha = (np.mean(logP_trace_split) + np.mean(logP_trace_combined)) / 2.0
    sp = logP_trace_split + 10000 
    cm = logP_trace_combined + 10000
    
    ssp = np.sum( np.nan_to_num( np.exp(sp.astype(np.float128))) )
    scm = np.sum( np.nan_to_num( np.exp(cm.astype(np.float128))) )
    
    bf = ssp / scm
    K = np.log10(bf)
    
    print "Calculated Bayes Factor:"
    print "Log10 factor for split/combined models: %f" % K

    # plot the convergence

    plt.figure(figsize=(6,4))

    subsample = 100
    sub_sampled = np.arange(0.0, float(samples/subsample)) * subsample
    K_series = np.cumsum(logP_trace_split)[::subsample] / \
               np.cumsum(logP_trace_combined)[::subsample]
    
    plt.plot( sub_sampled, (10 / np.exp(1)) * (K_series), lw=2 )
    plt.xlabel('Samples')
    plt.ylabel('Convergence')
    plt.savefig('convergence.pdf', bbox_inches='tight', pad_inches=0.2)
    print "Saved: convergence.pdf"
    

    return

if __name__ == '__main__':
    main()
    
    

