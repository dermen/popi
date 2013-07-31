
import sys
import pymc
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from pprint import pprint

"""
A simple script that uses PyMC to numerically integrate a Bayes Factor comparing
two models: one is that the data come from a Gaussian distribution (the 'first'
model here) or that the date come from a mixture of a Gaussian and Pareto (the
'second' model). These models a specifically interesting in the interpretation 
of PC1 amplitudes for inter-vs-intra shot correlator FFTs in CXS data.

To preform this comparison, place your PC1 amplitudes in a flat text file, and
call

    $ python calc_bayes_factor.py <file>

the output should be easy to read. A convergence test plot will also be written
to disk. You're looking for the plot therein to flatten off for many iterations.
"""

class BayesFactor(object):
    
    def __init__(self, prior_dict, n_samples, data, verbose=False):
        
        self.data = data
        self.n_samples = int(n_samples)
        self._set_prior(prior_dict)
        
        self.name1 = 'Gaussian Model'
        self.name2 = 'Gaussian/Pareto Mixture Model'
        
        self.verbose = verbose
        
        
    def _set_prior(self, prior_dict):
        
        self.sigma_min = prior_dict['sigma_min']
        self.sigma_max = prior_dict['sigma_max']
        
        self.mu_min = prior_dict['mu_min']
        self.mu_max = prior_dict['mu_max']
        
        self.nu_min = prior_dict['nu_min']
        self.nu_max = prior_dict['nu_max']
        
        self.m_min = prior_dict['m_min']
        self.m_max = prior_dict['m_max']
        
        print "\n --- Initialized Prior ---"
        pprint(prior_dict)
        

    def _logP_gaussian(self):

        # convert sigmas to taus (note max/min switch)
        tau_max = np.power( self.sigma_min, -2 )
        tau_min = np.power( self.sigma_max, -2 )

        # the priors & gaussian likelihood
        prior_mu  = pymc.Uniform('prior_mu', lower=self.mu_min, upper=self.mu_max,
                                 doc='PriorOnMu', verbose=self.verbose)
        prior_tau = pymc.Uniform('prior_tau', lower=tau_min, upper=tau_max,
                                 doc='PriorOnTau', verbose=self.verbose)
        likelihood = pymc.Normal('gaussian', mu=prior_mu, tau=prior_tau,
                                 value=self.data, observed=True, verbose=self.verbose)


        # now, calculate the Bayes factor based on the log-likelihood of
        # each model by sampling from the prior (MC integration)

        logP_trace = np.zeros(self.n_samples)
        print " --- Running %d Samples: Gaussian Model ---" % self.n_samples

        for i in range(self.n_samples):

            if i % 1000 == 0:
                print "Sample: %d/%d" % (i, self.n_samples)

            # sample from the posterior
            prior_mu.random()
            prior_tau.random()
            logP_trace[i] = likelihood.logp

        # sometimes we get large positive numbers... that's not supposed to happen
        bad_inds = np.where(logP_trace > 0.0)[0]
        if len(bad_inds) > 1:
            print "\nWarning: found %d values with probabilities greater than 1!" % len(bad_inds)
            logP_trace[bad_inds] = -10.**300

        logP = np.sum(np.nan_to_num( np.exp( logP_trace ) )) / float(self.n_samples)
    
        return logP, logP_trace


    def _logP_gaussian_power_mix(self):
        """
        self.data : ndarray, float
            An array of the experimentally observed amplitudes.
        """

        # convert sigmas to taus (note max/min switch)
        tau_max = np.power( self.sigma_min, -2 )
        tau_min = np.power( self.sigma_max, -2 )

        # the priors & gaussian likelihood
        prior_mu  = pymc.Uniform('prior_mu', lower=self.mu_min, upper=self.mu_max,
                                 doc='PriorOnMu', verbose=self.verbose)
        prior_tau = pymc.Uniform('prior_tau', lower=tau_min, upper=tau_max,
                                 doc='PriorOnTau', verbose=self.verbose)
        prior_nu  = pymc.Uniform('prior_nu', lower=self.nu_min, upper=self.nu_max,
                                 doc='PriorOnNu', verbose=self.verbose)
        prior_m   = pymc.Uniform('prior_m', lower=self.m_min, upper=self.m_max,
                                 doc='PriorOnM', verbose=self.verbose)
        prior_alpha = pymc.Uniform('prior_alpha', lower=0.0, upper=1.0,
                                 doc='PriorOnAlpha', verbose=self.verbose)
                             
        gauss_lkldh = pymc.Normal('gaussian', mu=prior_mu, tau=prior_tau,
                                  value=self.data, observed=True, verbose=self.verbose)
        pl_lkldh    = pymc.Pareto('powerlaw', prior_nu, prior_m,
                                  value=self.data, observed=True, verbose=self.verbose)

        logP_trace = np.zeros(self.n_samples)
        print " --- Running %d Samples: Gaussian Model ---" % self.n_samples

        for i in range(self.n_samples):

            if i+1 % 1000 == 0:
                print "Sample: %d/%d" % (i+1, self.n_samples)

            # sample from the posterior
            prior_mu.random()
            prior_tau.random()
            prior_nu.random()
            prior_m.random()
            
            a = prior_alpha.random()
            logP_trace[i] = np.log(a*gauss_lkldh.value + (1.0-a)*pl_lkldh.value)

        # sometimes we get large positive numbers... that's not supposed to happen
        bad_inds = np.where(logP_trace > 0.0)[0]
        if len(bad_inds) > 1:
            print "\nWarning: found %d values with probabilities greater than 1!" % len(bad_inds)
            logP_trace[bad_inds] = -10.**300

        logP = np.sum(np.nan_to_num( np.exp( logP_trace ) )) / float(samples)

        return logP, logP_trace


    def _compute_bayes_factor(self):
                
        # calculate the bayes factor
        mean1 = np.sum(np.nan_to_num( np.exp( self.logP_trace1 ) )) / float(len(self.logP_trace1))
        mean2 = np.sum(np.nan_to_num( np.exp( self.logP_trace2 ) )) / float(len(self.logP_trace2))
        self.K = np.log10( mean1 / mean2 )

        print "\n--- Calculated Bayes Factor ---"
        print "Log10 factor for %s/%s models: %f" % (name1, name2, self.K)
        print "LogP for %s:\t%f" % (self.name1, mean1)
        print "LogP for %s:\t%f" % (self.name2, mean2)
        print
    
        return


    def _plot_convergence(self, subsample=100):

        plt.figure(figsize=(6,4))
        sub_sampled = np.arange(0.0, float(self.self.n_samples/subsample)) * subsample
        K_series = np.cumsum(np.nan_to_num( np.exp( self.logP_trace1 ) ))[::subsample] / \
                   np.cumsum(np.nan_to_num( np.exp( self.logP_trace2 ) ))[::subsample]            

        plt.plot( sub_sampled, np.log10(self.K_series), lw=2 )
        plt.xlabel('Samples')
        plt.ylabel('Convergence')
        plt.savefig('convergence.pdf', bbox_inches='tight', pad_inches=0.2)
        print "Saved: convergence.pdf"

        return


    def compare_models(self, plot_convergence=True):

        self.logP1, self.logP_trace1 = self._logP_gaussian()
        self.logP2, self.logP_trace2 = self._logP_gaussian_power_mix()
        self._compute_bayes_factor()
        
        if plot_convergence:
            self._plot_convergence()
    
        return
    
        
    def ml_params(self, model='mixture'):
        """
        Assuming uniform priors, does a simplex search for the most likely values.
    
        Parameters
        ----------
        model : {'mixture', 'gaussian'}
    
        Returns
        -------
        params : tuple
            Either:
                (mu, sigma, nu, m, alpha)
            Or:
                (mu, sigma)     
        """
        
        if model == 'mixture':
            objective = self.mix_nlogP
            x0 = (self.data.mean(), self.data.std(), 1.0, 1e-6, 0.5)
        elif model == 'gaussian':
            objective = self.gauss_nlogP
            x0 = (self.data.mean(), self.data.std())
        else:
            raise ValueError('Cant understand model: %s' % model)
        
        xopt = optimize.fmin(objective, x0, xtol=10.**-15, ftol=10.**-15,
                             maxiter=10**6, maxfun=10**6)
            
        return xopt
    
    
    def mix_nlogP(self, params):
        """ unnormalized """
        mu, sigma, nu, m, alpha = params
        pG = alpha * np.exp( - (1.0/(2.0*np.power(sigma,2))) * np.power(data-mu,2) )
        pP = (1.0-alpha) * np.power( nu * np.power(m,nu) / np.power(data, nu+1) )
        pP[ data < m ] = 0.0
        logP = -1.0 * np.sum( np.log( pG + pP ) )
        return logP
        
        
    def gauss_nlogP(self, params):
        """ unnormalized """
        mu, sigma = params
        pG = np.exp( - (1.0/(2.0*np.power(sigma,2))) * np.power(data-mu,2) )
        logP = -1.0 * np.sum( np.log( pG ) )
        return logP
        


def main():
    
    print "usage: python calc_bayes_factor.py <file> <{inter, intra}>"
    
    n_samples = 1e4
    
    f = np.load(sys.argv[1])
    data = f[sys.argv[2]]
    data += np.abs(data.min()) # shift so all are non-zero
    print "Loaded data file: %s" % sys.argv[1]
    
    # I'm going to violate good practice and look at the data to set my prior
    # since the priors are just uniforms, this shouldn't change the answer
    # hopefully just speed up convergence...
    
    prior_dict = {'sigma_min' : data.std() / 2.0,
                  'sigma_max' : data.std() * 2.0,
                  'mu_min'    : data.mean() - data.std(),
                  'mu_max'    : data.max()  + data.std(),
                  'nu_min'    : 0.0,
                  'nu_max'    : 5.0,          # set this one
                  'm_min'     : 0.0,
                  'm_max'     : data.max()    # check this one
                  }
                  
    bf = BayesFactor(prior_dict, n_samples, data, verbose=False)
    bf.compare_models()
    
    return
    

if __name__ == '__main__':
    main()
    
    

