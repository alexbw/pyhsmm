from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8

import pyhsmm
import pyhsmm.internals.transitions as transitions
pyhsmm.internals.states.use_eigen() # makes HMMs faster, message passing done in C++ with Eigen
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import cov

##################
#  loading data  #
##################

data = np.load('data/c57_spines_data.npz')['data']

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 50
T = data.shape[0]

obs_hypparams = dict(
        mu_0=data.mean(0),
        sigma_0=0.5*cov(data),
        kappa_0=0.25,
        nu_0=data.shape[1]+5
        )

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)]
trans_distn = transitions.UniformTransitions(
        pi=pyhsmm.distributions.MultinomialAndConcentration(1.,1./2,K=Nmax),
        lmbda_a_0=10*98.,lmbda_b_0=10*2.) # mean self-trans prob 0.98, mean dwell time 1/(1-0.98)=50 frames

posteriormodel = pyhsmm.models.HMM(
        init_state_concentration=Nmax, # doesn't matter with one observation sequence
        obs_distns=obs_distns,
        trans_distn=trans_distn)

posteriormodel.add_data(data)

savedstuff = []
for idx in progprint_xrange(50):
    posteriormodel.resample_model()
    savedstuff.append(
            (
                trans_distn.pi.weights.copy(),
                trans_distn.lmbda,
                posteriormodel.states_list[0].stateseq.copy()
            ))

plt.figure()
savedstuff = savedstuff[10:]
cluster_numbers = [(np.bincount(s[2]) > T*0.05).sum() for s in savedstuff]
lmbda_values = [s[1] for s in savedstuff]
plt.plot(cluster_numbers,lmbda_values,'bx')
plt.xlim(np.min(cluster_numbers)-1,np.max(cluster_numbers)+1)
plt.gcf().suptitle('Samples')

plt.show()
