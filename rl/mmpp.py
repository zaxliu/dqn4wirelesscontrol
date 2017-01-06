# Markov Modulated Poisson Process (MMPP)
# Author: Jingchu Liu

import numpy as np

from scipy.stats import poisson

from sklearn.utils import check_random_state

from hmmlearn.base import _BaseHMM
from hmmlearn.utils import normalize

__all__ = ["MMPP"]


class MMPP(_BaseHMM):
    """Hidden Markov Model with Poisson emissions."""

    def __init__(self, n_components=1, 
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm='viterbi', random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)


    def _init(self, X, lengths=None):
        if not self._check_input_symbols(X):
            raise ValueError("expected a sample from "
                             "a Poisson distribution.")

        super(MMPP, self)._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
              # Draw rates from Exponential distributions
              self.emissionrates_ = -1.0*np.log(
                self.random_state.rand(self.n_components)
              )

    def _check(self):
        super(MMPP, self)._check()

        # TODO: check shape and range of emissionlambdas

    def _compute_log_likelihood(self, X):
        # Utilize the broadcasting feature of poisson.pmf
        framelogprob = np.log(poisson.pmf(np.concatenate(X)[:, None], 
                                          self.emissionrates_[None, :]
                              )
                       )
        framelogprob = np.nan_to_num(framelogprob)  # prevend -inf
        return framelogprob

    def _generate_sample_from_state(self, state, random_state=None):
        return [poisson.rvs(self.emissionrates_[state])]

    def _initialize_sufficient_statistics(self):
        stats = super(MMPP, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, 1))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(MMPP, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            stats['obs'] = np.sum(np.concatenate(X)[:, None] * posteriors,
                                  axis=0).transpose()

    def _do_mstep(self, stats):
        super(MMPP, self)._do_mstep(stats)
        if 'e' in self.params:
            self.emissionrates_ = stats['obs'] \
               / np.sum(stats['trans'], axis=1)
        # # check for NaNs
        # np.nan_to_num(self.startprob_)
        # normalize(self.startprob_)
        # np.nan_to_num(self.transmat_)
        # normalize(self.transmat_)
        # np.nan_to_num(self.emissionrates_)
        # normalize(self.emissionrates_)

    def _check_input_symbols(self, X):
        symbols = np.concatenate(X)
        if (len(symbols) == 1 or          # not enough data
            symbols.dtype.kind != 'i' or  # not an integer
            (symbols < 0).any()):         # contains negative integers
            return False

        return True
