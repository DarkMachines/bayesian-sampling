"""
Specialized MCMC results class
============================
"""

import os
import pickle

from bilby.core.result import Result

import arviz


class MCMCResult(Result):

    def rhat(self, **kwargs):
        """
        :returns: Diagnostic for MCMC chain
        """
        return arviz.rhat(self.to_arviz(), **kwargs)

    def ess(self, method="mean", **kwargs):
        """
        :returns: Effective sample size of MCMC chain
        """
        return arviz.ess(self.to_arviz(), method=method, **kwargs)

    def metric(self, **kwargs):
        """
        :returns: Performance metric for MCMC chain
        """
        # TODO is this properly accounting for all like calls? how about burn in etc
        return self.ess(relative=True, **kwargs)

class EmceeResult(MCMCResult):

    def to_arviz(self):
        """
        Monkey patch 'to_arviz' method from bilby to properly support emcee

        :returns: arViz inference object
        """
        n = os.path.join(self.outdir, f"emcee_{self.label}", "sampler.pickle")
        with open(n, 'rb') as f:
            sampler = pickle.load(f)
        return arviz.from_emcee(sampler)


if __name__ == "__main__":

    import bilby
    from example import priors, likelihood

    sampler = "emcee"

    res = bilby.run_sampler(
              likelihood=likelihood,
              priors=priors,
              sampler=sampler,
              label=sampler,
              nwalkers=10,
              nsteps=100,
              result_class=EmceeResult
    )

    print(res.ess())
    print(res.metric())
    print(res.rhat())
