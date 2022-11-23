"""
Specialized NS results class
============================
"""

from abc import abstractmethod, abstractproperty, ABC
from functools import cached_property

import os
import pickle
import shutil

import numpy as np
from scipy.special import xlogy
from scipy.stats import mode

import anesthetic
from anesthetic.utils import insertion_p_value

from bilby.core.result import Result

import nestcheck.error_analysis
import nestcheck.data_processing
from nestcheck.estimators import param_mean


class NestedSamplingResult(Result, ABC):

    @cached_property
    def anesthetic(self):
        samples = self.to_anesthetic()
        samples._compute_insertion_indexes()
        return samples

    @abstractmethod
    def to_nestcheck(self):
        """
        :returns: Nestcheck object
        """

    @abstractmethod
    def to_anesthetic(self):
        """
        :returns: Anesthetic object
        """

    @abstractproperty
    def nlike(self):
        """
        :returns: Number of likelihood evaluations
        """

    @property
    def ndim(self):
        return len(self.parameter_labels)

    @property
    def parameters(self):
        return self.anesthetic.to_numpy()[:, :self.ndim].T

    @property
    def nlive(self):
        return int(mode(self.anesthetic.nlive)[0][0])

    def test(self, **kwargs):
        """
        :returns: Diagnostic for nested samples
        """
        return insertion_p_value(self.anesthetic.insertion, self.nlive, **kwargs)

    def weights(self, **kwargs):
        """
        :returns: Nested sampling weights
        """
        logw = self.anesthetic.logw(**kwargs).to_numpy()
        weights = np.exp(logw - self.anesthetic.logZ())
        return weights / weights.sum(axis=0)

    def ess(self, method="kish", nsamples=1000):
        """
        :returns: Effective sample size of nested samples
        """
        weights = self.weights(nsamples=nsamples)

        if method == "kish":
            sum_ = np.sum(weights**2, axis=0)
            return np.mean(1. / sum_)

        if method == "information":
            terms = xlogy(weights, weights)
            return np.mean(np.exp(-terms.sum(axis=0)))

        if method == "mean":
            mean = np.matmul(self.parameters, weights)
            var_mean = np.var(mean, axis=1)
            var = np.matmul(self.parameters**2, weights) - mean**2
            mean_var = np.mean(var, axis=1)
            return mean_var / var_mean

        if method == "bootstrap":
            run = self.to_nestcheck()
            std_mean = nestcheck.error_analysis.run_std_bootstrap(run, [param_mean], n_simulate=nsamples)
            var_mean = std_mean**2

            mean = np.matmul(self.parameters, weights)
            var = np.matmul(self.parameters**2, weights) - mean**2
            mean_var = np.mean(var, axis=1)
            return mean_var / var_mean

        raise RuntimeError(f"unknown method '{method}'")

    def metric(self, **kwargs):
        """
        :returns: Performance metric
        """
        return self.ess(**kwargs) / self.nlike

def dynesty_to_anesthetic(results):
    """
    :returns: Dynesty results object in anesthetic
    """
    logL = results.logl
    logL_birth = logL[results.samples_it]
    return anesthetic.NestedSamples(data=results.samples, logL=logL, logL_birth=logL_birth)

class DynestyResult(NestedSamplingResult):

    @property
    def sampler(self):
        """
        :returns: Unpickled sampler object
        """
        n = os.path.join(self.outdir, f"{self.label}_dynesty.pickle")
        with open(n, 'rb') as f:
            return pickle.load(f)

    def to_anesthetic(self):
        """
        :returns: anethetic nested samples object
        """
        return dynesty_to_anesthetic(self.sampler)

    def to_nestcheck(self):
        """
        :returns: nestcheck nested samples object
        """
        return nestcheck.data_processing.process_polychord_run(self.sampler)

    @property
    def nlike(self):
        return self.num_likelihood_evaluations

class PolyChordResult(NestedSamplingResult):

    @property
    def basedir(self):
        """
        :returns: Base directory of output files
        """
        return os.path.join(self.outdir, "chains")

    @property
    def root(self):
        """
        :returns: Root of output files
        """
        return os.path.join(self.basedir, self.label)

    def to_anesthetic(self):
        """
        :returns: anethetic nested samples object
        """
        return anesthetic.read.polychord.read_polychord(self.root)

    def to_nestcheck(self):
        """
        :returns: nestcheck nested samples object
        """
        return nestcheck.data_processing.process_polychord_run(self.label, self.basedir)

    @property
    def nlike(self):
        with open(f"{self.root}.resume", "r") as f:
            return int(f.readlines()[21])

class MultiNestResult(NestedSamplingResult):

    @property
    def basedir(self):
        """
        :returns: Base directory of output files
        """
        return os.path.join(self.outdir, f"pm_{self.label}")

    @property
    def root(self):
        """
        :returns: Root of output files
        """
        return os.path.join(self.basedir, "")

    def to_anesthetic(self):
        """
        :returns: anethetic nested samples object
        """
        return anesthetic.read.multinest.read_multinest(self.root)

    def to_nestcheck(self):
        """
        :returns: nestcheck nested samples object
        """
        # fix some IO silliness with dashes as bilby uses empty MN root parameter
        shutil.copyfile(os.path.join(self.basedir, "dead-birth.txt"),
                    os.path.join(self.basedir, "-dead-birth.txt"))
        shutil.copyfile(os.path.join(self.basedir, "phys_live-birth.txt"),
                    os.path.join(self.basedir, "-phys_live-birth.txt"))
        return nestcheck.data_processing.process_multinest_run("", self.basedir)

    @property
    def nlike(self):
        with open(f"{self.root}resume.dat", "r") as f:
            return int(f.readlines()[1].split()[1])


if __name__ == "__main__":

    import bilby
    from example import priors, likelihood

    sampler = "pymultinest"

    res = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sampler,
            nlive=100,
            label=sampler,
            result_class=MultiNestResult,
            )

    methods = ["kish", "information", "mean", "bootstrap"]
    ess = {method: res.ess(method=method) for method in methods}
    print(ess)

    metric = {method: res.metric(method=method) for method in methods}
    print(metric)

    print(res.test())
