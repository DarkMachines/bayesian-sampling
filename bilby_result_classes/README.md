# Install dependencies

    pip install -r requirements.txt

NB this installs a beta version of anesthetic that avoids a version number requirement clash with bilby.

# Usage

Usage e.g.,

    from bilby_result_classes import MCMCResult, EmceeResult, PolyChordResult, MultiNestResult, DynestyResult
    bilby.run_sampler(..., sampler="pymultinest", result_class=MultiNestResult)
    bilby.run_sampler(..., sampler="emcee", result_class=Emcee)

The results will have nested sampling or MCMC specific methods. E.g., MCMC methods `rhat`, `ess` and `metric`, and NS methods `test`, `ess` and `metric`.
