"""
Bilby results classes
=====================

Usage e.g.,

    bilby.run_sampler(..., sampler="pymultinest", result_class=MultiNestResult)

The results will have nested sampling or MCMC specific methods.
"""

from .mcmc import MCMCResult, EmceeResult
from .ns import PolyChordResult, MultiNestResult, DynestyResult, UltraNestResult, JaxNSResult, NessaiResult, DynamicDynestyResult
