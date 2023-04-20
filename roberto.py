from bilby_result_classes import PolyChordResult, MultiNestResult, DynestyResult, UltraNestResult, JaxNSResult,  NessaiResult, DynamicDynestyResult

import bilby

def summarize(json, **kwargs):

    print("")
    print(json)
    res = bilby.result.read_in_result(json, **kwargs)

    methods = ["kish", "unbiased_kish", "information", "mean", "bootstrap"]
    ess = {method: res.ess(method=method, nsamples=100) for method in methods}
    ess = {k: v for k, v in ess.items() if v is not None}
    print("Effective sample size = ", ess)

    metric = {method: res.metric(method=method, nsamples=100) for method in methods}
    metric = {k: v for k, v in metric.items() if v is not None}
    print("Efficiency (ESS / # like calls) = ", metric)

    print("Test = ", res.test())

#summarize("results2D/ultranest_EggBox_result.json", result_class=UltraNestResult)
#summarize("results2D/pypolychord_EggBox_result.json", result_class=PolyChordResult)
#summarize("results2D/dynesty_EggBox_result.json", result_class=DynestyResult)
summarize("results2D/jaxns_EggBox_result.json", result_class=JaxNSResult)
summarize("results2D/nessai_EggBox_result.json", result_class=NessaiResult)
# summarize("results2D/pymultinest_EggBox_result.json", result_class=MultiNestResult)
summarize("results2D/dynamic_dynesty_EggBox_result.json", result_class=DynamicDynestyResult)

