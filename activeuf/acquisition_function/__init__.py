from activeuf.acquisition_function.dts import DoubleThompsonSampling
from activeuf.acquisition_function.infomax import InfoMax
from activeuf.acquisition_function.random import RandomAcquisitionFunction
from activeuf.acquisition_function.maxminlcb import MaxMinLCB
from activeuf.acquisition_function.infogain import InfoGain
from activeuf.acquisition_function.ultrafeedback import UltraFeedback
from activeuf.acquisition_function.ids import InformationDirectedSampling
from activeuf.acquisition_function.rucb import RelativeUpperConfidenceBound
from activeuf.acquisition_function.drts import DoubleReverseThompsonSampling
from activeuf.acquisition_function.deltaucb import DeltaUCB

__all__ = [
    "RandomAcquisitionFunction",
    "UltraFeedback",
    "DoubleThompsonSampling",
    "InfoMax",
    "MaxMinLCB",
    "InfoGain",
    "InformationDirectedSampling",
    "RelativeUpperConfidenceBound",
    "DeltaUCB",
]

_acquisition_function_map = {
    "random": RandomAcquisitionFunction,
    "ultrafeedback": UltraFeedback,
    "dts": DoubleThompsonSampling,
    "infomax": InfoMax,
    "maxminlcb": MaxMinLCB,
    "infogain": InfoGain,
    "ids": InformationDirectedSampling,
    "rucb": RelativeUpperConfidenceBound,
    "drts": DoubleReverseThompsonSampling,
    "deltaucb": DeltaUCB,
}


def init_acquisition_function(key: str, *args, **kwargs):
    if key in _acquisition_function_map:
        return _acquisition_function_map[key](*args, **kwargs)
    else:
        raise ValueError(
            f"Acquisition function '{key}' not found. Available: {list(_acquisition_function_map.keys())}"
        )
