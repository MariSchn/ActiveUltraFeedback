from dataclasses import dataclass, field


@dataclass
class RandomConfig:
    seed: int = field(
        metadata={"help": "Random seed for the random acquisition function."}
    )


@dataclass
class UltraFeedbackConfig:
    seed: int = field(
        metadata={"help": "Random seed for the ultrafeedback acquisition function."}
    )


@dataclass
class DTSConfig:
    max_iterations: int = field(metadata={"help": "Maximum iterations for DTS."})
    beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for the DTS acquisition function."},
    )


@dataclass
class InfoGainConfig:
    beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for the DTS acquisition function."},
    )


@dataclass
class IDSConfig:
    argmax_tol: float = field(metadata={"help": "Tolerance for argmax in IDS."})
    decision_buffer: float = field(metadata={"help": "Decision buffer for IDS."})
    use_candidate_set: bool = field(
        metadata={"help": "Whether to use candidate set in IDS."}
    )


@dataclass
class RUCBConfig:
    argmax_tol: float = field(metadata={"help": "Tolerance for argmax in RUCB."})
    decision_buffer: float = field(metadata={"help": "Decision buffer for RUCB."})
    use_candidate_set: bool = field(
        metadata={"help": "Whether to use candidate set in RUCB."}
    )
    beta: float = field(default=1.0, metadata={"help": "Beta parameter for RUCB."})


@dataclass
class MaxMinLCBConfig:
    argmax_tol: float = field(metadata={"help": "Tolerance for argmax in MaxMinLCB."})
    decision_buffer: float = field(metadata={"help": "Decision buffer for MaxMinLCB."})
    use_candidate_set: bool = field(
        metadata={"help": "Whether to use candidate set in MaxMinLCB."}
    )
    seed: int = field(metadata={"help": "Random seed for MaxMinLCB."})
    beta: float = field(default=1.0, metadata={"help": "Beta parameter for MaxMinLCB."})


@dataclass
class DeltaUCBConfig:
    beta: float = field(default=1.0, metadata={"help": "Beta parameter for DeltaUCB."})


@dataclass
class DeltaQuantileConfig:
    quantile: float = field(
        metadata={"help": "The center of the rank window (0.0 to 1.0)."}
    )
    epsilon: float = field(
        metadata={
            "help": "The half-width of the selection window. The function looks for pairs in the range [quantile - epsilon, quantile + epsilon]."
        }
    )
    beta: float = field(metadata={"help": "Beta parameter for DeltaQuantile."})


@dataclass
class DRTSConfig:
    max_iterations: int = field(metadata={"help": "Maximum iterations for DRTS."})
    beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for the DRTS acquisition function."},
    )
