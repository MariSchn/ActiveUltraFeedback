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
    beta: float = field(
        metadata={"help": "Beta parameter for the DTS acquisition function."}
    )
    max_iterations: int = field(metadata={"help": "Maximum iterations for DTS."})


@dataclass
class InfoGainConfig:
    beta: float = field(
        metadata={"help": "Beta parameter for the DTS acquisition function."}
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
    beta: float = field(metadata={"help": "Beta parameter for RUCB."})
    argmax_tol: float = field(metadata={"help": "Tolerance for argmax in RUCB."})
    decision_buffer: float = field(metadata={"help": "Decision buffer for RUCB."})
    use_candidate_set: bool = field(
        metadata={"help": "Whether to use candidate set in RUCB."}
    )


@dataclass
class MaxMinLCBConfig:
    beta: float = field(metadata={"help": "Beta parameter for MaxMinLCB."})
    argmax_tol: float = field(metadata={"help": "Tolerance for argmax in MaxMinLCB."})
    decision_buffer: float = field(metadata={"help": "Decision buffer for MaxMinLCB."})
    use_candidate_set: bool = field(
        metadata={"help": "Whether to use candidate set in MaxMinLCB."}
    )
    seed: int = field(metadata={"help": "Random seed for MaxMinLCB."})
