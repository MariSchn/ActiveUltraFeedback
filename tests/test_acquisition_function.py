import torch

from activeuf.acquisition_function import *

REWARDS = torch.tensor([
    [   1,   0,   1, 0.5,   0, 0.2],
    [   0,   1,   1, 0.5, 0.2, 0.3],
    [ 0.5, 0.2, 0.1, 0.3, 0.4, 0.6],
    [0.1, 0.3, 0.2, 0.4, 0.5, 0.7],
])
STDS = torch.tensor([
    [0.5000, 0.8000, 1.0000, 0.6000, 2.6000, 0.5000],
    [3.5000, 1.0000, 0.5000, 0.5000, 0.3000, 0.3000],
    [0.9000, 0.4000, 0.2000, 0.5000, 0.5000, 0.6000],
    [0.3000, 0.2000, 0.2000, 0.3000, 1.1000, 0.5000]
])

INFOMAX_EXPECTED_OUTPUT = [
    [4, 2],
    [0, 1],
    [0, 5],
    [4, 5],
]


def test_infomax():
    infomax_acquisition = InfoMax()

    selected_indices = infomax_acquisition(
        rewards=REWARDS,
        std_deviation=STDS
    )

    for i, indices in enumerate(selected_indices):
        assert indices == INFOMAX_EXPECTED_OUTPUT[i], f"Expected {INFOMAX_EXPECTED_OUTPUT[i]} but got {indices} for input {i}"
    print("All tests passed for InfoMax acquisition function.")

if __name__ == "__main__":
    test_infomax()