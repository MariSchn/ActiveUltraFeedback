import os.path as path

from datasets import load_dataset, load_from_disk, Dataset

from activeuf.schemas import BinaryPreferenceConversation

def prepare_splits(
        dataset_path: str = None,
        train_path: str = None, 
        test_path: str = None, 
    ) -> tuple[Dataset, Dataset]:

    # try loading train data directly from HuggingFace
    if dataset_path:
        if dataset_path == "trl-lib/ultrafeedback_binarized":
            train_split = load_dataset(dataset_path, split="train")
            test_split = load_dataset(dataset_path, split="test")

        else:
            raise NotImplementedError(f"Loading of {dataset_path} must be implemented manually.")

    # check whether train data exists locally
    elif train_path and path.exists(train_path):
        train_split = load_from_disk(train_path)

        # if test data was also defined, load it (otherwise set it to [])
        if test_path and path.exists(test_path):
            test_split = load_from_disk(test_path)

    else:
        raise Exception(f"Dataset loading failed.")

    # sanity check: ensure each row is a BinaryPreferenceConversation
    for split in [train_split, test_split]:
        for x in split:
            BinaryPreferenceConversation.model_validate(x)

    return train_split, test_split