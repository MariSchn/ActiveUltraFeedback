import argparse

from datasets import load_from_disk, Dataset

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    source: str
    prompt: str
    prompt_id: str


class Annotation(BaseModel):
    aspect: str

    text: str

    rating: str
    rating_rationale: str

    type: str = ""
    type_rationale: str = ""


class Message(BaseModel):
    role: str
    content: str


class Completion(BaseModel):
    model: str
    principle: str
    system_prompt: str
    messages: list[Message]
    response_text: str

    annotations: list[Annotation] = Field(default_factory=list)
    critique: str = ""
    overall_score: str = ""


class PromptWithCompletions(Prompt):
    completions: list[Completion] = Field(default_factory=list)


class BinaryPreferenceConversation(Prompt):
    chosen: list[Message]
    rejected: list[Message]
    messages: list[Message]

    score_chosen: float
    score_rejected: float

    completion_chosen: Completion
    completion_rejected: Completion

def convert_responses_to_activeuf_format(x: dict) -> dict:
    y = {}

    # setting this to dummy value instead of throwing it away, to be safe
    y["source"] = ""

    y["prompt_id"] = x["prompt_id"]
    y["prompt"] = x["chosen"][0]["content"]
    y["completions"] = [
        {
            # setting these to dummy values instead of throwing them away, to be safe
            "annotations": [],
            "critique": "",
            "overall_score": "",
            "system_prompt": "",
            "principle": "", 

            "model": x["chosen_model"],
            "messages": [x["chosen"][0]],
            "response_text": x["chosen"][1]["content"] if x["chosen"][1]["content"] else "",  # str, not list
        },
        {
            # setting these to dummy values instead of throwing them away, to be safe
            "annotations": [],
            "critique": "",
            "overall_score": "",
            "system_prompt": "",
            "principle": "", 

            "model": x["rejected_model"],
            "messages": [x["rejected"][0]],
            "response_text": x["rejected"][1]["content"] if x["rejected"][1]["content"] else "",  # str, not list
        }
    ]
    
    assert PromptWithCompletions.model_validate(y), "Validation failed"
    return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to the input dataset (load_from_disk-compatible folder).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to save the converted dataset (save_to_disk folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds: Dataset = load_from_disk(args.input_path)
    converted: Dataset = ds.map(convert_responses_to_activeuf_format)
    converted.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()