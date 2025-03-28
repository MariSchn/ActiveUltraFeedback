from pydantic import BaseModel
from typing import Optional

class Annotation(BaseModel):
    principle: str

    rating: str
    rating_rationale: str
    
    type_rating: Optional[str] = None
    type_rationale: Optional[str] = None

class Completion(BaseModel):
    model_name: str
    principle: str
    prompt: str
    text: str

    annotations: Optional[list[Annotation]] = []

class Sample(BaseModel):
    instruction: str
    correct_answers: list[str]
    incorrect_answers: list[str]

    model_names: list[str]
    completions: Optional[list[Completion]] = []


if __name__ == "__main__":
    annotation = Annotation(
        principle="helpfulness", 
        rating="2", 
        rating_rationale="I liked it"
    )
    print(annotation)

    completion = Completion(
        model_name="gpt-2",
        principle="honesty",
        prompt="How can I help you?",
        text="I can help you with that.",
        annotations=[annotation],
    )
    print(completion)

    sample = Sample(
        instruction="How can I help you?",
        correct_answers=["I can help you with that."],
        incorrect_answers=["I can't help you with that."],
        model_names=["gpt-2"],
        completions=[completion],
    )
    print(sample)
