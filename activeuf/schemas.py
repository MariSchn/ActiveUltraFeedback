from pydantic import BaseModel

class Prompt(BaseModel):
    source: str | None
    prompt: str
    prompt_id: str

class Annotation(BaseModel):
    aspect: str

    rating: str
    rating_rationale: str
    
    type_rating: str | None
    type_rationale: str | None

class Completion(BaseModel):
    prompt: Prompt

    model_path: str
    principle: str
    principle_prompt: str
    response_text: str

    annotations: list[Annotation] = []
    overall_score: str | None
    critique: str | None

class Message(BaseModel):
    content: str
    role: str

class BinaryPreferenceConversation(BaseModel):
    source: str | None
    prompt: str
    prompt_id: str

    chosen: list[Message]
    rejected: list[Message]
    messages: list[Message]

    score_chosen: float
    score_rejected: float

    completion_chosen: Completion | None
    completion_rejected: Completion | None

if __name__ == "__main__":
    annotation = Annotation(
        aspect="helpfulness", 
        rating="2", 
        rating_rationale="I liked it"
    )
    print(annotation)

    completion = Completion(
        model_path="gemma-3-1b",
        principle="honesty",
        principle_prompt="How can I help you?",
        response_text="I can help you with that.",
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
