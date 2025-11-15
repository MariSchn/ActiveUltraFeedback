from pydantic import BaseModel, Field, root_validator


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
