from pydantic import BaseModel, Field, root_validator

from activeuf.configs import PROMPT_SOURCES

class Prompt(BaseModel):
    source: str | None
    prompt: str
    prompt_id: str

    @root_validator(pre=True)
    def check_source(cls, values):
        source = values["source"]
        if source is not None and source not in PROMPT_SOURCES:
            raise ValueError(f"Invalid source: {source}. Must be one of {PROMPT_SOURCES}.")
        return values

class Annotation(BaseModel):
    aspect: str

    rating: str
    rating_rationale: str
    
    type_rating: str | None = None
    type_rationale: str | None = None

class Completion(BaseModel):
    model_name: str
    principle: str
    principle_prompt: str
    response_text: str

    annotations: Optional[list[Annotation]] = []

    critique: Optional[str] = None
    overall_score: Optional[str] = None

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
    overall_score: str | None = None
    critique: str | None = None

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
