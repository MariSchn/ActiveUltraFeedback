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

class Message(BaseModel):
    role: str
    content: str

class Completion(BaseModel):
    model_api: str | None
    model_path: str | None
    principle: str
    system_prompt: str
    messages: list[Message]
    response_text: str

    annotations: list[Annotation] = Field(default_factory=list)
    overall_score: str | None = None
    critique: str | None = None

    @root_validator(pre=True)
    def check_model(cls, values):
        model_api = values["model_api"]
        model_path = values["model_path"]
        if bool(model_api) == bool(model_path):
            raise ValueError(f"Either model_api or model_path (but not both) must be provided.")
        return values

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