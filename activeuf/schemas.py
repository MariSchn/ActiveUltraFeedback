from pydantic import BaseModel, Field, root_validator
from typing import Optional

from activeuf.configs import PROMPT_SOURCES

class Prompt(BaseModel):
    source: str
    prompt: str
    prompt_id: str

    @root_validator(pre=True)
    def check_source(cls, values):
        source = values["source"]
        if source is not None and source not in PROMPT_SOURCES:
            raise ValueError(f"Invalid source: {source}. Must be one of {PROMPT_SOURCES}.")
        return values

class Annotation(BaseModel):
    annotator_name: str
    aspect: str

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
