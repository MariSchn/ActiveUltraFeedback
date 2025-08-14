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

class Message(BaseModel):
    role: str
    content: str

class Completion(BaseModel):
    model: str
    principle: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: list[Message]
    response_text: str

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
