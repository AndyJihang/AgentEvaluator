
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Question(BaseModel):
    id: str
    text: str

class TaskSpec(BaseModel):
    task_id: str
    task_name: Optional[str] = ""
    prompt: str
    questions: List[Question]
    capsule_url: Optional[str] = None
    time_limit_sec: int = 1800
    env: Dict[str, Any] = Field(default_factory=dict)

    # Callbacks for A2A
    callback_submit_url: str
    callback_auth_token: Optional[str] = ""
    callback_results_url: Optional[str] = None

class AnswerItem(BaseModel):
    id: str
    answer: str

class Report(BaseModel):
    task_id: str
    answers: List[AnswerItem]
    trajectory: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Grade(BaseModel):
    task_id: str
    passed: bool
    score: float
    feedback: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

class DispatchRequest(BaseModel):
    purple_base_url: str
    auth_token: str = ""
    task_id: Optional[str] = None
    purple_results_url: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class DispatchResponse(BaseModel):
    run_id: str
    status: str
    message: str = ""
