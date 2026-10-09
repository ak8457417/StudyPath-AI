from typing import Annotated, TypedDict, List
import operator
from pydantic import BaseModel, Field

class StudyPlan(BaseModel):
    weeks: List[dict] = Field(description="List of weekly study plans")
    quiz_status: dict = Field(description="Quiz completion status for each week")

class AgentState(TypedDict):
    subject: str
    topic: str
    current_level: str
    target_level: str
    hours_per_week: int
    prior_knowledge: str
    diagnostic: dict
    study_notes: dict                        # Content Generator Agent output
    quiz_questions: list
    quiz_answers: list
    feedback_report: str
    adjusted_difficulty: str                 # Feedback loop output
    messages: Annotated[list, operator.add]  # LangChain memory messages
    week_number: int
    plan: dict
    error: str

