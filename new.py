import streamlit as st
import os
import datetime
import random
import json
import re
import time
from datetime import timedelta
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
from youtube_search import YoutubeSearch
from bson.objectid import ObjectId
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated, Optional
import operator

# ── LangGraph imports ──────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="StudyPath AI", page_icon="📚", layout="wide")

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .roadmap {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        position: relative;
    }
    .roadmap-step {
        flex: 1;
        text-align: center;
        padding: 1rem;
        background: #101936;
        border-radius: 10px;
        margin: 0 0.5rem;
        position: relative;
        z-index: 2;
        color: white;
    }
    .roadmap-connector {
        position: absolute;
        top: 40%;
        left: 0;
        right: 0;
        height: 4px;
        background: #4CAF50;
        z-index: 1;
    }
    .video-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .video-card:hover { transform: translateY(-5px); }
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        background: #e0e0e0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: #4CAF50;
        transition: width 0.5s ease;
    }
    .backlog-item {
        border-left: 4px solid #FF9800;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        background: #fff3e0;
        border-radius: 0 4px 4px 0;
    }
    .backlog-item.completed {
        border-left: 4px solid #4CAF50;
        background: #e8f5e9;
    }
    .quiz-question {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1976D2;
        color: black;
    }
    .quiz-option {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .quiz-option.correct { background: #c8e6c9; border: 1px solid #4CAF50; }
    .quiz-option.incorrect { background: #ffcdd2; border: 1px solid #f44336; }
    .quiz-explanation { background: #e8f5e9; padding: 1rem; border-radius: 5px; margin-top: 0.5rem; }
    .improvement-card {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1976D2;
    }
    .project-card {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #9c27b0;
        color: black;
    }
    .score-display { font-size: 5rem; text-align: center; font-weight: bold; }
    .score-display.pass { color: #4CAF50; }
    .score-display.fail { color: #f44336; }
    /* Diagnostic Agent Card */
    .diagnostic-card {
        background: #fff8e1;
        border-left: 4px solid #FF9800;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        color: black;
    }
    .agent-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .badge-diagnostic { background: rgba(255,152,0,0.15); color: #E65100; border: 1px solid rgba(255,152,0,0.4); }
    .badge-quiz       { background: rgba(25,118,210,0.12); color: #1565C0; border: 1px solid rgba(25,118,210,0.3); }
    .badge-progress   { background: rgba(76,175,80,0.12);  color: #2E7D32; border: 1px solid rgba(76,175,80,0.3); }
    .badge-content    { background: rgba(156,39,176,0.12); color: #6A1B9A; border: 1px solid rgba(156,39,176,0.3); }
    .badge-langgraph  { background: rgba(0,150,136,0.12);  color: #00695C; border: 1px solid rgba(0,150,136,0.3); }
    .gap-tag {
        display: inline-block;
        background: #ffecb3;
        color: #BF360C;
        border: 1px solid #FFCC02;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 3px 4px;
    }
    .feedback-report {
        background: #f1f8e9;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #4CAF50;
        color: black;
    }
    .study-notes-card {
        background: #f3e5f5;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #9c27b0;
        color: black;
        margin: 0.5rem 0 1rem 0;
    }
    .langgraph-flow {
        background: #e0f2f1;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #00897B;
        margin: 0.5rem 0 1rem 0;
        color: black;
        font-size: 0.85rem;
    }
    .memory-card {
        background: #e8eaf6;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #3949AB;
        margin: 0.5rem 0 1rem 0;
        color: black;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
class StudyPlan(BaseModel):
    weeks: List[dict] = Field(description="List of weekly study plans")
    quiz_status: dict = Field(description="Quiz completion status for each week")


# ═══════════════════════════════════════════════════════════
#  LANGGRAPH STATE — shared state across all agents
# ═══════════════════════════════════════════════════════════
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


# ─────────────────────────────────────────────
# MongoDB Client
# ─────────────────────────────────────────────
class MongoDBClient:
    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MongoDB URI not found. Please set MONGO_URI in your environment variables.")
        self.client = MongoClient(mongo_uri)
        self.db = self.client.study_plans
        self.plans = self.db.plans
        self.backlog = self.db.backlog
        self.quizzes = self.db.quizzes
        self.memory_store = self.db.memory_store   # NEW: persisted LangChain memory

    def save_plan(self, plan):
        return self.plans.insert_one(plan).inserted_id

    def update_plan(self, plan_id, updated_plan):
        return self.plans.update_one({"_id": ObjectId(plan_id)}, {"$set": updated_plan})

    def update_progress(self, plan_id, topic, completed):
        self.plans.update_one({"_id": ObjectId(plan_id)}, {"$set": {f"progress.{topic}": completed}})

    def update_quiz_status(self, plan_id, week, status):
        self.plans.update_one({"_id": ObjectId(plan_id)}, {"$set": {f"quiz_status.week{week}": status}})

    def get_plan(self, plan_id):
        return self.plans.find_one({"_id": ObjectId(plan_id)})

    def get_all_plans(self):
        return list(self.plans.find())

    def add_to_backlog(self, plan_id, topic_data):
        self.backlog.insert_one({
            "plan_id": ObjectId(plan_id),
            "topic_data": topic_data,
            "added_on": datetime.datetime.now(),
            "completed": False
        })

    def get_backlog(self, plan_id):
        return list(self.backlog.find({"plan_id": ObjectId(plan_id)}))

    def update_backlog_item(self, backlog_id, completed):
        self.backlog.update_one({"_id": ObjectId(backlog_id)}, {"$set": {"completed": completed}})

    def remove_from_backlog(self, backlog_id):
        self.backlog.delete_one({"_id": ObjectId(backlog_id)})

    def save_quiz(self, quiz_data):
        return self.quizzes.insert_one(quiz_data).inserted_id

    def get_quiz(self, plan_id, week):
        return self.quizzes.find_one({"plan_id": ObjectId(plan_id), "week": week})

    def save_quiz_result(self, quiz_id, score, areas_for_improvement=None):
        self.quizzes.update_one(
            {"_id": ObjectId(quiz_id)},
            {"$set": {"score": score, "completed_at": datetime.datetime.now(),
                      "areas_for_improvement": areas_for_improvement or []}}
        )

    def save_diagnostic(self, plan_id, diagnostic):
        self.plans.update_one({"_id": ObjectId(plan_id)}, {"$set": {"diagnostic": diagnostic}})

    # ── LangChain Memory persistence ────────────────────────────────────────
    def save_memory(self, plan_id: str, messages: list):
        """Persist conversation memory for a plan across sessions."""
        self.memory_store.update_one(
            {"plan_id": plan_id},
            {"$set": {"messages": messages, "updated_at": datetime.datetime.now()}},
            upsert=True
        )

    def load_memory(self, plan_id: str) -> list:
        """Load persisted conversation memory for a plan."""
        doc = self.memory_store.find_one({"plan_id": plan_id})
        return doc.get("messages", []) if doc else []

    def append_memory_message(self, plan_id: str, role: str, content: str):
        """Append a single message to the memory store."""
        msg = {"role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()}
        self.memory_store.update_one(
            {"plan_id": plan_id},
            {"$push": {"messages": msg}, "$set": {"updated_at": datetime.datetime.now()}},
            upsert=True
        )

    # ── Adjusted difficulty persistence (feedback loop) ──────────────────────
    def save_adjusted_difficulty(self, plan_id: str, week: int, difficulty: str):
        self.plans.update_one(
            {"_id": ObjectId(plan_id)},
            {"$set": {f"adjusted_difficulty.week{week}": difficulty}}
        )

    def get_adjusted_difficulty(self, plan_id: str, week: int) -> str:
        doc = self.plans.find_one({"_id": ObjectId(plan_id)})
        if doc:
            return doc.get("adjusted_difficulty", {}).get(f"week{week}", "medium")
        return "medium"

    # ── Study notes persistence ───────────────────────────────────────────────
    def save_study_notes(self, plan_id: str, week: int, notes: dict):
        self.plans.update_one(
            {"_id": ObjectId(plan_id)},
            {"$set": {f"study_notes.week{week}": notes}}
        )

    def get_study_notes(self, plan_id: str, week: int) -> dict:
        doc = self.plans.find_one({"_id": ObjectId(plan_id)})
        if doc:
            return doc.get("study_notes", {}).get(f"week{week}", {})
        return {}


# ─────────────────────────────────────────────
# Gemini Helper
# ─────────────────────────────────────────────
def get_gemini_model(temperature=0.7):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found. Please enter it in the Settings tab.")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        google_api_key=api_key
    )

    # try:
    #     return ChatOllama(
    #         model="gemma3:4b",  # or "gemma:2b", "gemma2", etc.
    #         temperature=temperature,
    #     )
    # except Exception as e:
    #     st.error(f"Error connecting to local Ollama instance: {e}")
    #     return None


def safe_json_parse(raw: str):
    cleaned = raw.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════
#  LANGGRAPH AGENT NODES
#  Each function is a node in the LangGraph StateGraph.
# ═══════════════════════════════════════════════════════════

# ── Node 1: Diagnostic Agent ─────────────────────────────────────────────────
def diagnostic_agent_node(state: AgentState) -> AgentState:
    """
    Diagnostic Agent Node: assesses the student's profile, identifies knowledge
    gaps, and returns a structured diagnostic report.
    """
    model = get_gemini_model(temperature=0.3)
    if not model:
        state["diagnostic"] = _fallback_diagnostic(state["subject"], state["topic"], state["current_level"])
        state["messages"] = state.get("messages", []) + [
            {"role": "system", "content": f"Diagnostic complete for {state['subject']} at {state['current_level']} level."}
        ]
        return state

    prompt_template = """
You are a Diagnostic Agent for an AI-powered study platform.
Analyze this student profile and return ONLY valid JSON — no markdown, no explanation.

Subject: {subject}
Topic: {topic}
Current Level: {current_level}
Target Level: {target_level}
Weekly Hours Available: {hours_per_week}
Prior Knowledge (student's own words): {prior_knowledge}

Return exactly this JSON structure:
{{
  "assessed_level": "beginner | intermediate | advanced",
  "confidence_score": 0.0,
  "knowledge_gaps": ["gap1", "gap2", "gap3", "gap4"],
  "strong_areas": ["area1", "area2"],
  "recommended_difficulty": "easy | medium | hard",
  "learning_path": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ..."],
  "focus_areas": ["area1", "area2", "area3"],
  "estimated_weeks": 4,
  "diagnostic_summary": "Two-sentence summary of the student's state and what they need most."
}}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    try:
        response = chain.invoke({
            "subject": state["subject"],
            "topic": state["topic"],
            "current_level": state["current_level"],
            "target_level": state["target_level"],
            "hours_per_week": state["hours_per_week"],
            "prior_knowledge": state.get("prior_knowledge") or f"Currently at {state['current_level']} level."
        })
        result = safe_json_parse(response.content)
        if isinstance(result, dict) and "assessed_level" in result:
            state["diagnostic"] = result
        else:
            state["diagnostic"] = _fallback_diagnostic(state["subject"], state["topic"], state["current_level"])
    except Exception as e:
        state["diagnostic"] = _fallback_diagnostic(state["subject"], state["topic"], state["current_level"])
        state["error"] = str(e)

    # Append to conversation memory
    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": f"Diagnostic Agent: Assessed {state['subject']} profile. Level: {state['diagnostic'].get('assessed_level')}. Gaps: {', '.join(state['diagnostic'].get('knowledge_gaps', [])[:2])}."}
    ]
    return state


# ── Node 2: Content Generator Agent ──────────────────────────────────────────
def content_generator_agent_node(state: AgentState) -> AgentState:
    """
    Content Generator Agent Node: produces personalized study notes,
    explanations, and summaries tailored to the student's diagnostic profile.
    NEW AGENT — fulfills requirements for dynamic content generation.
    """
    model = get_gemini_model(temperature=0.6)
    if not model:
        state["study_notes"] = _fallback_study_notes(state)
        return state

    diagnostic = state.get("diagnostic", {})
    gaps       = ", ".join(diagnostic.get("knowledge_gaps", [state["subject"]]))
    difficulty = diagnostic.get("recommended_difficulty", "medium")
    level      = diagnostic.get("assessed_level", state["current_level"].lower())

    # Use conversation memory to personalize further
    memory_context = ""
    if state.get("messages"):
        last_msgs = state["messages"][-4:]  # last 4 messages for context
        memory_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_msgs])

    prompt_template = """
You are a Content Generator Agent for an AI-powered study platform.
Generate comprehensive, personalized study material for this student.
Return ONLY valid JSON — no markdown, no preamble.

Subject: {subject}
Student Level: {level}
Target: {target_level}
Knowledge Gaps to Address: {gaps}
Difficulty: {difficulty}
Recent Conversation Context:
{memory_context}

Return exactly this JSON:
{{
  "summary": "A 3-4 sentence executive summary of what the student will learn and why it matters.",
  "key_concepts": [
    {{
      "concept": "Concept name",
      "explanation": "Clear 2-3 sentence explanation suited to {level} level",
      "example": "A concrete, real-world example",
      "tip": "A memory aid or study tip"
    }}
  ],
  "study_notes": "Detailed markdown study notes (300-400 words) covering the core material for {subject} targeting the gaps: {gaps}.",
  "quick_reference": ["Key point 1", "Key point 2", "Key point 3", "Key point 4", "Key point 5"],
  "recommended_resources": ["Resource 1 with why it helps", "Resource 2", "Resource 3"]
}}
Produce at least 4 key_concepts. Tailor explanation depth strictly to {level} level.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    try:
        response = chain.invoke({
            "subject": state["subject"],
            "level": level,
            "target_level": state["target_level"],
            "gaps": gaps,
            "difficulty": difficulty,
            "memory_context": memory_context or "No prior context."
        })
        result = safe_json_parse(response.content)
        if isinstance(result, dict) and "summary" in result:
            state["study_notes"] = result
        else:
            state["study_notes"] = _fallback_study_notes(state)
    except Exception as e:
        state["study_notes"] = _fallback_study_notes(state)
        state["error"] = str(e)

    # Append to conversation memory
    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": f"Content Generator Agent: Generated study notes for {state['subject']}. Covered {len(state['study_notes'].get('key_concepts', []))} key concepts targeting gaps."}
    ]
    return state


def _fallback_study_notes(state: AgentState) -> dict:
    subject = state.get("subject", "the subject")
    level   = state.get("current_level", "Beginner").lower()
    return {
        "summary": f"This module covers the foundational concepts of {subject} tailored for {level} students. You will build a solid understanding of core principles and apply them through examples.",
        "key_concepts": [
            {"concept": f"Introduction to {subject}", "explanation": f"The basics of {subject} form the foundation for all advanced study.", "example": "Start with simple real-world analogies.", "tip": "Read and re-read until core ideas feel intuitive."},
            {"concept": "Core Principles", "explanation": "Every field has organizing principles that tie concepts together.", "example": "Identify 2-3 rules that explain 80% of outcomes.", "tip": "Write them out in your own words."}
        ],
        "study_notes": f"## {subject} Study Notes\n\nFocus on understanding the fundamentals before moving to advanced topics. Practice regularly and test yourself with quizzes.",
        "quick_reference": [f"Understand basics of {subject}", "Practice with examples", "Review key concepts daily", "Take notes while studying", "Test with quizzes"],
        "recommended_resources": [f"Official {subject} documentation", "YouTube tutorials for beginners", "Practice problem sets"]
    }


# ── Node 3: Quiz Agent ────────────────────────────────────────────────────────
def quiz_agent_node(state: AgentState) -> AgentState:
    """
    Quiz Agent Node: generates adaptive MCQ questions targeting diagnosed gaps.
    Uses adjusted_difficulty from the feedback loop if available.
    """
    model = get_gemini_model(temperature=0.6)
    if not model:
        state["quiz_questions"] = []
        return state

    diagnostic  = state.get("diagnostic", {})
    week_number = state.get("week_number", 1)
    plan        = state.get("plan", {})

    week_data  = plan.get("weeks", [{}] * week_number)[week_number - 1] if plan.get("weeks") else {}
    topics     = [t["name"] for t in week_data.get("topics", [])] if week_data else [state["subject"]]
    gaps       = ", ".join(diagnostic.get("knowledge_gaps", topics))

    # Feedback loop: use adjusted difficulty if available, else diagnostic recommendation
    difficulty = state.get("adjusted_difficulty") or diagnostic.get("recommended_difficulty", "medium")

    # Include memory context
    memory_context = ""
    if state.get("messages"):
        memory_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-3:]])

    prompt_template = """
You are a Quiz Agent for an AI-powered study platform.
Generate exactly 8 multiple-choice questions about the topics below.
Difficulty: {difficulty}. Specifically target these knowledge gaps: {gaps}.

Week focus: {focus_area}
Topics covered: {topics}
Subject: {subject}
Student conversation context:
{memory_context}

Return ONLY a valid JSON array — no markdown, no preamble:
[
  {{
    "question": "Question text here?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "B) ...",
    "topic": "Relevant topic name",
    "difficulty": "{difficulty}",
    "explanation": "Two-sentence explanation of why this is correct."
  }}
]

Rules: All 4 options must be plausible. correct_answer must exactly match one option string.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    try:
        response = chain.invoke({
            "subject": state["subject"],
            "focus_area": week_data.get("focus_area", state["subject"]),
            "topics": ", ".join(topics),
            "gaps": gaps,
            "difficulty": difficulty,
            "memory_context": memory_context or "No prior context."
        })
        questions = safe_json_parse(response.content)
        if isinstance(questions, list) and len(questions) >= 1 and "question" in questions[0]:
            state["quiz_questions"] = questions
        else:
            state["quiz_questions"] = []
    except Exception as e:
        state["quiz_questions"] = []
        state["error"] = str(e)

    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": f"Quiz Agent: Generated {len(state['quiz_questions'])} questions at {difficulty} difficulty for Week {week_number}."}
    ]
    return state


# ── Node 4: Progress Tracker Agent ───────────────────────────────────────────
def progress_tracker_agent_node(state: AgentState) -> AgentState:
    """
    Progress Tracker Agent Node: analyses quiz performance, produces a feedback
    report, AND feeds adjusted difficulty back to the Diagnostic Agent loop.
    """
    quiz_answers = state.get("quiz_answers", [])
    questions    = state.get("quiz_questions", [])
    diagnostic   = state.get("diagnostic", {})
    week_number  = state.get("week_number", 1)
    plan         = state.get("plan", {})

    correct_count = sum(1 for a in quiz_answers if a.get("correct", False))
    total         = len(questions) if questions else 1
    score_pct     = round((correct_count / total) * 100) if total > 0 else 0

    wrong_concepts = [questions[i].get("topic", "General")
                      for i, a in enumerate(quiz_answers)
                      if not a.get("correct", False) and i < len(questions)]
    right_concepts = [questions[i].get("topic", "General")
                      for i, a in enumerate(quiz_answers)
                      if a.get("correct", False) and i < len(questions)]

    # ── Feedback loop: adjust difficulty based on score ──────────────────────
    current_difficulty = diagnostic.get("recommended_difficulty", "medium")
    if score_pct >= 85:
        adjusted = "hard"
    elif score_pct >= 65:
        adjusted = "medium"
    else:
        adjusted = "easy"
    state["adjusted_difficulty"] = adjusted

    model = get_gemini_model(temperature=0.5)
    if not model:
        state["feedback_report"] = _fallback_feedback_report(plan, week_number, score_pct, right_concepts, wrong_concepts)
        return state

    # Include memory for personalized feedback
    memory_context = ""
    if state.get("messages"):
        memory_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-5:]])

    prompt_template = """
You are a Progress Tracker Agent. Write a detailed, encouraging feedback report in Markdown.

Student Quiz Results:
- Subject: {subject}
- Week {week_number} focus: {focus_area}
- Score: {correct}/{total} = {score_pct}%
- Concepts answered correctly: {right}
- Concepts answered incorrectly: {wrong}
- Original knowledge gaps (from Diagnostic Agent): {gaps}
- Previous difficulty: {prev_difficulty}
- Adjusted difficulty for next week: {adjusted_difficulty}
- Student conversation history context:
{memory_context}

Write a complete report with ALL these sections:

## 🎯 Performance Summary
## 💪 Strengths Identified
## 🔧 Areas Needing Improvement
## 📈 Difficulty Adjustment for Next Week
Explain WHY the difficulty is being adjusted from {prev_difficulty} to {adjusted_difficulty} based on the score.
## 🗺️ 3-Day Study Plan
## 💬 Motivational Message

Be specific, constructive, and thorough. Use bullet points. Minimum 300 words.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model

    try:
        week_data = plan.get("weeks", [{}] * week_number)[week_number - 1] if plan.get("weeks") else {}
        response = chain.invoke({
            "subject": state["subject"],
            "week_number": week_number,
            "focus_area": week_data.get("focus_area", state["subject"]),
            "correct": correct_count,
            "total": total,
            "score_pct": score_pct,
            "right": right_concepts or ["None"],
            "wrong": wrong_concepts or ["None"],
            "gaps": diagnostic.get("knowledge_gaps", []),
            "prev_difficulty": current_difficulty,
            "adjusted_difficulty": adjusted,
            "memory_context": memory_context or "No prior conversation context."
        })
        if response.content and len(response.content.strip()) > 100:
            state["feedback_report"] = response.content
        else:
            state["feedback_report"] = _fallback_feedback_report(plan, week_number, score_pct, right_concepts, wrong_concepts)
    except Exception as e:
        state["feedback_report"] = _fallback_feedback_report(plan, week_number, score_pct, right_concepts, wrong_concepts)
        state["error"] = str(e)

    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": f"Progress Tracker Agent: Score {score_pct}%. Difficulty adjusted from {current_difficulty} to {adjusted} for next week. {'Pass ✅' if score_pct >= 70 else 'Needs improvement ⚠️'}"}
    ]
    return state


def _fallback_feedback_report(plan, week_number, score_pct, right_concepts, wrong_concepts):
    grade = "Excellent! 🏆" if score_pct >= 80 else "Good work! 📈" if score_pct >= 60 else "Keep practising! 💪"
    subject = plan.get("subject", "your subject") if plan else "your subject"
    return f"""
## 🎯 Performance Summary
You scored **{score_pct}%** on Week {week_number} of your {subject} plan. {grade}

## 💪 Strengths Identified
{"".join(f"- ✅ **{c}**\n" for c in right_concepts) if right_concepts else "- Keep working — strengths will emerge!"}

## 🔧 Areas Needing Improvement
{"".join(f"- ❗ **{c}** — Review this concept and retry related questions.\n" for c in wrong_concepts) if wrong_concepts else "- No major gaps found. Excellent work!"}

## 📈 Difficulty Adjustment for Next Week
{"Difficulty increased — you're ready for a harder challenge!" if score_pct >= 85 else "Difficulty stays at medium — solid progress!" if score_pct >= 65 else "Difficulty decreased — consolidate before moving up."}

## 🗺️ 3-Day Study Plan
- **Day 1:** Re-read notes on {", ".join(wrong_concepts[:2]) if wrong_concepts else "this week's topics"}
- **Day 2:** Solve 5 practice problems on each weak concept
- **Day 3:** Retake the quiz and aim for {min(score_pct + 20, 100)}%+

## 💬 Keep Going!
Consistency beats perfection. Your score of **{score_pct}%** shows real effort — keep it up! 🚀
"""


# ─────────────────────────────────────────────
# LangGraph Workflow Builder
# ─────────────────────────────────────────────
def build_diagnostic_workflow() -> StateGraph:
    """
    Build a LangGraph workflow for the initial plan generation pipeline:
    Diagnostic Agent → Content Generator Agent
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("diagnostic_agent", diagnostic_agent_node)
    workflow.add_node("content_generator_agent", content_generator_agent_node)
    workflow.set_entry_point("diagnostic_agent")
    workflow.add_edge("diagnostic_agent", "content_generator_agent")
    workflow.add_edge("content_generator_agent", END)
    return workflow.compile()


def build_quiz_workflow() -> StateGraph:
    """
    Build a LangGraph workflow for the quiz pipeline:
    Quiz Agent → Progress Tracker Agent (with feedback loop back to difficulty)
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("quiz_agent", quiz_agent_node)
    workflow.add_node("progress_tracker_agent", progress_tracker_agent_node)
    workflow.set_entry_point("quiz_agent")
    workflow.add_edge("quiz_agent", END)
    return workflow.compile()


def build_feedback_workflow() -> StateGraph:
    """
    Build a LangGraph workflow for the post-quiz feedback loop:
    Progress Tracker Agent updates difficulty → feeds into next Quiz Agent run.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("progress_tracker_agent", progress_tracker_agent_node)
    workflow.set_entry_point("progress_tracker_agent")
    workflow.add_edge("progress_tracker_agent", END)
    return workflow.compile()


# ─────────────────────────────────────────────
# LangChain Memory Manager
# ─────────────────────────────────────────────
class SessionMemoryManager:
    """
    Manages LangChain-compatible conversation memory across sessions.
    Uses ChatMessageHistory (langchain_community) and persists to MongoDB.
    """
    def __init__(self, mongo: MongoDBClient, plan_id: str):
        self.mongo     = mongo
        self.plan_id   = plan_id
        self.chat_memory = ChatMessageHistory()
        self._load()

    def _load(self):
        """Restore persisted messages into LangChain ChatMessageHistory."""
        messages = self.mongo.load_memory(self.plan_id)
        for msg in messages:
            if msg.get("role") == "user":
                self.chat_memory.add_user_message(msg["content"])
            elif msg.get("role") in ("assistant", "ai"):
                self.chat_memory.add_ai_message(msg["content"])

    def add_user_message(self, content: str):
        self.chat_memory.add_user_message(content)
        self.mongo.append_memory_message(self.plan_id, "user", content)

    def add_ai_message(self, content: str):
        self.chat_memory.add_ai_message(content)
        self.mongo.append_memory_message(self.plan_id, "assistant", content)

    def get_history(self) -> list:
        """Return messages as plain dicts for AgentState."""
        result = []
        for msg in self.chat_memory.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result

    def get_summary(self) -> str:
        msgs = self.chat_memory.messages
        if not msgs:
            return "No conversation history yet."
        lines = []
        for msg in msgs[-6:]:
            role = "Student" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content[:120]}{'...' if len(msg.content) > 120 else ''}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Fallback helpers for diagnostic
# ─────────────────────────────────────────────
def diagnostic_agent(subject: str, topic: str, current_level: str,
                     target_level: str, hours_per_week: int,
                     prior_knowledge: str = "") -> dict:
    """Wrapper that runs the diagnostic agent via LangGraph."""
    initial_state: AgentState = {
        "subject": subject, "topic": topic,
        "current_level": current_level, "target_level": target_level,
        "hours_per_week": hours_per_week, "prior_knowledge": prior_knowledge,
        "diagnostic": {}, "study_notes": {}, "quiz_questions": [],
        "quiz_answers": [], "feedback_report": "", "adjusted_difficulty": "",
        "messages": [], "week_number": 1, "plan": {}, "error": ""
    }
    workflow = build_diagnostic_workflow()
    try:
        result = workflow.invoke(initial_state)
        return result.get("diagnostic", _fallback_diagnostic(subject, topic, current_level))
    except Exception:
        return _fallback_diagnostic(subject, topic, current_level)


def _fallback_diagnostic(subject, topic, current_level):
    level_map = {"Beginner": "beginner", "Intermediate": "intermediate", "Advanced": "advanced"}
    lvl = level_map.get(current_level, "beginner")
    return {
        "assessed_level": lvl,
        "confidence_score": 0.5,
        "knowledge_gaps": [
            f"Foundational concepts of {topic}",
            "Core principles and theory",
            "Practical application",
            "Advanced use cases"
        ],
        "strong_areas": [f"Basic awareness of {subject}"],
        "recommended_difficulty": "medium",
        "learning_path": [
            f"Step 1: Build foundational understanding of {topic}",
            "Step 2: Study core principles with worked examples",
            "Step 3: Solve practice problems at increasing difficulty",
            "Step 4: Apply knowledge to real-world scenarios"
        ],
        "focus_areas": [topic, subject],
        "estimated_weeks": 4,
        "diagnostic_summary": (
            f"Student is at {lvl} level in {subject}, aiming to master {topic}. "
            f"Recommend starting with fundamentals and progressively increasing difficulty."
        )
    }


def run_content_generator(subject, topic, current_level, target_level,
                          hours_per_week, diagnostic, prior_knowledge="",
                          memory_messages=None) -> dict:
    """Run only the Content Generator Agent via LangGraph."""
    initial_state: AgentState = {
        "subject": subject, "topic": topic,
        "current_level": current_level, "target_level": target_level,
        "hours_per_week": hours_per_week, "prior_knowledge": prior_knowledge,
        "diagnostic": diagnostic, "study_notes": {}, "quiz_questions": [],
        "quiz_answers": [], "feedback_report": "", "adjusted_difficulty": "",
        "messages": memory_messages or [], "week_number": 1, "plan": {}, "error": ""
    }
    # Single-node graph just for content generation
    workflow = StateGraph(AgentState)
    workflow.add_node("content_generator_agent", content_generator_agent_node)
    workflow.set_entry_point("content_generator_agent")
    workflow.add_edge("content_generator_agent", END)
    app = workflow.compile()
    try:
        result = app.invoke(initial_state)
        return result.get("study_notes", _fallback_study_notes(initial_state))
    except Exception:
        return _fallback_study_notes(initial_state)


def run_quiz_agent(plan, week_number, diagnostic, adjusted_difficulty="", memory_messages=None) -> list:
    """Run the Quiz Agent via LangGraph."""
    initial_state: AgentState = {
        "subject": plan.get("subject", ""), "topic": plan.get("subject", ""),
        "current_level": plan.get("current_level", "Beginner"),
        "target_level": plan.get("target_level", "Intermediate"),
        "hours_per_week": plan.get("hours_per_week", 10),
        "prior_knowledge": "", "diagnostic": diagnostic, "study_notes": {},
        "quiz_questions": [], "quiz_answers": [], "feedback_report": "",
        "adjusted_difficulty": adjusted_difficulty,
        "messages": memory_messages or [], "week_number": week_number,
        "plan": plan, "error": ""
    }
    workflow = build_quiz_workflow()
    try:
        result = workflow.invoke(initial_state)
        return result.get("quiz_questions", [])
    except Exception:
        return []


def run_progress_tracker(plan, week_number, quiz_answers, questions, diagnostic,
                         adjusted_difficulty="", memory_messages=None) -> tuple:
    """Run the Progress Tracker Agent via LangGraph. Returns (feedback_report, new_difficulty)."""
    initial_state: AgentState = {
        "subject": plan.get("subject", ""), "topic": plan.get("subject", ""),
        "current_level": plan.get("current_level", "Beginner"),
        "target_level": plan.get("target_level", "Intermediate"),
        "hours_per_week": plan.get("hours_per_week", 10),
        "prior_knowledge": "", "diagnostic": diagnostic, "study_notes": {},
        "quiz_questions": questions, "quiz_answers": quiz_answers,
        "feedback_report": "", "adjusted_difficulty": adjusted_difficulty,
        "messages": memory_messages or [], "week_number": week_number,
        "plan": plan, "error": ""
    }
    workflow = build_feedback_workflow()
    try:
        result = workflow.invoke(initial_state)
        return result.get("feedback_report", ""), result.get("adjusted_difficulty", "medium")
    except Exception:
        score_pct = round(sum(1 for a in quiz_answers if a.get("correct", False)) / max(len(questions), 1) * 100)
        right = [questions[i].get("topic") for i, a in enumerate(quiz_answers) if a.get("correct") and i < len(questions)]
        wrong = [questions[i].get("topic") for i, a in enumerate(quiz_answers) if not a.get("correct") and i < len(questions)]
        return _fallback_feedback_report(plan, week_number, score_pct, right, wrong), "medium"


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────
def display_diagnostic_report(diagnostic: dict):
    lvl_color = {"beginner": "#FF9800", "intermediate": "#1976D2", "advanced": "#4CAF50"}.get(
        diagnostic.get("assessed_level", "beginner"), "#FF9800"
    )
    gaps_html   = "".join(f"<span class='gap-tag'>{g}</span>" for g in diagnostic.get("knowledge_gaps", []))
    strong_html = "".join(
        f"<span class='gap-tag' style='background:#e8f5e9;color:#2E7D32;border-color:#4CAF50;'>{s}</span>"
        for s in diagnostic.get("strong_areas", [])
    )
    path_html = "".join(
        f"<div style='padding:4px 0;color:#555;font-size:0.85rem;'>{'⟶' if i else '▸'} {step}</div>"
        for i, step in enumerate(diagnostic.get("learning_path", []))
    )
    st.markdown(f"""
    <div class='diagnostic-card'>
      <div class='agent-badge badge-diagnostic'>🔍 Diagnostic Agent Report</div>
      <div style='display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:0.8rem;'>
        <div>
          <div style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:1px;'>Assessed Level</div>
          <div style='font-weight:700;font-size:1rem;color:{lvl_color};'>{diagnostic.get("assessed_level","").upper()}</div>
        </div>
        <div>
          <div style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:1px;'>Recommended Difficulty</div>
          <div style='font-weight:700;font-size:1rem;color:#1976D2;'>{diagnostic.get("recommended_difficulty","medium").upper()}</div>
        </div>
        <div>
          <div style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:1px;'>Estimated Weeks</div>
          <div style='font-weight:700;font-size:1rem;color:#555;'>{diagnostic.get("estimated_weeks",4)}</div>
        </div>
      </div>
      <div style='margin-bottom:0.5rem;'>
        <div style='font-size:0.75rem;color:#888;margin-bottom:4px;'>KNOWLEDGE GAPS TO ADDRESS</div>
        {gaps_html}
      </div>
      <div style='margin-bottom:0.5rem;'>
        <div style='font-size:0.75rem;color:#888;margin-bottom:4px;'>STRONG AREAS</div>
        {strong_html if strong_html else "<span style='color:#aaa;font-size:0.8rem;'>None identified yet</span>"}
      </div>
      <div style='margin-top:0.6rem;'>
        <div style='font-size:0.75rem;color:#888;margin-bottom:4px;'>PERSONALISED LEARNING PATH</div>
        {path_html}
      </div>
      <div style='margin-top:0.8rem;font-size:0.85rem;color:#555;border-top:1px solid #ffe082;padding-top:0.5rem;'>
        <strong>Summary:</strong> {diagnostic.get("diagnostic_summary","")}
      </div>
    </div>
    """, unsafe_allow_html=True)


def display_study_notes(notes: dict):
    """Render Content Generator Agent output using native Streamlit components."""
    if not notes:
        return

    # ── Header badge ──────────────────────────────────────────────────────────
    st.markdown(
        "<div class='agent-badge badge-content'>📖 Content Generator Agent — Study Notes</div>",
        unsafe_allow_html=True
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = notes.get("summary", "")
    if summary:
        st.info(summary)

    # ── Key Concepts — one native container per concept ───────────────────────
    key_concepts = notes.get("key_concepts", [])
    if key_concepts:
        st.markdown("#### 🔑 Key Concepts")
        for c in key_concepts:
            concept     = c.get("concept", "")
            explanation = c.get("explanation", "")
            example     = c.get("example", "")
            tip         = c.get("tip", "")
            with st.container():
                st.markdown(
                    f"<div style='background:#fff;border-radius:8px;border:1px solid #e1bee7;"
                    f"padding:0.8rem 1rem;margin-bottom:0.6rem;'>"
                    f"<span style='font-weight:700;color:#6A1B9A;font-size:0.97rem;'>📌 {concept}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(explanation)
                    if example:
                        st.markdown(f"**Example:** {example}")
                with col2:
                    if tip:
                        st.markdown(
                            f"<div style='background:#f3e5f5;border-radius:8px;padding:0.5rem 0.7rem;"
                            f"font-size:0.82rem;color:#7B1FA2;'>💡 {tip}</div>",
                            unsafe_allow_html=True
                        )
                st.divider()

    # ── Quick Reference ───────────────────────────────────────────────────────
    quick_ref = notes.get("quick_reference", [])
    if quick_ref:
        st.markdown("#### ⚡ Quick Reference")
        for pt in quick_ref:
            st.markdown(f"- {pt}")

    # ── Full study notes in expander ──────────────────────────────────────────
    full_notes = notes.get("study_notes", "")
    if full_notes:
        with st.expander("📄 Full Study Notes (Markdown)", expanded=False):
            st.markdown(full_notes)

    # ── Recommended resources ─────────────────────────────────────────────────
    resources = notes.get("recommended_resources", [])
    if resources:
        st.markdown("#### 📚 Recommended Resources")
        for r in resources:
            st.markdown(f"- {r}")


def display_memory_panel(memory_manager: SessionMemoryManager):
    """Display LangChain conversation memory."""
    summary = memory_manager.get_summary()
    st.markdown(f"""
    <div class='memory-card'>
      <div class='agent-badge' style='background:rgba(57,73,171,0.12);color:#283593;border:1px solid rgba(57,73,171,0.3);'>🧠 LangChain Session Memory</div>
      <pre style='font-size:0.78rem;color:#333;white-space:pre-wrap;margin:0.5rem 0 0;'>{summary}</pre>
    </div>
    """, unsafe_allow_html=True)


def display_langgraph_flow(active_node: str = ""):
    nodes = [
        ("🔍 Diagnostic", "diagnostic"),
        ("📖 Content Gen", "content"),
        ("🧩 Quiz", "quiz"),
        ("📊 Progress", "progress"),
    ]
    parts = []
    for label, key in nodes:
        color = "#00897B" if active_node == key else "#aaa"
        bg    = "#e0f2f1" if active_node == key else "#f5f5f5"
        parts.append(f"<span style='background:{bg};color:{color};border:1px solid {color};border-radius:20px;padding:3px 10px;font-size:0.78rem;font-weight:700;'>{label}</span>")
    arrow = "<span style='color:#aaa;font-size:0.9rem;margin:0 4px;'>→</span>"
    st.markdown(f"""
    <div class='langgraph-flow'>
      <div class='agent-badge badge-langgraph'>⚡ LangGraph Multi-Agent Workflow</div>
      <div style='margin-top:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;'>
        {arrow.join(parts)}
        <span style='color:#aaa;font-size:0.78rem;margin-left:8px;'>→ Feedback Loop ↩</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Existing Helper Functions (unchanged)
# ─────────────────────────────────────────────
def generate_study_plan(subject, current_level, target_level, hours_per_week, diagnostic: dict = None):
    model = get_gemini_model()
    if not model:
        return None

    parser = JsonOutputParser(pydantic_object=StudyPlan)

    gap_hint = ""
    if diagnostic:
        gaps = ", ".join(diagnostic.get("knowledge_gaps", []))
        path = " → ".join(diagnostic.get("learning_path", []))
        gap_hint = f"\nPrioritise these knowledge gaps: {gaps}\nFollow this learning path: {path}"

    prompt_template = """
    Create a detailed {weeks}-week study plan for {subject} from {current_level} to {target_level} level.
    Weekly study hours: {hours_per_week}h.{gap_hint}
    Structure the response as valid JSON with this exact format:
    {{
        "weeks": [
            {{
                "week_number": 1,
                "focus_area": "Introduction to X",
                "objectives": ["Objective 1", "Objective 2"],
                "topics": [
                    {{
                        "name": "Topic 1",
                        "hours": 2,
                        "description": "Learning fundamentals..."
                    }}
                ],
                "recommended_hours": 10
            }}
        ],
        "quiz_status": {{"week1": false, "week2": false, "week3": false, "week4": false}}
    }}
    Important: Only double quotes. No markdown. Proper JSON syntax. Close all brackets.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | parser

    try:
        response = chain.invoke({
            "subject": subject,
            "current_level": current_level,
            "target_level": target_level,
            "hours_per_week": hours_per_week,
            "weeks": 4,
            "gap_hint": gap_hint,
            "format_instructions": parser.get_format_instructions()
        })
        return response
    except Exception as e:
        st.error(f"Error generating plan: {str(e)}")
        return None


def find_youtube_video(query):
    results = YoutubeSearch(query, max_results=1).to_dict()
    if results:
        video = results[0]
        return {
            "id": video["id"],
            "title": video["title"],
            "url": f"https://youtube.com/watch?v={video['id']}",
            "thumbnail": video["thumbnails"][0]
        }
    return None


def generate_quiz_questions(plan, week_number, diagnostic=None, adjusted_difficulty="", memory_messages=None):
    """Generate quiz questions via LangGraph Quiz Agent."""
    return run_quiz_agent(plan, week_number, diagnostic or {}, adjusted_difficulty, memory_messages)


def generate_improvement_suggestions(plan, week_number, incorrect_questions):
    model = get_gemini_model()
    if not model:
        return None

    week_data    = plan['weeks'][week_number - 1]
    subject      = plan.get('subject', 'the subject')
    topic_issues = {}
    for q in incorrect_questions:
        topic = q.get('topic', 'General')
        if topic not in topic_issues:
            topic_issues[topic] = []
        topic_issues[topic].append(q['question'])

    if not topic_issues:
        return []

    prompt_template = """
    Student struggled in {subject} (Week {week_number}: {focus_area}) on these topics:
    {topic_issues}
    For each topic provide: area_for_improvement, learning_approach, youtube_query.
    Format as JSON:
    [{{"topic":"T","area_for_improvement":"...","learning_approach":"...","youtube_query":"..."}}]
    """
    topic_issues_formatted = "".join(
        f"Topic: {t}\n" + "".join(f"  - {q}\n" for q in qs)
        for t, qs in topic_issues.items()
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain  = prompt | model
    response = chain.invoke({
        "subject": subject, "week_number": week_number,
        "focus_area": week_data['focus_area'],
        "topic_issues": topic_issues_formatted
    })

    improvement_areas = json.loads(response.content.replace('```json', '').replace('```', ''))
    for area in improvement_areas:
        area['video'] = find_youtube_video(area['youtube_query'])
    return improvement_areas


def generate_project_ideas(plan, week_number):
    model = get_gemini_model()
    if not model:
        return None

    week_data = plan['weeks'][week_number - 1]
    subject   = plan.get('subject', 'the subject')
    topics    = [t['name'] for t in week_data['topics']]

    prompt_template = """
    Generate 3 project ideas for Week {week_number} of {subject}. Topics learned: {topics}. Focus: {focus_area}.
    Format as JSON:
    [{{"title":"...","description":"...","key_concepts":["c1","c2"],"difficulty":"Beginner","estimated_hours":5}}]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain  = prompt | model
    response = chain.invoke({
        "subject": subject, "week_number": week_number,
        "focus_area": week_data['focus_area'], "topics": ", ".join(topics)
    })
    return json.loads(response.content.replace('```json', '').replace('```', ''))


def check_weekly_goals(plan, current_week):
    if not plan or not plan.get('weeks') or current_week <= 0 or current_week > len(plan['weeks']):
        return True, []
    week_data = plan['weeks'][current_week - 1]
    incomplete = [t for t in week_data['topics']
                  if not plan.get('progress', {}).get(t['name'], False)]
    return len(incomplete) == 0, incomplete


def can_proceed_to_week(plan, week_number):
    if week_number == 1:
        return True
    if not plan or 'quiz_status' not in plan:
        return False
    return plan['quiz_status'].get(f'week{week_number - 1}', False)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

    mongo = None
    try:
        mongo = MongoDBClient()
    except ValueError as e:
        st.error(str(e))

    defaults = {
        'plan': None, 'progress': {}, 'editing': False, 'current_week': 1,
        'plans_list': mongo.get_all_plans() if mongo else [],
        'backlog': [], 'quiz_questions': [], 'quiz_answers': {},
        'quiz_submitted': False, 'quiz_score': 0, 'improvement_areas': [],
        'project_ideas': [], 'diagnostic_result': None,
        'feedback_report': "", 'agent_quiz_answers': [],
        'study_notes': None,               # Content Generator Agent output
        'adjusted_difficulty': "",         # Feedback loop difficulty
        'memory_manager': None,            # LangChain memory
        'langgraph_active_node': "",       # For UI display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.title("📚 StudyPath AI — Smart Self-Study Companion")
    st.markdown("---")

    # Always show LangGraph flow at top
    display_langgraph_flow(st.session_state.langgraph_active_node)

    tabs = st.tabs(["🚀 Create Plan", "📋 My Plans", "📝 Backlog", "🧩 Quiz Center", "📖 Study Notes", "⚙️ Settings"])
    tab1, tab2, tab3, tab4, tab5, tab6 = tabs

    # ── Settings ──
    with tab6:
        st.header("⚙️ API Configuration")
        google_api_key = st.text_input("Google API Key (for Gemini)", value=st.session_state.google_api_key, type="password")
        mongo_uri      = st.text_input("MongoDB Connection URI", value=os.getenv("MONGO_URI", ""), type="password")
        if st.button("Save Configuration"):
            os.environ["GOOGLE_API_KEY"] = google_api_key
            os.environ["MONGO_URI"]      = mongo_uri
            st.session_state.google_api_key = google_api_key
            try:
                mongo = MongoDBClient()
                st.session_state.plans_list = mongo.get_all_plans()
                st.success("Configuration saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Error connecting to MongoDB: {e}")

    # ── Tab 1: Create Plan ──
    with tab1:
        if not st.session_state.google_api_key:
            st.warning("Please configure your Google API Key in the Settings tab first.")
        elif not mongo:
            st.warning("Please configure your MongoDB connection in the Settings tab first.")
        else:
            with st.expander("🚀 Create Your Study Plan", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    subject    = st.text_input("📖 Subject/Topic", placeholder="e.g., Machine Learning")
                    prior_know = st.text_area("🧩 Your prior knowledge (optional)",
                                              placeholder="e.g., I know Python basics but have never studied ML algorithms.",
                                              height=100)
                with col2:
                    current_level = st.selectbox("📊 Your Current Level", ["Beginner", "Intermediate", "Advanced"])
                with col3:
                    target_level   = st.selectbox("🎯 Target Level", ["Intermediate", "Advanced", "Expert"])
                    hours_per_week = st.slider("⏰ Weekly Study Hours", 1, 40, 10)

                if st.button("✨ Generate Smart Plan"):
                    # ── Step 1: LangGraph Diagnostic Agent ──────────────────
                    st.session_state.langgraph_active_node = "diagnostic"
                    with st.spinner("🔍 LangGraph → Diagnostic Agent analysing your profile..."):
                        diagnostic = diagnostic_agent(
                            subject, subject, current_level, target_level,
                            hours_per_week, prior_know
                        )
                        st.session_state.diagnostic_result = diagnostic

                    st.success("✅ Diagnostic Agent complete!")
                    display_diagnostic_report(diagnostic)

                    # ── Step 2: LangGraph Content Generator Agent ────────────
                    st.session_state.langgraph_active_node = "content"
                    with st.spinner("📖 LangGraph → Content Generator Agent producing study notes..."):
                        notes = run_content_generator(
                            subject, subject, current_level, target_level,
                            hours_per_week, diagnostic, prior_know
                        )
                        st.session_state.study_notes = notes

                    st.success("✅ Content Generator Agent complete! View notes in the 📖 Study Notes tab.")

                    # ── Step 3: Generate Study Plan ──────────────────────────
                    with st.spinner("🧠 Generating your personalised study plan..."):
                        try:
                            plan = generate_study_plan(subject, current_level, target_level,
                                                       hours_per_week, diagnostic)
                            if plan:
                                for week in plan['weeks']:
                                    for topic in week['topics']:
                                        topic['video'] = find_youtube_video(f"{topic['name']} {subject}")

                                start_date = datetime.datetime.now()
                                plan['start_date'] = start_date
                                plan['end_date']   = start_date + timedelta(weeks=len(plan['weeks']))

                                if 'quiz_status' not in plan:
                                    plan['quiz_status'] = {f"week{i+1}": False for i in range(len(plan['weeks']))}

                                plan_id = mongo.save_plan({
                                    "subject": subject, "current_level": current_level,
                                    "target_level": target_level, "hours_per_week": hours_per_week,
                                    "start_date": start_date, "end_date": plan['end_date'],
                                    "weeks": plan['weeks'], "quiz_status": plan['quiz_status'],
                                    "created_at": datetime.datetime.now(), "progress": {},
                                    "diagnostic": diagnostic
                                })

                                # Save study notes
                                mongo.save_study_notes(str(plan_id), 1, notes)

                                # Initialise LangChain memory for this plan
                                mem = SessionMemoryManager(mongo, str(plan_id))
                                mem.add_user_message(f"I want to study {subject} from {current_level} to {target_level} level. {prior_know}")
                                mem.add_ai_message(f"Diagnostic complete. Assessed as {diagnostic.get('assessed_level')}. Study plan created with {len(plan['weeks'])} weeks.")
                                st.session_state.memory_manager = mem

                                st.session_state.plan = {**plan, "_id": plan_id}
                                st.session_state.current_week = 1
                                st.session_state.plans_list = mongo.get_all_plans()
                                st.session_state.langgraph_active_node = ""
                                st.success("🎉 Plan generated! Head to 'My Plans' or '📖 Study Notes' to get started.")
                        except Exception as e:
                            st.error(f"Error generating plan: {e}")

    # ── Tab 2: My Plans ──
    with tab2:
        if not mongo:
            st.warning("Please configure your MongoDB connection in the Settings tab first.")
        elif not st.session_state.plans_list:
            st.info("You don't have any study plans yet. Create one in the 'Create Plan' tab!")
        else:
            plan_options = {
                f"{p.get('subject','Unnamed')} ({p.get('current_level','?')} ➡️ {p.get('target_level','?')})": str(p['_id'])
                for p in st.session_state.plans_list
            }
            selected_plan_label = st.selectbox("Select a plan:", options=list(plan_options.keys()))

            if selected_plan_label:
                plan_id       = plan_options[selected_plan_label]
                selected_plan = mongo.get_plan(plan_id)
                st.session_state.plan     = selected_plan
                st.session_state.progress = selected_plan.get('progress', {})
                st.session_state.backlog  = mongo.get_backlog(plan_id)

                if 'diagnostic' in selected_plan:
                    st.session_state.diagnostic_result = selected_plan['diagnostic']

                # Load LangChain memory for selected plan
                if (st.session_state.memory_manager is None or
                        getattr(st.session_state.memory_manager, 'plan_id', '') != plan_id):
                    st.session_state.memory_manager = SessionMemoryManager(mongo, plan_id)

                # Load adjusted difficulty for current week
                if 'quiz_status' not in selected_plan:
                    selected_plan['quiz_status'] = {f"week{i+1}": False for i in range(len(selected_plan['weeks']))}
                    mongo.update_plan(plan_id, {"quiz_status": selected_plan['quiz_status']})

                if 'start_date' in selected_plan:
                    days_since   = (datetime.datetime.now() - selected_plan['start_date']).days
                    current_week = min(max(1, (days_since // 7) + 1), len(selected_plan['weeks']))
                    st.session_state.current_week = current_week

                st.markdown("---")
                current_week = st.session_state.current_week

                # Show Diagnostic Report
                if st.session_state.diagnostic_result:
                    with st.expander("🔍 View Diagnostic Agent Report", expanded=False):
                        display_diagnostic_report(st.session_state.diagnostic_result)

                # Show LangChain memory
                if st.session_state.memory_manager:
                    with st.expander("🧠 View Session Memory (LangChain)", expanded=False):
                        display_memory_panel(st.session_state.memory_manager)

                # Adjusted difficulty indicator
                adj_diff = mongo.get_adjusted_difficulty(plan_id, current_week)
                if adj_diff and adj_diff != "medium":
                    diff_color = {"easy": "#FF9800", "hard": "#4CAF50"}.get(adj_diff, "#1976D2")
                    st.markdown(f"""
                    <div style='background:#f5f5f5;border-left:4px solid {diff_color};border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.5rem;'>
                      <span style='font-size:0.8rem;color:#666;'>🔄 <strong>Feedback Loop Active:</strong> Difficulty adjusted to </span>
                      <strong style='color:{diff_color};'>{adj_diff.upper()}</strong>
                      <span style='font-size:0.75rem;color:#888;'> based on your last quiz score</span>
                    </div>""", unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.header(f"📅 Week {current_week} of {len(selected_plan['weeks'])}")
                with col2:
                    new_week = st.number_input("Change week:", min_value=1,
                                               max_value=len(selected_plan['weeks']),
                                               value=current_week, step=1)
                    if new_week != current_week:
                        st.session_state.current_week = new_week
                        current_week = new_week

                # Quiz Status
                quiz_status = selected_plan.get('quiz_status', {})
                st.subheader("📝 Weekly Quiz Status")
                quiz_cols = st.columns(len(selected_plan['weeks']))
                for i in range(len(selected_plan['weeks'])):
                    week_num  = i + 1
                    with quiz_cols[i]:
                        prev_done = True if week_num == 1 else quiz_status.get(f"week{week_num-1}", False)
                        if prev_done:
                            status     = quiz_status.get(f"week{week_num}", False)
                            new_status = st.checkbox(f"Week {week_num} Quiz", value=status, key=f"quiz_week_{week_num}")
                            if new_status != status:
                                quiz_status[f"week{week_num}"] = new_status
                                mongo.update_quiz_status(plan_id, week_num, new_status)
                                st.rerun()
                        else:
                            st.checkbox(f"Week {week_num} Quiz", value=False, disabled=True, key=f"quiz_week_{week_num}")
                            st.caption("⚠️ Complete previous quiz first")

                goals_completed, incomplete_topics = check_weekly_goals(selected_plan, current_week)
                if not goals_completed:
                    st.warning(f"⚠️ You have {len(incomplete_topics)} incomplete topics for week {current_week}!")
                    if st.button("Add incomplete topics to backlog"):
                        for topic in incomplete_topics:
                            mongo.add_to_backlog(plan_id, {
                                "week": current_week, "name": topic['name'],
                                "hours": topic['hours'], "description": topic['description'],
                                "video": topic.get('video')
                            })
                        st.session_state.backlog = mongo.get_backlog(plan_id)
                        st.success("Added to backlog!")
                else:
                    st.success(f"🎉 All goals for week {current_week} completed!")

                # Roadmap
                st.markdown("---")
                st.subheader("📅 Your Learning Roadmap")
                st.markdown('<div class="roadmap">', unsafe_allow_html=True)
                st.markdown('<div class="roadmap-connector"></div>', unsafe_allow_html=True)
                for i, week in enumerate(selected_plan['weeks']):
                    week_num  = i + 1
                    is_locked = not can_proceed_to_week(selected_plan, week_num)
                    lock_icon = "🔒" if is_locked and week_num > 1 else ""
                    style = "opacity:0.5;" if is_locked and week_num > 1 else ""
                    st.markdown(f'''
                    <div class="roadmap-step" style="{style}">
                      <h3>Week {week['week_number']}</h3>
                      <p><strong>{week['focus_area']}</strong></p>
                      <p>📅 {week['recommended_hours']} hours {lock_icon}</p>
                    </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Progress bar
                total_topics = sum(len(w['topics']) for w in selected_plan['weeks'])
                completed    = sum(1 for t in [t['name'] for w in selected_plan['weeks'] for t in w['topics']]
                                   if st.session_state.progress.get(t, False))
                progress     = completed / total_topics if total_topics > 0 else 0
                st.subheader(f"📊 Progress: {completed}/{total_topics} topics ({progress:.0%})")
                st.markdown(f'''
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{progress*100}%"></div>
                </div>''', unsafe_allow_html=True)

                edit_col1, edit_col2 = st.columns([5, 1])
                with edit_col2:
                    if st.button("✏️ Edit Plan" if not st.session_state.editing else "Cancel Editing"):
                        st.session_state.editing = not st.session_state.editing

                # Weekly content tabs
                st.markdown("---")
                st.subheader("📚 Weekly Content")
                week_tabs = st.tabs([f"Week {w+1}: {selected_plan['weeks'][w]['focus_area']}"
                                     for w in range(len(selected_plan['weeks']))])

                for i, week in enumerate(selected_plan['weeks']):
                    week_num  = i + 1
                    is_locked = not can_proceed_to_week(selected_plan, week_num)

                    with week_tabs[i]:
                        if is_locked and week_num > 1:
                            st.warning(f"⚠️ Week {week_num} is locked. Complete Week {week_num-1} quiz first.")

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Recommended Hours:** {week['recommended_hours']}h")
                            st.markdown("**Objectives:**")
                            for obj in week['objectives']:
                                st.markdown(f"- {obj}")

                        week_topics = [t['name'] for t in week['topics']]
                        week_done   = sum(st.session_state.progress.get(t, False) for t in week_topics)
                        with col2:
                            st.markdown(f"**Progress:** {week_done}/{len(week_topics)} topics")
                            if not is_locked:
                                if st.button("Take Weekly Quiz", key=f"take_quiz_{week_num}"):
                                    st.session_state.quiz_questions     = []
                                    st.session_state.quiz_answers       = {}
                                    st.session_state.agent_quiz_answers = []
                                    st.session_state.quiz_submitted     = False
                                    st.session_state.quiz_score         = 0
                                    st.session_state.feedback_report    = ""
                                    existing_quiz = mongo.get_quiz(plan_id, week_num)
                                    if existing_quiz:
                                        st.session_state.quiz_questions = existing_quiz['questions']
                                    else:
                                        adj_diff = mongo.get_adjusted_difficulty(plan_id, week_num)
                                        mem_msgs = st.session_state.memory_manager.get_history() if st.session_state.memory_manager else []
                                        with st.spinner("🧩 LangGraph → Quiz Agent generating questions..."):
                                            questions = generate_quiz_questions(
                                                selected_plan, week_num,
                                                st.session_state.diagnostic_result,
                                                adj_diff, mem_msgs
                                            )
                                            if questions:
                                                mongo.save_quiz({
                                                    "plan_id": ObjectId(plan_id),
                                                    "week": week_num, "questions": questions,
                                                    "created_at": datetime.datetime.now()
                                                })
                                                st.session_state.quiz_questions = questions
                                    st.session_state.current_week = week_num
                                    st.rerun()

                        st.markdown("### Topics")
                        topic_names  = [f"{t['name']} ({t['hours']}h)" for t in week['topics']]
                        selected_idx = st.selectbox("Select a topic:", range(len(topic_names)),
                                                    format_func=lambda i: topic_names[i],
                                                    key=f"topic_sel_{week_num}")

                        if selected_idx is not None:
                            topic = week['topics'][selected_idx]
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {topic['name']} ({topic['hours']}h)")
                                st.markdown(f"**Description:** {topic['description']}")

                                checked = st.checkbox("Mark as completed",
                                                      value=st.session_state.progress.get(topic['name'], False),
                                                      key=f"chk_{week_num}_{topic['name']}",
                                                      disabled=is_locked and week_num > 1)
                                if not is_locked and checked != st.session_state.progress.get(topic['name'], False):
                                    mongo.update_progress(selected_plan['_id'], topic['name'], checked)
                                    st.session_state.progress[topic['name']] = checked
                                    if st.session_state.memory_manager:
                                        st.session_state.memory_manager.add_user_message(f"Completed topic: {topic['name']}")
                                    st.rerun()

                                if not is_locked and st.button("Add to backlog", key=f"bl_{week_num}_{selected_idx}"):
                                    mongo.add_to_backlog(plan_id, {
                                        "week": current_week, "name": topic['name'],
                                        "hours": topic['hours'], "description": topic['description'],
                                        "video": topic.get('video')
                                    })
                                    st.session_state.backlog = mongo.get_backlog(plan_id)
                                    st.success(f"Added '{topic['name']}' to backlog!")

                            with col2:
                                if topic.get('video'):
                                    v = topic['video']
                                    st.markdown(f"""
                                    <div class="video-card">
                                        <a href="{v['url']}" target="_blank">
                                            <img src="{v['thumbnail']}" width="200">
                                            <p><strong>{v['title']}</strong></p>
                                        </a>
                                    </div>""", unsafe_allow_html=True)

                        if week_num == current_week and not is_locked:
                            st.markdown("---")
                            st.subheader("🚀 Project Ideas")
                            if st.button("Generate Project Ideas", key=f"proj_{week_num}"):
                                with st.spinner("Generating..."):
                                    projects = generate_project_ideas(selected_plan, week_num)
                                    if projects:
                                        st.session_state.project_ideas = projects
                            if st.session_state.project_ideas:
                                pcols = st.columns(min(3, len(st.session_state.project_ideas)))
                                for j, p in enumerate(st.session_state.project_ideas):
                                    with pcols[j % 3]:
                                        st.markdown(f"""
                                        <div class="project-card">
                                            <h4>{p['title']}</h4>
                                            <p>{p['description']}</p>
                                            <p>🔑 {", ".join(p['key_concepts'])}</p>
                                            <p>⚡ {p['difficulty']} · ⏱️ {p['estimated_hours']}h</p>
                                        </div>""", unsafe_allow_html=True)

    # ── Tab 3: Backlog ──
    with tab3:
        if not mongo:
            st.warning("Please configure your MongoDB connection in the Settings tab first.")
        elif not st.session_state.plan:
            st.info("Please select a plan first to view your backlog.")
        else:
            st.header("📝 Study Backlog")
            if not st.session_state.backlog:
                st.info("Your backlog is empty! You're all caught up.")
            else:
                show_completed = st.checkbox("Show completed items", value=True)
                filtered = [item for item in st.session_state.backlog
                            if show_completed or not item.get('completed', False)]
                for item in filtered:
                    td        = item['topic_data']
                    completed = item.get('completed', False)
                    st.markdown(f"""
                    <div class="backlog-item {'completed' if completed else ''}">
                        <h4>{'✅' if completed else '⏳'} {td['name']} (Week {td['week']})</h4>
                        <p>{td['description']}</p>
                        <p><small>Hours: {td['hours']}h</small></p>
                    </div>""", unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 5])
                    with col1:
                        new_status = st.checkbox("Completed", value=completed, key=f"bl_status_{item['_id']}")
                        if new_status != completed:
                            mongo.update_backlog_item(item['_id'], new_status)
                            st.session_state.backlog = mongo.get_backlog(str(st.session_state.plan['_id']))
                            st.rerun()
                    with col2:
                        if st.button("Remove", key=f"rm_{item['_id']}"):
                            mongo.remove_from_backlog(item['_id'])
                            st.session_state.backlog = mongo.get_backlog(str(st.session_state.plan['_id']))
                            st.rerun()
                    if td.get('video'):
                        v = td['video']
                        st.markdown(f"""
                        <div class="video-card">
                            <a href="{v['url']}" target="_blank">
                                <img src="{v['thumbnail']}" width="100%">
                                <p>{v['title']}</p>
                            </a>
                        </div>""", unsafe_allow_html=True)
                    st.markdown("---")

    # ── Tab 4: Quiz Center ──
    with tab4:
        if not mongo:
            st.warning("Please configure your MongoDB connection in the Settings tab first.")
        elif not st.session_state.plan:
            st.info("Please select a study plan first!")
        else:
            st.header("🧩 Quiz Center")
            st.markdown(
                "<div class='agent-badge badge-quiz'>🧩 Quiz Agent + 📊 Progress Tracker Agent (via LangGraph)</div>",
                unsafe_allow_html=True
            )

            plan_id = str(st.session_state.plan['_id'])

            if not st.session_state.quiz_questions:
                st.info("Select a week below to take a quiz. The Quiz Agent will generate questions targeting your specific knowledge gaps, at the difficulty level set by the feedback loop.")

                quiz_cols = st.columns(len(st.session_state.plan['weeks']))
                for i, week in enumerate(st.session_state.plan['weeks']):
                    week_num    = i + 1
                    is_unlocked = can_proceed_to_week(st.session_state.plan, week_num)
                    adj_diff    = mongo.get_adjusted_difficulty(plan_id, week_num)
                    with quiz_cols[i]:
                        if is_unlocked:
                            label = f"Week {week_num} Quiz"
                            if adj_diff and adj_diff != "medium":
                                label += f" ({adj_diff})"
                            if st.button(label, key=f"start_quiz_{week_num}"):
                                existing_quiz = mongo.get_quiz(plan_id, week_num)
                                if existing_quiz:
                                    st.session_state.quiz_questions = existing_quiz['questions']
                                else:
                                    mem_msgs = st.session_state.memory_manager.get_history() if st.session_state.memory_manager else []
                                    with st.spinner("🧩 LangGraph → Quiz Agent generating adaptive questions..."):
                                        questions = generate_quiz_questions(
                                            st.session_state.plan, week_num,
                                            st.session_state.diagnostic_result,
                                            adj_diff, mem_msgs
                                        )
                                        if questions:
                                            mongo.save_quiz({
                                                "plan_id": ObjectId(plan_id),
                                                "week": week_num, "questions": questions,
                                                "created_at": datetime.datetime.now()
                                            })
                                            st.session_state.quiz_questions = questions
                                st.session_state.current_week      = week_num
                                st.session_state.quiz_answers      = {}
                                st.session_state.agent_quiz_answers = []
                                st.session_state.quiz_submitted    = False
                                st.session_state.quiz_score        = 0
                                st.session_state.feedback_report   = ""
                                st.rerun()
                        else:
                            st.button(f"Week {week_num} Quiz", disabled=True, key=f"start_quiz_{week_num}")
                            st.caption("🔒 Locked")

            else:
                questions    = st.session_state.quiz_questions
                current_week = st.session_state.current_week
                week_data    = st.session_state.plan['weeks'][current_week - 1]

                st.subheader(f"Week {current_week} Quiz: {week_data['focus_area']}")

                if st.session_state.diagnostic_result:
                    diag     = st.session_state.diagnostic_result
                    adj_diff = st.session_state.adjusted_difficulty or diag.get("recommended_difficulty", "medium")
                    st.info(f"🔍 **Diagnostic Agent** set this quiz to **{adj_diff.upper()}** difficulty (feedback loop adjusted), targeting: {', '.join(diag.get('knowledge_gaps', [])[:3])}")

                if not st.session_state.quiz_submitted:
                    for i, question in enumerate(questions):
                        st.markdown(f"""
                        <div class="quiz-question">
                            <h3>Question {i+1} <small style='font-size:0.7rem;color:#888;'>({question.get('topic','')} · {question.get('difficulty','medium')})</small></h3>
                            <p>{question['question']}</p>
                        </div>""", unsafe_allow_html=True)

                        option_key   = f"quiz_{current_week}_q{i}"
                        selected_ans = st.radio("Choose:", options=question['options'], key=option_key)
                        st.session_state.quiz_answers[i] = selected_ans
                        st.markdown("---")

                    if st.button("📊 Submit Quiz & Get AI Feedback"):
                        correct_count = 0
                        agent_answers = []
                        incorrect_qs  = []

                        for i, question in enumerate(questions):
                            user_ans    = st.session_state.quiz_answers.get(i, "")
                            correct_ans = question.get('correct_answer', '')
                            is_correct  = user_ans.strip() == correct_ans.strip()
                            if is_correct:
                                correct_count += 1
                            else:
                                incorrect_qs.append(question)

                            agent_answers.append({
                                "question": question['question'],
                                "selected": user_ans,
                                "correct_answer": correct_ans,
                                "correct": is_correct,
                                "explanation": question.get('explanation', '')
                            })

                        st.session_state.agent_quiz_answers = agent_answers
                        score_pct = int(correct_count / len(questions) * 100)
                        passed    = score_pct >= 70

                        if passed:
                            mongo.update_quiz_status(plan_id, current_week, True)
                            st.session_state.plan['quiz_status'][f'week{current_week}'] = True

                        # ── LangGraph: Progress Tracker Agent + Feedback Loop ──
                        st.session_state.langgraph_active_node = "progress"
                        mem_msgs = st.session_state.memory_manager.get_history() if st.session_state.memory_manager else []
                        with st.spinner("📊 LangGraph → Progress Tracker Agent generating feedback + adjusting difficulty..."):
                            report, new_difficulty = run_progress_tracker(
                                st.session_state.plan, current_week,
                                agent_answers, questions,
                                st.session_state.diagnostic_result or {},
                                st.session_state.adjusted_difficulty or "",
                                mem_msgs
                            )
                            st.session_state.feedback_report    = report
                            st.session_state.adjusted_difficulty = new_difficulty

                        # Persist adjusted difficulty for next quiz
                        mongo.save_adjusted_difficulty(plan_id, current_week + 1, new_difficulty)

                        # Update LangChain memory
                        if st.session_state.memory_manager:
                            st.session_state.memory_manager.add_user_message(f"Completed Week {current_week} quiz with score {score_pct}%.")
                            st.session_state.memory_manager.add_ai_message(f"Progress Tracker: Score {score_pct}%. Difficulty adjusted to {new_difficulty} for next week. {'Passed!' if passed else 'Needs more practice.'}")

                        if incorrect_qs:
                            with st.spinner("Generating improvement suggestions..."):
                                improvements = generate_improvement_suggestions(
                                    st.session_state.plan, current_week, incorrect_qs
                                )
                                st.session_state.improvement_areas = improvements or []

                        st.session_state.langgraph_active_node = ""
                        st.session_state.quiz_submitted = True
                        st.session_state.quiz_score     = score_pct
                        st.rerun()

                else:
                    score  = st.session_state.quiz_score
                    passed = score >= 70
                    color  = "pass" if passed else "fail"

                    st.markdown(f"""
                    <div class="score-display {color}">{score}%</div>
                    <h3 style="text-align:center">{'🎉 Passed!' if passed else '😔 Not yet — keep going!'}</h3>
                    """, unsafe_allow_html=True)

                    # Difficulty feedback indicator
                    new_diff = st.session_state.adjusted_difficulty
                    if new_diff:
                        diff_emoji = {"easy": "⬇️", "medium": "➡️", "hard": "⬆️"}.get(new_diff, "➡️")
                        st.info(f"🔄 **Feedback Loop:** Difficulty for next quiz adjusted to **{new_diff.upper()}** {diff_emoji} based on this score.")

                    with st.expander("📋 Review Your Answers", expanded=False):
                        for i, q in enumerate(questions):
                            ans        = st.session_state.agent_quiz_answers[i] if i < len(st.session_state.agent_quiz_answers) else {}
                            is_correct = ans.get("correct", False)
                            st.markdown(f"""
                            <div class="quiz-question">
                                <h3>Q{i+1} {'✅' if is_correct else '❌'}</h3>
                                <p>{q['question']}</p>
                            </div>""", unsafe_allow_html=True)
                            for opt in q['options']:
                                if opt.strip() == q.get('correct_answer','').strip():
                                    st.markdown(f"<div class='quiz-option correct'>✅ {opt}</div>", unsafe_allow_html=True)
                                elif opt.strip() == ans.get('selected','').strip() and not is_correct:
                                    st.markdown(f"<div class='quiz-option incorrect'>❌ {opt}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div class='quiz-option'>{opt}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='quiz-explanation'><strong>Explanation:</strong> {q.get('explanation','')}</div>", unsafe_allow_html=True)
                            st.markdown("---")

                    if st.session_state.feedback_report:
                        st.markdown("<div class='agent-badge badge-progress'>📊 Progress Tracker Agent — Feedback Report</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='feedback-report'>{st.session_state.feedback_report}</div>", unsafe_allow_html=True)

                    if st.session_state.improvement_areas:
                        st.subheader("🔧 Targeted Improvement Resources")
                        for area in st.session_state.improvement_areas:
                            st.markdown(f"""
                            <div class="improvement-card">
                                <h3>{area['topic']}</h3>
                                <p><strong>Area to improve:</strong> {area['area_for_improvement']}</p>
                                <p><strong>Approach:</strong> {area['learning_approach']}</p>
                            </div>""", unsafe_allow_html=True)
                            if area.get('video'):
                                v = area['video']
                                st.markdown(f"""
                                <div class="video-card">
                                    <a href="{v['url']}" target="_blank">
                                        <img src="{v['thumbnail']}" width="100%">
                                        <p>{v['title']}</p>
                                    </a>
                                </div>""", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔄 Retake Quiz"):
                            st.session_state.quiz_answers       = {}
                            st.session_state.agent_quiz_answers = []
                            st.session_state.quiz_submitted     = False
                            st.session_state.quiz_score         = 0
                            st.session_state.feedback_report    = ""
                            st.session_state.improvement_areas  = []
                            st.rerun()
                    with col2:
                        if passed and current_week < len(st.session_state.plan['weeks']):
                            if st.button(f"➡️ Move to Week {current_week+1}"):
                                st.session_state.current_week       = current_week + 1
                                st.session_state.quiz_questions     = []
                                st.session_state.quiz_answers       = {}
                                st.session_state.agent_quiz_answers = []
                                st.session_state.quiz_submitted     = False
                                st.session_state.feedback_report    = ""
                                st.rerun()
                    with col3:
                        if st.button("Exit Quiz"):
                            st.session_state.quiz_questions     = []
                            st.session_state.quiz_answers       = {}
                            st.session_state.agent_quiz_answers = []
                            st.session_state.quiz_submitted     = False
                            st.session_state.quiz_score         = 0
                            st.session_state.feedback_report    = ""
                            st.session_state.improvement_areas  = []
                            st.rerun()

    # ── Tab 5: Study Notes (Content Generator Agent) ──
    with tab5:
        st.header("📖 Study Notes — Content Generator Agent")
        st.markdown(
            "<div class='agent-badge badge-content'>📖 Content Generator Agent (via LangGraph)</div>",
            unsafe_allow_html=True
        )

        if not mongo:
            st.warning("Please configure your MongoDB connection in the Settings tab first.")
        elif not st.session_state.plan:
            st.info("Please select or create a study plan first!")
        else:
            plan_id      = str(st.session_state.plan['_id'])
            current_week = st.session_state.current_week

            # Week selector for notes
            note_week = st.selectbox(
                "Select week for study notes:",
                options=list(range(1, len(st.session_state.plan['weeks']) + 1)),
                index=current_week - 1,
                format_func=lambda w: f"Week {w}: {st.session_state.plan['weeks'][w-1]['focus_area']}"
            )

            # Load saved notes for selected week
            saved_notes = mongo.get_study_notes(plan_id, note_week)
            if saved_notes:
                st.session_state.study_notes = saved_notes

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Regenerate Notes", key="regen_notes"):
                    st.session_state.study_notes = None

            if st.session_state.study_notes:
                display_study_notes(st.session_state.study_notes)
            else:
                st.info("No study notes generated yet for this week. Click below to generate.")
                if st.button("📖 Generate Study Notes with Content Agent"):
                    st.session_state.langgraph_active_node = "content"
                    mem_msgs = st.session_state.memory_manager.get_history() if st.session_state.memory_manager else []
                    with st.spinner("📖 LangGraph → Content Generator Agent producing personalised notes..."):
                        notes = run_content_generator(
                            st.session_state.plan.get("subject", ""),
                            st.session_state.plan.get("subject", ""),
                            st.session_state.plan.get("current_level", "Beginner"),
                            st.session_state.plan.get("target_level", "Intermediate"),
                            st.session_state.plan.get("hours_per_week", 10),
                            st.session_state.diagnostic_result or {},
                            prior_knowledge="",
                            memory_messages=mem_msgs
                        )
                        st.session_state.study_notes = notes
                        mongo.save_study_notes(plan_id, note_week, notes)
                        if st.session_state.memory_manager:
                            st.session_state.memory_manager.add_ai_message(
                                f"Content Generator Agent: Produced study notes for Week {note_week} of {st.session_state.plan.get('subject','')}."
                            )
                    st.session_state.langgraph_active_node = ""
                    st.rerun()

            # Chat with memory context
            st.markdown("---")
            st.subheader("💬 Ask About This Topic")
            st.caption("Uses LangChain session memory to give contextual answers.")
            user_question = st.text_input("Ask a question about your study material:", key="topic_question")
            if st.button("Ask", key="ask_topic") and user_question:
                model = get_gemini_model(temperature=0.5)
                if model and st.session_state.diagnostic_result:
                    mem_history = ""
                    if st.session_state.memory_manager:
                        mem_history = st.session_state.memory_manager.get_summary()
                    diag = st.session_state.diagnostic_result
                    prompt = ChatPromptTemplate.from_template("""
You are a helpful study tutor. Answer the student's question using their context.

Student profile: {level} level studying {subject}.
Knowledge gaps: {gaps}.
Prior conversation:
{history}

Student question: {question}

Provide a clear, educational answer tailored to their level.
""")
                    chain = prompt | model
                    try:
                        resp = chain.invoke({
                            "level": diag.get("assessed_level", "beginner"),
                            "subject": st.session_state.plan.get("subject", ""),
                            "gaps": ", ".join(diag.get("knowledge_gaps", [])),
                            "history": mem_history,
                            "question": user_question
                        })
                        st.markdown(f"**Answer:** {resp.content}")
                        if st.session_state.memory_manager:
                            st.session_state.memory_manager.add_user_message(user_question)
                            st.session_state.memory_manager.add_ai_message(resp.content[:300])
                    except Exception as e:
                        st.error(f"Could not answer: {e}")


if __name__ == "__main__":
    main()