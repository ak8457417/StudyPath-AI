import os
import json
import re
import streamlit as st
from youtube_search import YoutubeSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from models import StudyPlan, AgentState

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

