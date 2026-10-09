import os
import datetime
from datetime import timedelta

import streamlit as st
from dotenv import load_dotenv
from bson.objectid import ObjectId
from langchain_core.prompts import ChatPromptTemplate

from styles import apply_styles
from database import MongoDBClient
from memory import SessionMemoryManager
from services import (
    can_proceed_to_week,
    check_weekly_goals,
    diagnostic_agent,
    find_youtube_video,
    generate_improvement_suggestions,
    generate_project_ideas,
    generate_quiz_questions,
    generate_study_plan,
    get_gemini_model,
    run_content_generator,
    run_progress_tracker,
)
from ui_components import (
    display_diagnostic_report,
    display_langgraph_flow,
    display_memory_panel,
    display_study_notes,
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="StudyPath AI", page_icon="📚", layout="wide")
apply_styles()

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
