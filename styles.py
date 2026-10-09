import streamlit as st

def apply_styles():
    """Apply the original StudyPath AI custom CSS."""
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
