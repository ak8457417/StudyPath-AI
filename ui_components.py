import streamlit as st
from memory import SessionMemoryManager

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

