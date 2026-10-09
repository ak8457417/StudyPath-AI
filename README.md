# StudyPath AI — modular structure

The original `new.py` has been split into modules to make the code easier to maintain.
The original `main()` function and its UI flow are retained.

## Files
- `app.py` — Streamlit entry point and main UI flow.
- `styles.py` — original custom CSS.
- `models.py` — Pydantic `StudyPlan` and LangGraph `AgentState`.
- `database.py` — MongoDB persistence operations.
- `memory.py` — persistent LangChain conversation memory.
- `services.py` — Gemini integration, LangGraph agents/workflows, fallback logic, study-plan/quiz/project helpers.
- `ui_components.py` — reusable Streamlit display components.
- `.env.example` — placeholder environment variables only; never put real secrets here.
- `.gitignore` — excludes local secrets and Python environment/cache files.

## Run
From this folder, activate your existing virtual environment and run:

```bash
streamlit run app.py
```

## Environment
Create a local `.env` file (do not commit it) with:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
MONGO_URI=your_mongodb_connection_uri_here
```

The application still uses the same libraries and MongoDB collections as the original file.
No real API keys or credentials are included in this package.
