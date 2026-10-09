import os
import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

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

