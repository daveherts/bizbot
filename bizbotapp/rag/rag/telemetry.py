import datetime
import json
import os

LOG_PATH = os.path.expanduser("~/bb/bizbotapp/logs/bizbot_queries.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_query(query: str, context: str, response: str):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "context_summary": summarize_context(context),
        "response": response
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def summarize_context(context: str, max_chars: int = 200) -> str:
    return context.replace("\n", " ")[:max_chars] + ("..." if len(context) > max_chars else "")
