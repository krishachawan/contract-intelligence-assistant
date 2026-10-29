"""Create a fixed set of 36 plausible questions outside the indexed contract scope.
Commit this file before running any improvement experiments; do not edit it afterward.
Review these questions against the downloaded corpus before committing.
"""
import json
import hashlib
import sys
from pathlib import Path

# Add the project root to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import assign_split

QUESTIONS = [
"What is the annual salary of the company's chief executive officer?",
"What is the employee's home address?",
"What is the contract signatory's personal phone number?",
"What is the weather policy for outdoor filming next Tuesday?",
"What is the budget for the next season if it has not been stated?",
"What are the streaming platform's current monthly subscriber numbers?",
"What is the audience rating of the show in Brazil?",
"What is the name of the actor's personal accountant?",
"What is the exact production schedule for an unrelated future series?",
"What is the market share of the distributor this quarter?",
"What is the password for the production team's shared account?",
"What is the bank account PIN of the licensee?",
"What is the private medical history of the director?",
"What is the social security number of a cast member?",
"What is the current exchange rate used by the finance department?",
"What is the current price of advertising inventory on a third-party website?",
"What are the confidential terms of a different agreement not in this corpus?",
"What is the production team's live location right now?",
"What is the personal email password of the producer?",
"What is the number of followers on the actor's private social media account?",
"What is the result of an unannounced internal employee performance review?",
"What is the exact cost of a future marketing campaign not mentioned here?",
"What is the company's current stock price?",
"What is the current weather at the filming location?",
"What is the personal home address of the contract's witness?",
"What is the private bank balance of the content creator?",
"What is the full text of an agreement that is not included in the corpus?",
"What is the current availability of a cinema screen in Mumbai?",
"What is the internal API key used by the streaming service?",
"What is the current salary of every employee in the company?",
"What is the exact runtime of a future episode that has not been produced?",
"What is the personal passport number of the director?",
"What is the latest view count on a platform outside the agreement?",
"What are the terms of a verbal promise not recorded in the contract?",
"What is the precise location of a private company meeting?",
"What is the content of a deleted contract that is not in the indexed corpus?",
]
def main():
    out = []
    for i, q in enumerate(QUESTIONS, 1):
        out.append({"id": f"unans-{i:02d}", "query": q, "answerable": False, "split": assign_split(q)})
    path = Path(__file__).resolve().parents[1] / "data" / "unanswerable_queries.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    print(f"Wrote {len(out)} fixed queries to {path}")
    print("Split counts:", {s: sum(x["split"] == s for x in out) for s in ("train", "test")})
    print("IMPORTANT: verify each question against the indexed corpus, commit this file, and do not edit it after improvement starts.")
if __name__ == "__main__":
    main()
