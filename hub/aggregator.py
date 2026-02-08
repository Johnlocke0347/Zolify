from fastapi import FastAPI, HTTPException
import uvicorn
import json
import time
from pydantic import BaseModel

app = FastAPI(title="Zolify Genesis Aggregator")

leaderboard = {}

class ProofSubmission(BaseModel):
    miner_uid: str
    f1: float
    zk_proof: str
    seed: int

@app.post("/submit")
async def submit_work(submission: ProofSubmission):
    if submission.f1 < 0.8:
        return {"status": "rejected", "reason": "F1 score below threshold"}

    uid = submission.miner_uid
    if uid not in leaderboard:
        leaderboard[uid] = {"points": 0, "submissions": 0}
    
    leaderboard[uid]["points"] += (submission.f1 * 100)
    leaderboard[uid]["submissions"] += 1
    leaderboard[uid]["last_seen"] = time.time()

    print(f"Accepted proof from {uid} | F1: {submission.f1}")
    return {"status": "success", "total_points": leaderboard[uid]["points"]}

@app.get("/leaderboard")
async def get_leaderboard():
    return sorted(leaderboard.items(), key=lambda x: x[1]['points'], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
