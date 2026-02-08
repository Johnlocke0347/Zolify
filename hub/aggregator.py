from fastapi import FastAPI, HTTPException
import uvicorn
import time
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI(title="Zolify Genesis Aggregator")

HUB_URL = "https://unsleepy-kyler-vyingly.ngrok-free.dev"
leaderboard = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/leaderboard", response_class=HTMLResponse)
async def get_web_leaderboard():
    sorted_scores = sorted(leaderboard.items(), key=lambda x: x[1]['points'], reverse=True)
    
    rows = "".join([
        f"<tr><td>{uid}</td><td>{data['points']:.2f}</td><td>{data['submissions']}</td></tr>" 
        for uid, data in sorted_scores
    ])
    
    return f"""
    <html>
        <head><title>Zolify Genesis Leaderboard</title></head>
        <body style="font-family: sans-serif; background: #121212; color: white; padding: 40px;">
            <h1> Zolify Miner Leaderboard</h1>
            <table border="1" style="width: 100%; text-align: left; border-collapse: collapse;">
                <tr style="background: #333;">
                    <th style="padding:10px;">Miner UID</th>
                    <th style="padding:10px;">Total Points</th>
                    <th style="padding:10px;">Submissions</th>
                </tr>
                {rows}
            </table>
            <p style="color: #888; margin-top: 20px;">Live Hub: {HUB_URL}</p>
        </body>
    </html>
    """

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

@app.get("/api/leaderboard")
async def get_leaderboard_json():
    return sorted(leaderboard.items(), key=lambda x: x[1]['points'], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
