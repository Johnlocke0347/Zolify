import requests
import json
import hashlib
import time

def generate_challenge_salt():
    """Validator sends unique salt per challenge"""
    return f"zolify-subnet1-{int(time.time()//60):06d}-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"

def check_services():
    base_url = "http://base:8080/health"
    sentiment_url = "http://sentiment:8081/train"
    
    # Generate challenge
    challenge_salt = generate_challenge_salt()
    
    endpoints = [
        {"url": base_url, "expected": "active"},
        {"url": sentiment_url, "salt": challenge_salt, "expected": "model_loaded"}
    ]
    
    results = []
    for ep in endpoints:
        try:
            resp = requests.get(ep["url"], timeout=10)
            data = resp.json()
            healthy = ep["expected"] in str(data)
            results.append({"url": ep["url"], "healthy": healthy, "salt_used": challenge_salt if "salt" in ep else None})
        except Exception as e:
            results.append({"url": ep["url"], "healthy": False, "error": str(e)})
    
    overall = all(r["healthy"] for r in results)
    print(json.dumps({"services": results, "challenge_salt": challenge_salt, "overall": overall}))
    return overall

if __name__ == "__main__":
    check_services()
