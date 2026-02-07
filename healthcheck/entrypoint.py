import requests
import time
import json

def check_services():
    endpoints = [
        "http://localhost:8080/health",  # base container
        "http://localhost:8081/train"   # imdb model
    ]
    
    results = []
    for endpoint in endpoints:
        try:
            resp = requests.get(endpoint, timeout=5)
            results.append({"url": endpoint, "status": resp.status_code, "healthy": resp.status_code == 200})
        except:
            results.append({"url": endpoint, "status": "timeout", "healthy": False})
    
    print(json.dumps({"services": results, "overall": all(r["healthy"] for r in results)}))
    return all(r["healthy"] for r in results)

if __name__ == "__main__":
    check_services()
