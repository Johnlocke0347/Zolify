import sys
import os
import requests
import json

HUB_URL = "http://127.0.0.1:8000/submit"
data = {
    "miner_uid": "genesis_tester_01",
    "f1": 0.92,
    "zk_proof": "0x746573745f70726f6f66",
    "seed": 42
}

headers = {
    "ngrok-skip-browser-warning": "69420",
    "Content-Type": "application/json"
}

def run_test():
    try:
        r = requests.post(HUB_URL, json=data, headers=headers, timeout=10)
        r.raise_for_status()
        print(f"Submission successful: {r.status_code}")
        return True
    except Exception as e:
        print(f"Submission failed: {e}")
        return False

if __name__ == "__main__":
    if run_test():
        sys.exit(0)
    else:
        sys.exit(1)
