import requests
import json
import sys

HUB_URL = "https://unsleepy-kyler-vyingly.ngrok-free.dev/submit"

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
        
        response_data = r.json()
        print(f"Status: SUCCESS | Response: {response_data}")
        return True
    except Exception as e:
        print(f"Status: FAILED | Error: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    if not success:
        sys.exit(1)
