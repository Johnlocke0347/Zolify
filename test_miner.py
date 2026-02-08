import requests

HUB_URL = "https://unsleepy-kyler-vyingly.ngrok-free.dev/submit"

data = {
    "miner_uid": "genesis_tester_01",
    "f1": 0.92,
    "zk_proof": "0x746573745f70726f6f66",
    "seed": 42
}

headers = {
    "ngrok-skip-browser-warning": "69420"
}

try:
    r = requests.post(HUB_URL, json=data, headers=headers)
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"Connection failed: {e}")