#!/bin/bash
python3 -m http.server 8080 --bind 0.0.0.0 &

python3 /app/test_miner.py || echo "Miner script crashed, keeping container alive for debugging"

wait
