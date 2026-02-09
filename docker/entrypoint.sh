#!/bin/bash
python3 -m http.server 8080 --bind 0.0.0.0 &
python3 /app/hub/aggregator.py &

sleep 10

python3 /app/train.py &

tail -f /dev/null
