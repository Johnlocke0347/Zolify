#!/bin/bash
python3 -m http.server 8080 --bind 0.0.0.0 &
sleep 2
python3 /app/test_miner.py
