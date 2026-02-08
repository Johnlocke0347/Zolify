#!/bin/bash
while true; do
  echo "HTTP/1.1 200 OK"
  echo "Content-Type: application/json"
  echo
  echo '{"status":"active","f1":0.87,"proof":"0x1a2b3c4d5e6f7890"}'
  sleep 0.1
done
