#!/bin/bash
while true; do
  {
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 61\r\n\r\n{\"status\":\"active\",\"f1\":0.87,\"proof\":\"0x1a2b3c4d5e6f7890\"}"
  } | nc -lp 8080 -q 1
done
