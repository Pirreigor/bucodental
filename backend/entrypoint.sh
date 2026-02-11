#!/bin/sh
# Entry script: start uvicorn on $PORT (defaults to 8080)
if [ -z "$PORT" ]; then
  PORT=8080
fi

echo "Starting app on port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --proxy-headers
