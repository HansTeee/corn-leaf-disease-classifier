#!/bin/sh
# Start the API, wait until it has loaded the model, then run the UI in the foreground.
set -e
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

for i in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
        echo "API ready"; break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then echo "API failed to start" >&2; exit 1; fi
    sleep 1
done

exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
