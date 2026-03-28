#!/bin/bash
# Launch the Graph-Regularised Vol Marking application
# Backend: FastAPI on port 8000
# Frontend: Vite dev server on port 5173 (proxies /api to backend)

set -e

echo "=== Graph-Regularised Vol Marking ==="
echo ""

# Start backend
echo "Starting backend on http://127.0.0.1:8000 ..."
cd "$(dirname "$0")"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on http://localhost:5173 ..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Application running:"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://127.0.0.1:8080"
echo "  API docs: http://127.0.0.1:8080/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
