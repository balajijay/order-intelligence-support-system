#!/bin/bash

# Auto-locate and activate the project environment safely
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment 'venv' not found. Please run ./setup.sh first."
    exit 1
fi

echo "🚀 Compiling state graph engine and executing routing checks..."
python3 agent.py
