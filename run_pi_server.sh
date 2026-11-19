#!/bin/bash
################################################################################
# Quick Launcher for Raspberry Pi Server
################################################################################

cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run server
python3 raspberry_pi_server/pi_server_gui.py
