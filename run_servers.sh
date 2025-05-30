#!/bin/bash

# Start MCP server in the background
echo "Starting MCP server..."
python -m app.mcp_server &
MCP_PID=$!

# Wait for MCP server to start
sleep 2

# Start Streamlit app
echo "Starting Streamlit app..."
streamlit run app/main.py

# When Streamlit is closed, also stop the MCP server
kill $MCP_PID 