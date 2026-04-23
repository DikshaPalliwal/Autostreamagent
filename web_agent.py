"""
web_agent.py
-----------
FastAPI web server for the AutoStream agent with HTML frontend.
"""

import os
import json
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agent import build_agent, create_initial_state

# Initialize FastAPI app
app = FastAPI(title="AutoStream AI Agent", description="Conversational AI sales assistant for AutoStream")

# Global agent state (in production, use proper session management)
agent_states: Dict[str, Any] = {}

# Initialize the agent
agent = build_agent()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 AutoStream AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 90%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 5px;
        }

        .header p {
            opacity: 0.9;
            font-size: 1rem;
        }

        .chat-container {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 20px;
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 70%;
            animation: fadeIn 0.3s ease-in;
        }

        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }

        .message.assistant {
            background: white;
            color: #333;
            margin-right: auto;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .message .sender {
            font-weight: bold;
            font-size: 0.9rem;
            margin-bottom: 5px;
            opacity: 0.8;
        }

        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 10px;
        }

        .input-container input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s ease;
        }

        .input-container input:focus {
            border-color: #667eea;
        }

        .input-container button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s ease;
        }

        .input-container button:hover {
            transform: translateY(-2px);
        }

        .input-container button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .status {
            text-align: center;
            padding: 10px;
            color: #666;
            font-size: 0.9rem;
        }

        .typing {
            display: none;
            text-align: center;
            padding: 10px;
            color: #666;
            font-style: italic;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .lead-captured {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            text-align: center;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            animation: slideIn 0.5s ease-out;
        }

        @keyframes slideIn {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 AutoStream AI Assistant</h1>
            <p>Your AI sales assistant for automated video editing</p>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="sender">Aria (AutoStream AI)</div>
                <div>Hey there! 👋 Welcome to AutoStream — the AI-powered video editing platform for content creators. I'm Aria, your AI assistant. How can I help you today?</div>
            </div>
        </div>

        <div class="status" id="status">Ready to chat!</div>
        <div class="typing" id="typing">Aria is typing...</div>

        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message here..." autocomplete="off">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let sessionId = 'web_' + Date.now();
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const status = document.getElementById('status');
        const typing = document.getElementById('typing');

        // Auto-focus input
        messageInput.focus();

        // Send message on Enter key
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message to chat
            addMessage('user', 'You', message);
            messageInput.value = '';

            // Show typing indicator
            typing.style.display = 'block';
            status.style.display = 'none';
            sendButton.disabled = true;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                // Hide typing indicator
                typing.style.display = 'none';
                status.style.display = 'block';

                // Add assistant response
                addMessage('assistant', 'Aria (AutoStream AI)', data.response);

                // Check for lead capture
                if (data.lead_captured) {
                    showLeadCaptured(data.lead_info);
                }

            } catch (error) {
                console.error('Error:', error);
                typing.style.display = 'none';
                status.style.display = 'block';
                addMessage('assistant', 'System', 'Sorry, I encountered an error. Please try again.');
            }

            sendButton.disabled = false;
            messageInput.focus();
        }

        function addMessage(type, sender, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.innerHTML = `
                <div class="sender">${sender}</div>
                <div>${content.replace(/\n/g, '<br>')}</div>
            `;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showLeadCaptured(leadInfo) {
            const leadDiv = document.createElement('div');
            leadDiv.className = 'lead-captured';
            leadDiv.innerHTML = `
                <strong>✅ LEAD CAPTURED SUCCESSFULLY</strong><br>
                Name: ${leadInfo.name}<br>
                Email: ${leadInfo.email}<br>
                Platform: ${leadInfo.platform}
            `;
            chatContainer.appendChild(leadDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Initialize session
        fetch('/init_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/init_session")
async def init_session(request: ChatRequest):
    """Initialize a new chat session"""
    agent_states[request.session_id] = create_initial_state()
    return {"status": "initialized"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat messages"""
    try:
        # Get or create session state
        if request.session_id not in agent_states:
            agent_states[request.session_id] = create_initial_state()

        state = agent_states[request.session_id]

        # Add user message
        state["messages"] = state["messages"] + [
            {"role": "user", "content": request.message}
        ]

        # Run agent
        result = agent.invoke(state)

        # Update session state
        agent_states[request.session_id] = result

        # Get the latest assistant message
        assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
        latest_response = assistant_messages[-1]["content"] if assistant_messages else "I apologize, but I couldn't generate a response."

        # Check if lead was captured
        lead_captured = result.get("lead_captured", False)
        lead_info = None
        if lead_captured:
            lead_info = {
                "name": result.get("lead_name"),
                "email": result.get("lead_email"),
                "platform": result.get("lead_platform")
            }

        return {
            "response": latest_response,
            "lead_captured": lead_captured,
            "lead_info": lead_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "AutoStream AI Assistant"}

if __name__ == "__main__":
    import uvicorn
    print("🎬 Starting AutoStream AI Agent Web Server...")
    print("📱 Open your browser to: http://localhost:3000")
    print("❌ To stop the server, press Ctrl+C")
    uvicorn.run(app, host="127.0.0.1", port=3000)