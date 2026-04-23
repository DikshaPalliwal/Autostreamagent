# Autostreamagent
The AutoStream Social-to-Lead Agent is a conversational AI built on LangGraph that transforms social media interactions into qualified business leads. It classifies every user message into one of three intents — greeting, product inquiry, or high-intent purchase signal — and routes accordingly: casual messages get a warm response, product questions trigger a RAG pipeline that retrieves accurate pricing and policy answers from a local ChromaDB vector store, and high-intent signals activate a structured lead qualification flow that collects the user's name, email, and creator platform one field at a time before firing a lead capture tool. State is fully persisted across conversation turns using LangGraph's MemorySaver checkpointer, ensuring the agent remembers context from earlier in the conversation and never re-asks for information already provided. The result is an agent that behaves less like a chatbot and more like a trained sales assistant — grounded in accurate product knowledge, aware of where each user is in the funnel, and precise about when to execute backend actions.
# 🎬 AutoStream AI Agent

> A Conversational AI Agent built for **ServiceHive – Inflx Assignment**  
> Social-to-Lead Agentic Workflow using LangGraph + Gemini 1.5 Flash + RAG

---

## 📌 Project Overview

This agent represents **Aria**, the AI sales assistant for **AutoStream** – a fictional SaaS platform
providing automated video editing tools. Aria can:

- ✅ Greet users naturally
- ✅ Answer product & pricing questions using **RAG** (ChromaDB + sentence-transformers)
- ✅ Detect **high-intent users** ready to sign up
- ✅ Progressively collect lead info (name → email → platform)
- ✅ Execute a **mock lead capture tool** only after all details are collected
- ✅ Retain full conversation memory across 5–6+ turns via LangGraph state

---

## 🗂️ Project Structure

```
autostream-agent/
├── agent.py                  # Core LangGraph agent (CLI entry point) ⭐
├── rag_pipeline.py           # ChromaDB-based RAG setup + retrieval
├── tools.py                  # mock_lead_capture() tool
├── knowledge_base/
│   └── autostream_kb.md      # AutoStream product knowledge base
├── requirements.txt
├── .env                      # API key configuration
├── .env.example              # API key template
└── README.md
```

---

## ⚡ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB)
> on first run. This is a one-time download.

### 4. Set Up Your API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then add your **OpenAI API key** (get it at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)):

```
OPENAI_API_KEY=sk-proj-your_openai_api_key_here
```

### 5. Run the Agent (CLI Mode)

```bash
python agent.py
```

The agent will start in your terminal and wait for your input at the `You:` prompt. 🎬

**Example usage:**
```
You: Hello!
Aria: Hey! Welcome to AutoStream 🎬 — the AI-powered video editing platform...

You: What are your pricing plans?
Aria: Great question! AutoStream offers two plans...

You: I want to sign up
Aria: Wonderful! To get you started, what's your full name?
...
```

Type `exit` or `quit` to end the session.

---

## 💬 Example Conversation

```
You: Hi there!
Aria: Hey! Welcome to AutoStream 🎬 — the AI-powered video editing platform for content 
      creators. I'm Aria! How can I help you today?

You: What are your pricing plans?
Aria: Great question! AutoStream offers two plans:
      - **Basic** ($29/month): 10 videos/month, 720p resolution, standard support
      - **Pro** ($79/month): Unlimited videos, 4K resolution, AI captions, 24/7 support
      Would you like to get started?

You: That sounds great! I want to try the Pro plan for my YouTube channel.
Aria: Amazing choice! Pro is perfect for YouTube creators! 🎉 To get you set up, 
      could I get your full name?

You: Rishi Patel
Aria: Great to meet you, Rishi Patel! 😊 What's your email address?

You: rishi@example.com
Aria: Perfect! Last question — which platform do you primarily create content for?

You: YouTube

✅ LEAD CAPTURED SUCCESSFULLY
   Name     : Rishi Patel
   Email    : rishi@example.com
   Platform : YouTube

Aria: 🎉 You're all set, Rishi Patel! Our team will reach out to rishi@example.com 
      within 24 hours to help you get started. Welcome to AutoStream! 🚀
```

---

## 🏗️ Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because this workflow requires **deterministic, explicit state control** — the agent must move through defined stages (intent detection → RAG response → lead collection → tool execution) in a controlled, auditable way. LangGraph's `StateGraph` API makes these transitions transparent and debuggable, unlike AutoGen's more autonomous agent-to-agent model which can be unpredictable for structured workflows.

### How State is Managed

The `AgentState` TypedDict is the single source of truth. It stores: the full conversation `messages` list, the classified `intent`, collected lead fields (`lead_name`, `lead_email`, `lead_platform`), a `lead_captured` boolean, and an `awaiting_field` tracker. Every graph node receives this state, makes changes, and returns an updated copy. LangGraph merges the returned dict back into the live state, so all information persists across turns without any external database — the state object itself IS the memory. The `awaiting_field` field acts as a cursor, ensuring the lead collection sequence (name → email → platform) progresses correctly and is never skipped or triggered prematurely.

### RAG Pipeline

User queries are embedded using `sentence-transformers` (all-MiniLM-L6-v2) and matched against pre-loaded knowledge chunks in **ChromaDB** using cosine similarity. The top-3 most relevant chunks are injected as context into the LLM prompt, ensuring answers are grounded in the actual knowledge base and not hallucinated.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following integration approach would be used:

### Architecture

```
WhatsApp User
     │  sends message
     ▼
Meta WhatsApp Business API (or Twilio Sandbox)
     │  POST webhook event
     ▼
FastAPI Webhook Server  (/webhook endpoint)
     │  extracts phone number + message text
     ▼
Session Store (Redis or in-memory dict)
     │  loads AgentState keyed by phone number
     ▼
LangGraph Agent  (agent.invoke(state))
     │  returns updated state + assistant reply
     ▼
Session Store → saves updated state back
     │
WhatsApp API → sends reply message to user
```

### Implementation Steps

1. **Register a WhatsApp Business API** account via Meta or use [Twilio's WhatsApp Sandbox](https://www.twilio.com/en-us/whatsapp) for testing.

2. **Build a FastAPI webhook** that Meta will POST messages to:

```python
from fastapi import FastAPI, Request
from agent import build_agent, create_initial_state

app = FastAPI()
agent = build_agent()

# Simple in-memory session store keyed by phone number
# In production: use Redis for persistence
sessions: dict = {}

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()

    # Parse Meta's webhook payload
    phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text  = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    # Load or create session state per user
    state = sessions.get(phone, create_initial_state())
    state["messages"] = state["messages"] + [{"role": "user", "content": text}]

    # Run the agent
    result = agent.invoke(state)
    sessions[phone] = result  # Save updated state

    # Extract reply and send back via WhatsApp API
    reply = result["messages"][-1]["content"]
    send_whatsapp_message(phone, reply)  # Calls Meta's /messages API

    return {"status": "ok"}
```

3. **Expose the server** using [ngrok](https://ngrok.com/) for local testing, then deploy to a cloud provider (Railway, Render, AWS EC2) for production.

4. **For production state persistence**, replace the in-memory `sessions` dict with **Redis** (`redis-py`), so sessions survive server restarts and scale across multiple instances.

---

## 🛠️ Tech Stack

| Component         | Technology                          |
|-------------------|-------------------------------------|
| Language          | Python 3.9+                         |
| Agent Framework   | LangGraph (StateGraph)              |
| LLM               | OpenAI GPT-3.5 Turbo                |
| Embeddings        | sentence-transformers (MiniLM-L6)   |
| Vector Store      | ChromaDB (in-memory)                |
| Knowledge Base    | Markdown → chunked documents        |
| Runtime Mode      | CLI (Terminal)                      |

---

## 📋 Environment Variables

| Variable         | Description                              | Required |
|------------------|------------------------------------------|----------|
| `OPENAI_API_KEY` | OpenAI API key (get at platform.openai.com/api-keys) | ✅ Yes   |

---

## 📄 License

MIT License – built as part of the ServiceHive Inflx ML Intern Assignment.
