"""
agent.py
--------
Core LangGraph-based conversational AI agent for AutoStream.

Architecture:
  ┌─────────────┐
  │  User Input │
  └──────┬──────┘
         │
  ┌──────▼──────────┐
  │ classify_intent │  ← LLM classifies: greeting / inquiry / high_intent
  └──────┬──────────┘
         │
  ┌──────▼──────────────────────────────────────┐
  │            route_intent (conditional)        │
  └───┬───────────────┬────────────────────┬─────┘
      │               │                    │
┌─────▼──────┐ ┌──────▼───────┐  ┌────────▼────────────┐
│handle_greet│ │handle_inquiry│  │handle_lead_collection│
│            │ │  (RAG used)  │  │  (3-step: name →     │
│            │ │              │  │   email → platform)  │
└─────┬──────┘ └──────┬───────┘  └────────┬────────────┘
      │               │                    │
      └───────────────┴────────────────────┘
                      │
                    [END]
"""

import os
import re
from typing import TypedDict, Optional, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from rag_pipeline import initialize_rag, retrieve_context
from tools import mock_lead_capture

load_dotenv()  # Load GOOGLE_API_KEY from .env file


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The full state of the conversation at any point in time.
    LangGraph passes this dict between nodes and accumulates updates.
    """
    messages: list[dict]          # Full conversation history: [{role, content}, ...]
    intent: str                    # "greeting" | "inquiry" | "high_intent" | "unknown"
    lead_name: Optional[str]       # Collected lead name
    lead_email: Optional[str]      # Collected lead email
    lead_platform: Optional[str]   # Collected content platform
    lead_captured: bool            # Whether mock_lead_capture() was called
    awaiting_field: Optional[str]  # "name" | "email" | "platform" | None


# ---------------------------------------------------------------------------
# LLM + RAG Initialization
# ---------------------------------------------------------------------------

def get_llm() -> ChatOpenAI:
    """Initialize GPT-3.5 Turbo. Raises if API key is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Please add it to your .env file.\n"
            "Get a key at: https://platform.openai.com/api-keys"
        )
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,         # Low temperature = more deterministic, factual answers
        openai_api_key=api_key,
    )


# Global singletons (initialized once)
llm = get_llm()
kb_collection = initialize_rag()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Aria, an AI sales assistant for AutoStream — a SaaS platform
that provides automated video editing tools for content creators.

Your personality: Friendly, concise, and helpful. Never make up information.

Core responsibilities:
1. Answer product and pricing questions using ONLY the context provided to you.
2. Warmly greet users and introduce AutoStream briefly when appropriate.
3. Identify when users show genuine intent to sign up or try the product.
4. Never ask for personal details unless the user has clearly shown purchase intent.

AutoStream Plans Summary (for reference only — always use context provided):
- Basic: $29/month | 10 videos | 720p | No AI captions | Standard support
- Pro:   $79/month | Unlimited | 4K  | AI captions    | 24/7 support
- No refunds after 7 days. 24/7 support on Pro only.
"""


# ---------------------------------------------------------------------------
# Helper: Format conversation history for prompt injection
# ---------------------------------------------------------------------------

def _format_history(messages: list[dict], max_turns: int = 6) -> str:
    """Returns last N messages as a readable string for the LLM prompt."""
    if not messages:
        return "(No prior messages)"
    recent = messages[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Aria (AutoStream AI)"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    """
    Classify the latest user message into one of three intent categories.
    If we're mid-way through lead collection, we skip re-classification
    to avoid interrupting the collection flow.
    """
    # Don't re-classify if we're already collecting lead fields
    if state.get("awaiting_field"):
        return state

    last_msg = state["messages"][-1]["content"]

    classification_prompt = f"""Classify the following user message into EXACTLY ONE of these intent categories:

- "greeting"    → Simple hello, hi, how are you, general small talk
- "inquiry"     → Asking about product features, pricing, plans, or company policies
- "high_intent" → User explicitly wants to sign up, purchase, start a trial, or use the product

User message: "{last_msg}"

Rules:
- Reply with ONLY the intent word in lowercase. No punctuation. No explanation.
- If unsure between inquiry and high_intent, pick high_intent if the user mentions
  wanting to "try", "start", "sign up", "subscribe", "get", "buy", or names a specific plan.
"""

    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        raw = response.content.strip().lower().strip('"').strip("'")
    except Exception as e:
        print(f"Error in classify_intent: {e}")
        raw = "inquiry"  # fallback

    # Sanitize: ensure it's one of the valid values
    intent = raw if raw in ("greeting", "inquiry", "high_intent") else "inquiry"

    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Conditional Router
# ---------------------------------------------------------------------------

def route_intent(
    state: AgentState,
) -> Literal["handle_greeting", "handle_inquiry", "handle_lead_collection"]:
    """
    Determine which handler node to call based on current state.
    Priority order:
    1. If mid-collection (awaiting_field is set) → lead_collection
    2. If intent is high_intent and lead not yet captured → lead_collection
    3. If intent is greeting → greeting
    4. Otherwise → inquiry (RAG)
    """
    if state.get("awaiting_field"):
        return "handle_lead_collection"

    if state.get("intent") == "high_intent" and not state.get("lead_captured"):
        return "handle_lead_collection"

    if state.get("intent") == "greeting":
        return "handle_greeting"

    return "handle_inquiry"


# ---------------------------------------------------------------------------
# Node 2: handle_greeting
# ---------------------------------------------------------------------------

def handle_greeting(state: AgentState) -> AgentState:
    """Handle simple greetings with a warm, brief response."""
    last_msg = state["messages"][-1]["content"]
    history = _format_history(state["messages"][:-1])

    prompt = f"""{SYSTEM_PROMPT}

Conversation so far:
{history}

User: {last_msg}

Respond with a friendly greeting. Briefly mention that AutoStream helps content creators
edit videos faster with AI. Keep it to 2–3 sentences. Don't ask for personal details."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reply = response.content
    except Exception as e:
        print(f"Error in handle_greeting: {e}")
        reply = "Hello! Welcome to AutoStream, where AI helps content creators edit videos faster."

    new_msgs = state["messages"] + [{"role": "assistant", "content": reply}]
    return {**state, "messages": new_msgs}


# ---------------------------------------------------------------------------
# Node 3: handle_inquiry (RAG-powered)
# ---------------------------------------------------------------------------

def handle_inquiry(state: AgentState) -> AgentState:
    """
    Answer product/pricing questions using RAG.
    The user's question is used to retrieve relevant KB chunks,
    which are injected into the LLM prompt as context.
    """
    last_msg = state["messages"][-1]["content"]
    history = _format_history(state["messages"][:-1])

    # RAG: retrieve top-3 relevant knowledge chunks
    context = retrieve_context(last_msg, kb_collection, n_results=3)

    prompt = f"""{SYSTEM_PROMPT}

Relevant knowledge base context (use this to answer accurately):
{context}

Conversation so far:
{history}

User: {last_msg}

Instructions:
- Answer using ONLY information from the knowledge base context above.
- Be accurate, concise, and friendly.
- If the context doesn't cover the question, say "I don't have that info handy — 
  you can reach us at support@autostream.io."
- If the user seems interested after your answer, end with a gentle invite like
  "Would you like to get started with AutoStream?" — but only if natural."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reply = response.content
    except Exception as e:
        print(f"Error in handle_inquiry: {e}")
        reply = "I'm sorry, I'm having trouble processing your question right now. Please try again later."

    new_msgs = state["messages"] + [{"role": "assistant", "content": reply}]
    return {**state, "messages": new_msgs}


# ---------------------------------------------------------------------------
# Node 4: handle_lead_collection
# ---------------------------------------------------------------------------

def handle_lead_collection(state: AgentState) -> AgentState:
    """
    Progressively collect lead information across multiple turns.

    Turn flow:
    1. First detection of high_intent → Acknowledge + ask for Name
    2. Name received → ask for Email
    3. Email received → ask for Platform
    4. Platform received → call mock_lead_capture() + confirm

    The awaiting_field value tracks where we are in the collection sequence.
    """
    last_msg = state["messages"][-1]["content"].strip()
    awaiting = state.get("awaiting_field")

    # --- Step 1: Just detected high intent, start collection ---
    if awaiting is None:
        history = _format_history(state["messages"][:-1])

        # Use LLM to generate a warm acknowledgment
        prompt = f"""{SYSTEM_PROMPT}

Conversation so far:
{history}

User: {last_msg}

The user has shown clear intent to sign up or try AutoStream. 
Acknowledge their interest warmly and naturally, then ask for their full name
to get them started. Keep it to 2 sentences max."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            reply = response.content
        except Exception as e:
            print(f"Error in handle_lead_collection: {e}")
            reply = "I'm excited to help you get started with AutoStream! What's your full name?"

        new_state = {
            **state,
            "intent": "high_intent",
            "awaiting_field": "name",
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        }
        return new_state

    # --- Step 2: Collect Name → ask for Email ---
    elif awaiting == "name":
        name = _extract_name(last_msg)
        reply = f"Great to meet you, {name}! 😊 What's your email address?"

        new_state = {
            **state,
            "lead_name": name,
            "awaiting_field": "email",
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        }
        return new_state

    # --- Step 3: Collect Email → ask for Platform ---
    elif awaiting == "email":
        email = last_msg.strip().lower()

        # Basic email format validation
        if not _is_valid_email(email):
            reply = "Hmm, that doesn't look like a valid email. Could you double-check it?"
            new_state = {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": reply}],
            }
            return new_state

        reply = (
            f"Perfect! Last question — which platform do you primarily create content for? "
            f"(e.g., YouTube, Instagram, TikTok, Facebook)"
        )
        new_state = {
            **state,
            "lead_email": email,
            "awaiting_field": "platform",
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        }
        return new_state

    # --- Step 4: Collect Platform → call tool + confirm ---
    elif awaiting == "platform":
        platform = last_msg.strip()

        # 🔧 TOOL EXECUTION: Only called here, after all 3 fields are collected
        result = mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=platform,
        )

        reply = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"We've noted your interest in AutoStream and our team will reach out to "
            f"{state['lead_email']} within 24 hours to help you get started. "
            f"Welcome to the AutoStream creator community! 🚀"
        )

        new_state = {
            **state,
            "lead_platform": platform,
            "lead_captured": True,
            "awaiting_field": None,
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        }
        return new_state

    # Fallback (should never reach here)
    return state


# ---------------------------------------------------------------------------
# Utility: name + email helpers
# ---------------------------------------------------------------------------

def _extract_name(text: str) -> str:
    """
    Extract a name from user input.
    Handles cases like: "My name is John", "I'm Sarah", "Jane Doe"
    Falls back to using the raw input (title-cased) if no pattern matches.
    """
    patterns = [
        r"(?:my name is|i'm|i am|call me)\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"^([A-Za-z][A-Za-z\s]{1,40})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
    return text.title()  # fallback


def _is_valid_email(email: str) -> bool:
    """Simple regex check for a valid email format."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_agent():
    """
    Compile the LangGraph state machine.
    
    Graph structure:
      classify_intent
          ↓ (conditional edges based on route_intent)
      ┌───┴────────────────────────────────┐
      handle_greeting   handle_inquiry   handle_lead_collection
          ↓                  ↓                    ↓
         END               END                  END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_inquiry", handle_inquiry)
    graph.add_node("handle_lead_collection", handle_lead_collection)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_lead_collection": "handle_lead_collection",
        },
    )

    # All handlers lead to END (one turn at a time)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("handle_lead_collection", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Initial State Factory
# ---------------------------------------------------------------------------

def create_initial_state() -> AgentState:
    """Create a fresh conversation state."""
    return AgentState(
        messages=[],
        intent="unknown",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        awaiting_field=None,
    )


# ---------------------------------------------------------------------------
# Main: CLI Chat Loop
# ---------------------------------------------------------------------------

def main():
    print("Starting main")
    print("\n" + "=" * 55)
    print("  🎬  AutoStream AI Assistant  (powered by Gemini)")
    print("=" * 55)
    print("  Type 'exit' or 'quit' to end the session.")
    print("  Type 'reset' to start a new conversation.\n")

    agent = build_agent()
    state = create_initial_state()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Thanks for chatting! Have a great day! 👋")
            break

        if user_input.lower() == "reset":
            state = create_initial_state()
            print("\n[Session reset. Starting fresh conversation.]\n")
            continue

        # Append user message and run one graph turn
        state["messages"] = state["messages"] + [
            {"role": "user", "content": user_input}
        ]

        result = agent.invoke(state)
        state = result  # Persist updated state for next turn

        # Print the last assistant message
        last_msg = state["messages"][-1]
        if last_msg["role"] == "assistant":
            print(f"\nAria: {last_msg['content']}\n")


if __name__ == "__main__":
    main()
