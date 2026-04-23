"""
tools.py
--------
Contains tool functions that the agent can call during execution.
In a real production system, mock_lead_capture would be replaced by
an actual CRM API call (e.g., HubSpot, Salesforce, Zoho CRM).
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulate capturing a qualified lead into a CRM system.

    In production, this function would:
    - Call a CRM API (HubSpot, Salesforce, etc.)
    - Send a welcome email via SendGrid or Mailchimp
    - Create a Slack notification for the sales team
    - Log the lead to a database

    Args:
        name:     Full name of the lead.
        email:    Email address of the lead.
        platform: Content platform they primarily use (YouTube, Instagram, etc.)

    Returns:
        dict: A result object with status and captured data.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    lead_data = {
        "status": "success",
        "lead": {
            "name": name,
            "email": email,
            "platform": platform,
            "captured_at": timestamp,
            "source": "AutoStream AI Chat",
        },
    }

    # Simulate the CRM call output
    print("\n" + "=" * 55)
    print("✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"   Time     : {timestamp}")
    print("=" * 55 + "\n")

    return lead_data
