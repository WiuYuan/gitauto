#!/usr/bin/env python3
"""
Test script for WebUI <-> ExternalServer choice workflow
This script tests send_choice() and waits for user's selection via frontend.
"""

import time
import os
import sys
from src.services.external_server import ExternalServer

# -------------------------------
# Load existing WebSocket server port
# -------------------------------
if not os.path.exists(".webui_port"):
    print("❌ .webui_port not found. Please start the WebUI backend first.")
    sys.exit(1)

with open(".webui_port", "r", encoding="utf-8") as f:
    port = int(f.read().strip())

# Connect to existing ExternalServer
es = ExternalServer(port=port)
print(f"🌐 Connected to ExternalServer on port {port}\n")


# -------------------------------
# Helper functions
# -------------------------------
def wait_for_choice_response(choice_id: str, timeout: float = 30.0):
    """Wait for frontend response with timeout"""
    start = time.time()
    while time.time() - start < timeout:
        msgs = es.get_messages()
        for msg in msgs:
            print(msg)
            if msg.get("type") == "choice_response":
                data = msg.get("data", {})
                if data.get("choiceId") == choice_id:
                    return data
        time.sleep(0.2)
    return None


# -------------------------------
# Main test logic
# -------------------------------
def test_send_choice():
    """Send a single choice request and wait for frontend selection"""
    print("🧪 Sending test choice request to frontend...")

    choice_id = es.send_choice(
        question="Which framework do you prefer for frontend development?",
        options=["React", "Vue", "Svelte", "Angular"],
    )

    print(f"📨 Choice request sent with ID: {choice_id}")
    print("⏳ Waiting for user response via WebUI...")

    response = wait_for_choice_response(choice_id, timeout=30.0)

    if response:
        print(f"✅ Received response:")
        print(f"   Choice ID: {choice_id}")
        print(
            f"   Selected Option #{response['selectedOption']}: {response['optionText']}"
        )
        return True
    else:
        print("❌ No response received within timeout.")
        return False


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting WebUI Choice Test\n")

    success = test_send_choice()

    print("\n📊 Test Result:")
    print("✅ PASS" if success else "❌ FAIL")

    print(
        "\n💡 Note: Make sure your React frontend is running and connected to this port."
    )
