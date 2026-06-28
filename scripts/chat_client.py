"""
Testing

Usage:
    python scripts/chat_client.py
    python scripts/chat_client.py --lang urdu
    python scripts/chat_client.py --thread my-test-session-1
"""
import argparse
import uuid
import requests

BASE_URL = "http://localhost:8000/api/v1"


OPENING_MESSAGES = {
    "english": "Hello, I need to book a dental appointment",
    "urdu": "السلام علیکم، مجھے appointment book کرنی ہے",
    "punjabi": "Sat Sri Akal, menu appointment chahidi hai",
    "saraiki": "Salam, manu appointment chahidi hai",
}


def chat(message: str, thread_id: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "thread_id": thread_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Dental Agent Chat Client")
    parser.add_argument(
        "--lang",
        choices=["english", "urdu", "punjabi", "saraiki"],
        default="english",
    )
    parser.add_argument("--thread", default=None)
    args = parser.parse_args()

    thread_id = args.thread or f"test-{uuid.uuid4().hex[:8]}"
    print(f"\n{'='*60}")
    print(f"BrightSmile Dental Clinic — AI Receptionist")
    print(f"Thread ID: {thread_id}")
    print(f"Language: {args.lang}")
    print(f"Type 'quit' to exit | 'stats' to see appointments")
    print(f"{'='*60}\n")

    # Send opening message
    opening = OPENING_MESSAGES[args.lang]
    print(f"You: {opening}")

    try:
        result = chat(opening, thread_id)
        print(f"\nAgent: {result['reply']}")
        print(f"[Status: {result['validation_status']} | Lang: {result['language']}]\n")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server.")
        print("Start the server first with:")
        print("  uvicorn app.main:app --reload --port 8000")
        return

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "stats":
            try:
                r = requests.get(f"{BASE_URL}/appointments", timeout=10)
                data = r.json()
                print(f"\nAppointments: {data['stats']}")
                for appt in data["appointments"]:
                    print(
                        f"  {appt['token_id']} — "
                        f"{appt['patient_name']} — "
                        f"{appt['time_window']} — "
                        f"{appt['status']}"
                    )
                print()
            except Exception as e:
                print(f"Error fetching stats: {e}")
            continue

        try:
            result = chat(user_input, thread_id)
            print(f"\nAgent: {result['reply']}")

            status_line = f"[Status: {result['validation_status']} | Lang: {result['language']}]"
            if result.get("token_id"):
                status_line += f" [Token: {result['token_id']}]"
            print(f"{status_line}\n")

            if result.get("is_complete"):
                print("Booking complete! Conversation ended.")
                break

        except requests.exceptions.HTTPError as e:
            print(f"Server error: {e}")
        except requests.exceptions.ConnectionError:
            print("Connection lost. Is the server still running?")
            break


if __name__ == "__main__":
    main()