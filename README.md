# DentalAgent — AI Dental Receptionist

Multilingual AI receptionist for **BrightSmile Dental Clinic**. It books dental appointments through chat, answers clinic policy questions with RAG, validates dates/times, and assigns sequential token IDs.

Built with **FastAPI**, **LangGraph**, **OpenRouter**, and **FAISS**.

---

## Features

| Feature | Description |
|---------|-------------|
| **Appointment booking** | Collects name, phone, symptoms, and preferred date/time |
| **Smart date parsing** | Understands *tomorrow*, *next Monday*, *one week after*, *in 3 days*, etc. |
| **Date validation** | Rejects past dates, Sundays, invalid dates, and times outside clinic hours |
| **Schedule Q&A** | Answers questions like *"What timing is the doctor available tomorrow?"* |
| **Confirmation flow** | Reads back details and books immediately after user confirms |
| **Multilingual** | English, Urdu, Punjabi, Saraiki |
| **Policy RAG** | Answers clinic FAQs from local policy documents |
| **Safety guardrails** | Blocks prompt injection and inappropriate requests |
| **REST + WebSocket API** | HTTP chat and real-time WebSocket conversations |
| **Token system** | Assigns booking tokens (`D-100`, `D-101`, …) stored in CSV |

### Clinic hours (built-in rules)

- **Open:** Monday–Saturday, 9:00 AM – 7:00 PM (Asia/Karachi)
- **Closed:** Sundays

---

## Quick setup

### 1. Clone and install

```bash
git clone https://github.com/SomanAbbasi/Dental-Agent
cd DentalAgent

python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
CLINIC_NAME=BrightSmile Dental Clinic
CLINIC_PHONE=+92-42-0000000
```

Get an API key from [OpenRouter](https://openrouter.ai/).

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API usage

### Chat (single message)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"I want to book an appointment\", \"thread_id\": \"user-123\"}"
```

Use the **same `thread_id`** for follow-up messages in one conversation.

### WebSocket (multi-turn)

```
ws://localhost:8000/api/v1/ws/{thread_id}
```

Send plain text messages; receive JSON replies until `is_complete: true`.

### List appointments

```bash
curl http://localhost:8000/api/v1/appointments
```

---

## Example booking flow

```
User:  I want to book an appointment
Agent: Could you please tell me your full name?

User:  Ali Khan
Agent: What is your phone number?

User:  03001234567
Agent: What is the reason for your visit?

User:  Tooth pain
Agent: What date and time would you prefer?

User:  One week from now at 10 AM
Agent: [Readback with normalized date] ... Please reply yes to confirm.

User:  yes
Agent: Your appointment has been successfully booked! Token Number: D-100
```

### Schedule questions (handled without hallucinating)

```
User:  What timing is the doctor available tomorrow?
Agent: On Monday, July 07, 2026, the clinic is open from 9:00 AM to 7:00 PM.
       Would you like to book an appointment for that day?
```

### Invalid dates (rejected clearly)

```
User:  Book me on Sunday
Agent: I'm sorry, the clinic is closed on Sundays...

User:  Book me yesterday at 3 PM
Agent: That date has already passed. Please choose today or a future date.

User:  Book me at 8 PM
Agent: That time is outside clinic hours (9 AM – 7 PM).
```

---

## Project structure

```
DentalAgent/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/                 # Chat + appointments endpoints
│   ├── agents/              # LangGraph builder + router
│   ├── nodes/               # Pipeline nodes (extract, guardrail, write)
│   ├── schemas/             # Pydantic models
│   ├── config/              # Settings, LLM, prompts
│   ├── database/            # CSV token storage
│   ├── guardrails/          # Safety classifier
│   ├── rag/                 # FAISS policy retrieval
│   └── utils/
│       └── availability.py  # Date/time parsing & validation
├── data/
│   ├── policies/            # RAG source documents
│   └── slots/               # appointments.csv (runtime)
├── scripts/
│   └── chat_client.py       # CLI test client
├── requirements.txt
└── .env.example
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key |
| `OPENROUTER_MODEL` | No | `meta-llama/llama-3.1-8b-instruct:free` | LLM model |
| `OPENROUTER_BASE_URL` | No | OpenRouter URL | API base URL |
| `CLINIC_NAME` | No | BrightSmile Dental Clinic | Clinic name in replies |
| `CLINIC_PHONE` | No | +92-42-0000000 | Fallback phone number |
| `APP_ENV` | No | development | development / staging / production |
| `LOG_LEVEL` | No | DEBUG | Logging level |

---

## Logic changes (recent fixes)

### Booking reliability
- **Instant confirmation:** Saying *yes* now completes the booking in the same turn (guardrail → DB write) instead of requiring another message.
- **Removed dead `validator` node** that was never reached.

### Date & time intelligence (`app/utils/availability.py`)
- Parses relative dates: tomorrow, day after tomorrow, next weekday, in N days/weeks, one week after.
- Normalizes free-text dates into readable strings before saving (e.g. *"one week from now at 10 AM"* → *"Sunday, July 12, 2026 at 10:00 AM"*).
- Rejects past dates, Sundays, invalid calendar dates, and times outside 9 AM–7 PM.
- Answers schedule questions with real clinic hours instead of guessing.

### Conversation quality
- Availability answers now include a follow-up to continue booking.
- Confirmation only triggers on explicit yes — not on schedule questions.
- Fixed FAISS similarity filter (L2 distance: lower = better match).

### Cleanup
- Removed unused test suite, pre-commit config, and dead validator node.
- Trimmed `requirements.txt` to runtime dependencies only.

---

## Health check

```bash
curl http://localhost:8000/health
```

---

## License

MIT — use freely for learning and clinic demos.
