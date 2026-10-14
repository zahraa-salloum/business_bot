"""Second Spark business assistant (Watt).
Same code as business_agent.ipynb without the demonstration cells, for deployment (for example Hugging Face Spaces),
plus a small ZeroGPU block at the top that the notebook does not need.
Set GEMINI_API_KEY or OPENAI_API_KEY in .env or in the Space secrets."""

# Hugging Face ZeroGPU only // ZeroGPU stops any app that has no @spaces.GPU function at startup.
# The bot never uses the GPU (the model runs on Google's servers), so this function is never called.
try:
    import spaces

    @spaces.GPU
    def zerogpu_placeholder():
        """Never called. It only exists so the ZeroGPU Space accepts the app."""
        return None
except ImportError:
    pass  # running locally // the spaces package only exists on Hugging Face

# imports
import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, InternalServerError
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_API_KEY:
    # Gemini through its OpenAI compatible endpoint // same client, same code
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
elif OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
else:
    raise RuntimeError("No API key found. Add GEMINI_API_KEY or OPENAI_API_KEY to the .env file.")

print("Model:", MODEL_NAME)

# models to fall back on when the daily free quota of one is finished // Gemini only
MODEL_CHAIN = [MODEL_NAME]
if GEMINI_API_KEY:
    MODEL_CHAIN += [m for m in ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
                    if m != MODEL_NAME]

MAX_RETRIES = 6
LONG_WAIT = 90  # seconds // a longer wait than this means the daily quota is finished
SERVER_RETRIES = 3  # extra tries when the provider has a temporary server error

def call_model(**kwargs):
    """One chat completion call that survives rate limits and temporary server errors."""
    global MODEL_NAME
    attempt = 0
    server_errors = 0
    while True:
        try:
            return client.chat.completions.create(model=MODEL_NAME, **kwargs)
        except InternalServerError as e:
            # temporary problem on the provider side // wait a little and try again
            server_errors += 1
            if server_errors > SERVER_RETRIES:
                raise
            print(f"[SERVER ERROR] {type(e).__name__}, trying again in {5 * server_errors}s")
            time.sleep(5 * server_errors)
        except RateLimitError as e:
            # the API says how long to wait // default to 20 seconds if it does not
            found = re.search(r"retryDelay\W+(\d+)s", str(e))
            wait = int(found.group(1)) + 1 if found else 20
            if wait > LONG_WAIT:
                position = MODEL_CHAIN.index(MODEL_NAME)
                if position + 1 >= len(MODEL_CHAIN):
                    raise
                MODEL_NAME = MODEL_CHAIN[position + 1]
                print(f"[RATE LIMIT] daily quota finished, switching to {MODEL_NAME}")
                continue
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            print(f"[RATE LIMIT] waiting {wait}s before trying again")
            time.sleep(wait)

ME_DIR = "me"

# full business profile (PDF)
reader = PdfReader(os.path.join(ME_DIR, "about_business.pdf"))
about_business = "\n".join((page.extract_text() or "") for page in reader.pages).strip()

# short summary (TXT)
with open(os.path.join(ME_DIR, "business_summary.txt"), "r", encoding="utf-8") as f:
    business_summary = f.read().strip()

print(f"PDF profile: {len(about_business)} characters | Summary: {len(business_summary)} characters")

# PERSONA (who the agent is)
PERSONA = """You are Watt, the cheerful bench assistant of Second Spark, an electronics repair and refurbishing workshop in Hamra, Beirut.
You talk like a friendly technician who loves bringing dead gadgets back to life: warm, practical and a little playful.
You like a small repair or electricity pun now and then (a spark, a short circuit, a recharge), but never more than one per reply.
You speak as part of the team, so you say "we" and "our workshop" when talking about Second Spark.
You keep replies short and clear, usually two to five sentences, and you explain technical things in plain words."""

# INSTRUCTIONS (operational guardrails)
INSTRUCTIONS = """1. Answer ONLY from the business summary and the business profile given below. Never invent prices, services, opening hours, warranties, team members or policies that are not written there.
2. If the answer to a question is not in the business information, do not guess. Call the record_feedback tool with the question, then tell the customer honestly that you do not have that information yet and that the team has been notified.
3. When a customer shows interest in a repair, a refurbished device, a workshop seat, a trade in or a Business Care Plan, encourage them to leave their name and email so the team can follow up.
4. Never end a lead conversation without asking for an email. Once you have BOTH a name and an email, call the record_customer_interest tool with the email, the name and a short note about what they want. Do not call it with a made up or missing email. Use the email exactly as the customer typed it. If it looks incomplete (for example "lina at gmail"), ask them to type the full address and never complete or guess it yourself.
5. Business Care Plans have no public price. Never quote a number for them, collect the lead instead.
6. When a customer gives feedback, a complaint or a suggestion, call the record_feedback tool with their words, thank them and stay polite even if they are upset.
7. Never claim that something was recorded unless the tool was actually called and returned a success status.
8. Stay in character as Watt and stay on the topic of Second Spark. If asked about something unrelated, politely steer back to what the workshop can help with.
9. Never diagnose a device with certainty over chat. Suggest the free diagnostic instead."""

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

def push(event, payload):
    """Logging mechanism shared by the tools: prints the event and appends it to logs/<event>.jsonl"""
    record = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
    print(f"[PUSH] {event}: {json.dumps(record, ensure_ascii=False)}")
    with open(os.path.join(LOG_DIR, f"{event}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

def record_customer_interest(email, name, message):
    """Saves a lead: who the customer is and what they are interested in."""
    if not EMAIL_RE.fullmatch((email or "").strip()):
        # the model gets this back and asks the customer for a proper email
        return {"status": "error", "reason": "A valid email address is required before the lead can be recorded."}
    push("leads", {"email": email.strip(), "name": name.strip(), "message": message.strip()})
    return {"status": "recorded", "detail": f"Lead saved for {name.strip()} <{email.strip()}>."}

def record_feedback(question):
    """Saves customer feedback or a question the agent could not answer."""
    push("feedback", {"question": question.strip()})
    return {"status": "recorded", "detail": "Feedback or unanswered question saved for the team."}

TOOL_FUNCTIONS = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}

MCP_TOOLS = [
    {
        "name": "record_customer_interest",
        "description": "Save a lead once the customer has given BOTH a name and an email.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The email address of the customer."},
                "name": {"type": "string", "description": "The name of the customer."},
                "message": {"type": "string", "description": "Short note on what the customer is interested in."},
            },
            "required": ["email", "name", "message"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["recorded", "error"]},
                "detail": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["status"],
        },
    },
    {
        "name": "record_feedback",
        "description": "Log customer feedback, a complaint, or a question that can not be answered from the business information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The feedback or the unanswered question, in the words of the customer."},
            },
            "required": ["question"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["recorded"]},
                "detail": {"type": "string"},
            },
            "required": ["status"],
        },
    },
]

def to_openai_tools(names):
    """Converts the MCP style definitions into the format the chat completions API expects."""
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
        for t in MCP_TOOLS if t["name"] in names
    ]

INTENTS = ["general_question", "lead_collection", "feedback"]

ROUTER_PROMPT = """You are a message router for Second Spark, an electronics repair workshop. Classify the LATEST customer message into exactly one intent:

general_question: asks about the business (services, prices, hours, team, warranty, location) or is small talk.
lead_collection: wants to book a repair, buy a device, join a workshop, get a quote or be contacted, OR is giving a name or email.
feedback: gives feedback, a complaint, praise or a suggestion about the business.

Use the recent conversation only as context. Reply with JSON only, like {"intent": "general_question"}."""

LEAD_WORDS = ["book", "quote", "sign up", "register", "reserve", "contact me", "call me", "interested in",
              "appointment", "my email", "my name is", "i want to buy", "i would like to buy"]
FEEDBACK_WORDS = ["feedback", "complain", "complaint", "disappointed", "suggest", "bad experience",
                  "terrible", "unhappy", "not happy", "great job", "review"]

def heuristic_route(message):
    """Keyword fallback, used only if the LLM router fails."""
    text = message.lower()
    if EMAIL_RE.search(text) or any(w in text for w in LEAD_WORDS):
        return "lead_collection"
    if any(w in text for w in FEEDBACK_WORDS):
        return "feedback"
    return "general_question"

def route(message, history=None):
    """Returns (intent, source) where source says whether the LLM or the fallback decided."""
    recent = (history or [])[-4:]
    context = "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(no previous messages)"
    try:
        response = call_model(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Recent conversation:\n{context}\n\nLATEST customer message:\n{message}"},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        intent = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0)).get("intent")
        if intent in INTENTS:
            return intent, "llm"
    except Exception as e:
        print(f"[ROUTER] LLM router failed ({type(e).__name__}), using keyword fallback")
    return heuristic_route(message), "heuristic"

# each intent gets its own tools and its own extra instruction
ROUTES = {
    "general_question": {
        "tools": ["record_feedback"],
        "force_tool": False,
        "note": "The customer is asking a general question. Answer it from the business information. "
                "If the answer is not there, call record_feedback with the question. "
                "If they sound interested in a service, invite them to leave a name and email.",
    },
    "lead_collection": {
        "tools": ["record_customer_interest", "record_feedback"],
        "force_tool": False,
        "note": "The customer is a potential lead. Find out what they need and get their name and email. "
                "If either one is missing, ask for it. Once you have both, call record_customer_interest.",
    },
    "feedback": {
        "tools": ["record_feedback"],
        "force_tool": True,
        "note": "The customer is giving feedback, a complaint or a suggestion. Log it with record_feedback, "
                "then thank them and respond with care.",
    },
}

system_prompt = f"""## PERSONA
{PERSONA}

## INSTRUCTIONS
{INSTRUCTIONS}

## BUSINESS SUMMARY
{business_summary}

## BUSINESS PROFILE (from about_business.pdf)
{about_business}"""

def build_system_prompt(intent):
    return system_prompt + f"\n\n## CURRENT ROUTE: {intent}\n{ROUTES[intent]['note']}"

print(system_prompt[:900] + "\n...")

MAX_TOOL_ROUNDS = 4

def handle_tool_call(call, customer_text=""):
    """Runs one tool call requested by the model and returns the result as a dict."""
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
        email = str(args.get("email", "")).strip().lower()
        if name == "record_customer_interest" and email and email not in customer_text.lower():
            # guard // the email must be one the customer actually typed, the model can not guess it
            result = {"status": "error", "reason": "The customer never typed this email. Ask them to type their full email address."}
        else:
            result = TOOL_FUNCTIONS[name](**args)
    except KeyError:
        result = {"status": "error", "reason": f"Unknown tool {name}."}
        args = {}
    except (TypeError, json.JSONDecodeError) as e:
        result = {"status": "error", "reason": f"Bad arguments: {e}"}
        args = {}
    return {"tool": name, "arguments": args, "result": result}

def run_agent(message, history, intent):
    """Returns (draft reply, list of tool calls that were made)."""
    config = ROUTES[intent]
    tools = to_openai_tools(config["tools"])
    messages = [{"role": "system", "content": build_system_prompt(intent)}] + history + [{"role": "user", "content": message}]
    tool_events = []
    # everything the customer typed so far // used to check that a saved email is real
    customer_text = "\n".join(m["content"] for m in history if m["role"] == "user") + "\n" + message

    for step in range(MAX_TOOL_ROUNDS):
        kwargs = dict(messages=messages, tools=tools)
        if config["force_tool"] and step == 0:
            # feedback route // the first step must be a tool call
            kwargs["tool_choice"] = "required"
        response = call_model(**kwargs)
        reply = response.choices[0].message

        if not reply.tool_calls:
            return (reply.content or "").strip(), tool_events

        # the model asked for a tool // run it and send the result back
        messages.append(reply)
        for call in reply.tool_calls:
            event = handle_tool_call(call, customer_text)
            tool_events.append(event)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(event["result"])})

    return "Sorry, something got tangled in my wires. Could you say that again?", tool_events

REFLECTION_PROMPT = """You are a strict quality reviewer for the customer assistant of Second Spark. You get the instructions the assistant must follow, the business information, the recent conversation, the latest customer message, the tools the assistant actually called and its DRAFT reply.

Check the draft against the instructions. In particular:
1. Does it state any price, service, hour, warranty or policy that is NOT in the business information?
2. If the customer is a lead and no lead was recorded yet, does it ask for the missing name or email?
3. Does it claim something was recorded or passed to the team when no tool call with status "recorded" is listed?
4. Does it stay in character as Watt, short and friendly?

Use the recent conversation as context: anything the customer already said there (a name, an email, what they want) counts as known.
Only set "passes" to false for a clear break of the instructions. Do not revise a draft just to change its style or wording.

Reply with JSON only:
{"passes": true or false, "issues": ["short description of each problem"], "revised_reply": "the corrected reply, or an empty string if the draft passes"}

If you revise, keep the voice of the draft and change only what is needed to fix the issues."""

reflection_log = []

def reflect(message, draft, tool_events, intent, history=None):
    """Checks the draft against the instructions and returns the final reply."""
    recent = (history or [])[-6:]
    context = "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(no previous messages)"
    review_input = (
        f"## INSTRUCTIONS\n{INSTRUCTIONS}\n\n"
        f"## BUSINESS INFORMATION\n{business_summary}\n\n{about_business}\n\n"
        f"## ROUTE\n{intent}\n\n"
        f"## TOOLS ACTUALLY CALLED\n{json.dumps(tool_events, ensure_ascii=False) if tool_events else 'none'}\n\n"
        f"## RECENT CONVERSATION\n{context}\n\n"
        f"## LATEST CUSTOMER MESSAGE\n{message}\n\n"
        f"## DRAFT REPLY\n{draft}"
    )
    final, issues, revised = draft, [], False
    try:
        response = call_model(
            messages=[
                {"role": "system", "content": REFLECTION_PROMPT},
                {"role": "user", "content": review_input},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        review = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        issues = review.get("issues") or []
        revised_reply = (review.get("revised_reply") or "").strip()
        if review.get("passes") is False and revised_reply:
            final, revised = revised_reply, True
    except Exception as e:
        issues = [f"reflection skipped ({type(e).__name__})"]

    entry = push("reflection", {"intent": intent, "model": MODEL_NAME, "user": message, "draft": draft,
                                "revised": revised, "issues": issues, "final": final})
    reflection_log.append(entry)
    return final

def chat(message, history):
    # keep only role and content from the Gradio history // this is the short term memory
    history = [{"role": m["role"], "content": m["content"]} for m in (history or [])
               if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]

    intent, source = route(message, history)
    print(f"[ROUTER] {intent} (decided by {source}) | model: {MODEL_NAME}")

    draft, tool_events = run_agent(message, history, intent)
    return reflect(message, draft, tool_events, intent, history)

demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="\u26A1 Second Spark, chat with Watt",
    description="Ask about repairs, refurbished devices, workshops and care plans. Fix it, do not bin it.",
    examples=[
        "What do you repair and how much does a screen replacement cost?",
        "I want to join the Saturday workshop",
        "Do you offer data recovery?",
        "I have some feedback about my last repair",
    ],
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
)

if __name__ == "__main__":
    demo.launch()
