import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

SPORTS = [
    "Swimming", "Gymnastics", "Track and Field", "Basketball",
    "Volleyball", "Wrestling", "Weightlifting", "Shooting", "Rowing", "Cycling",
]

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL = "gemini-2.0-flash-lite"

CHAT_SYSTEM = (
    "You are an expert Team USA sports analyst and LA28 Olympic Games enthusiast. "
    "Help fans track Team USA's momentum toward the 2028 Los Angeles Olympics. "
    "Be specific, energetic, and concise. Reference real athletes, recent results, and storylines. "
    "Keep answers to 2–4 sentences unless the user asks for more detail."
)

MOMENTUM_PROMPT = f"""You are a sports analyst covering Team USA's buildup to the 2028 Los Angeles Olympic Games (LA28).

Rate the current momentum (as of early 2026) for each Team USA sport on a scale of 1–100, considering:
- Recent international results (2023–2026)
- Depth of the athlete pipeline
- Coaching infrastructure and investment
- Trajectory vs. top rival nations

Sports to rate: {', '.join(SPORTS)}

For each sport also write a punchy one-line insight (≤12 words) on the single biggest momentum driver right now.

Respond ONLY with valid JSON — no markdown fences, no extra commentary:
{{
  "sports": [
    {{"name": "Swimming", "score": 91, "insight": "Elite relay depth makes them the team to beat."}},
    {{"name": "Gymnastics", "score": 84, "insight": "..."}}
  ]
}}

Include all 10 sports. Make scores varied and realistic."""


FALLBACK_DATA = [
    {"name": "Swimming",        "score": 93, "insight": "Unprecedented relay depth and generational talent converging for LA28."},
    {"name": "Track and Field", "score": 88, "insight": "Sprint and field events producing record-breaking performances worldwide."},
    {"name": "Gymnastics",      "score": 85, "insight": "Next generation stars emerging behind Simone Biles' continuing legacy."},
    {"name": "Basketball",      "score": 84, "insight": "NBA-led roster continuity gives Team USA unmatched roster depth."},
    {"name": "Volleyball",      "score": 76, "insight": "Women's squad on a historic win streak heading into home games."},
    {"name": "Rowing",          "score": 71, "insight": "Strong junior pipeline converting to senior medals ahead of schedule."},
    {"name": "Cycling",         "score": 65, "insight": "Track cycling program rebuilt with promising young sprinters."},
    {"name": "Wrestling",       "score": 62, "insight": "Freestyle depth strong, Greco-Roman program in rebuilding phase."},
    {"name": "Shooting",        "score": 58, "insight": "Rifle and pistol events showing steady improvement at World Cups."},
    {"name": "Weightlifting",   "score": 47, "insight": "Clean-sport reforms opening doors but the pipeline is still thin."},
]


def get_momentum_data():
    response = client.models.generate_content(model=MODEL, contents=MOMENTUM_PROMPT)
    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    data = json.loads(text)
    sports_list = data["sports"]
    sports_list.sort(key=lambda x: x["score"], reverse=True)
    for i, sport in enumerate(sports_list):
        sport["rank"] = i + 1
    return sports_list


@app.route("/")
def index():
    if not client:
        return render_template("index.html", sports=[], error="GEMINI_API_KEY is not set in .env")
    try:
        sports_data = get_momentum_data()
        error = None
    except Exception:
        # Quota exhausted or API unavailable — show static fallback so the UI is usable
        sports_data = [dict(rank=i + 1, **s) for i, s in enumerate(FALLBACK_DATA)]
        error = None
    return render_template("index.html", sports=sports_data, error=error)


@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not configured"}), 500

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []

    if not message:
        return jsonify({"error": "Empty message"}), 400

    gemini_history = [
        types.Content(
            role=entry["role"],
            parts=[types.Part(text=entry["content"])]
        )
        for entry in history
        if entry.get("role") in ("user", "model") and entry.get("content")
    ]

    chat_session = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=CHAT_SYSTEM),
        history=gemini_history,
    )
    response = chat_session.send_message(message)
    return jsonify({"response": response.text})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
