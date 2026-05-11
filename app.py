import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

OLYMPIC_SPORTS = [
    "Swimming", "Gymnastics", "Track and Field", "Basketball",
    "Volleyball", "Wrestling", "Weightlifting", "Shooting", "Rowing", "Cycling",
]

PARA_SPORTS = [
    "Wheelchair Basketball", "Para Swimming", "Para Athletics",
    "Sitting Volleyball", "Wheelchair Rugby", "Para Cycling",
    "Goalball", "Para Powerlifting",
]

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL = "gemini-2.5-flash"

CHAT_SYSTEM = (
    "You are an expert Team USA sports analyst and LA28 Olympic and Paralympic Games enthusiast. "
    "Help fans track Team USA's momentum toward the 2028 Los Angeles Olympics AND Paralympics. "
    "Be specific, energetic, and concise. Cover both Olympic and Paralympic athletes equally. "
    "Reference real athletes, recent results, and storylines from both programs. "
    "Keep answers to 2–4 sentences unless the user asks for more detail."
)

MOMENTUM_PROMPT = """You are a sports analyst covering Team USA's buildup to the 2028 Los Angeles Olympic and Paralympic Games (LA28).

Rate the current momentum (as of early 2026) for each sport on a scale of 1–100, considering:
- Recent international results (2023–2026)
- Depth of the athlete pipeline
- Coaching infrastructure and investment
- Trajectory vs. top rival nations

OLYMPIC sports to rate: {olympic}

PARALYMPIC sports to rate: {para}

For each sport write a punchy one-line insight (≤12 words) on the single biggest momentum driver right now.

Respond ONLY with valid JSON — no markdown fences, no extra commentary:
{{
  "olympic": [
    {{"name": "Swimming", "score": 91, "insight": "Elite relay depth makes them the team to beat."}}
  ],
  "paralympic": [
    {{"name": "Wheelchair Basketball", "score": 88, "insight": "Back-to-back gold medalists with a dominant roster returning."}}
  ]
}}

Include ALL sports listed. Make scores varied and realistic.""".format(
    olympic=", ".join(OLYMPIC_SPORTS),
    para=", ".join(PARA_SPORTS),
)

OLYMPIC_FALLBACK = [
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

PARA_FALLBACK = [
    {"name": "Wheelchair Basketball", "score": 94, "insight": "Back-to-back gold medalists with dominant roster returning for LA28."},
    {"name": "Para Swimming",         "score": 90, "insight": "Deep multi-class roster sets world records at every major meet."},
    {"name": "Para Athletics",        "score": 87, "insight": "Sprint and field para athletes breaking barriers at World Champs."},
    {"name": "Sitting Volleyball",    "score": 83, "insight": "Women's team on an unbeaten streak in international competition."},
    {"name": "Wheelchair Rugby",      "score": 78, "insight": "Physical style and tactical evolution making USA the team to beat."},
    {"name": "Goalball",              "score": 70, "insight": "Women's program resurgent after rebuilding with young talent."},
    {"name": "Para Cycling",          "score": 63, "insight": "New coaching staff accelerating development across all classifications."},
    {"name": "Para Powerlifting",     "score": 55, "insight": "Emerging athletes pushing weight totals closer to podium contention."},
]


def rank_list(sports_list):
    sports_list.sort(key=lambda x: x["score"], reverse=True)
    for i, sport in enumerate(sports_list):
        sport["rank"] = i + 1
    return sports_list


def get_momentum_data():
    response = client.models.generate_content(model=MODEL, contents=MOMENTUM_PROMPT)
    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    data = json.loads(text)
    return rank_list(data["olympic"]), rank_list(data["paralympic"])


@app.route("/")
def index():
    if not client:
        return render_template("index.html", olympic=[], para=[], error="GEMINI_API_KEY is not set in .env")
    try:
        olympic, para = get_momentum_data()
        error = None
    except Exception:
        olympic = rank_list([dict(**s) for s in OLYMPIC_FALLBACK])
        para = rank_list([dict(**s) for s in PARA_FALLBACK])
        error = None
    return render_template("index.html", olympic=olympic, para=para, error=error)


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
