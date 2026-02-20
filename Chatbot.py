from groq import Groq
from json import load, dump, JSONDecodeError
import datetime
from dotenv import dotenv_values
import sys
import os
import asyncio
import edge_tts
import subprocess

# ================= ENV SETUP =================
env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")
MODEL = env_vars.get("MODEL", "llama3-8b-8192")

if not all([Username, Assistantname, GroqAPIKey]):
    raise ValueError("Missing required environment variables")

groq_client = Groq(api_key=GroqAPIKey)

CHATLOG_FILE = "ChatLog.json"
MAX_HISTORY = 20

# ================= PROFESSIONAL SYSTEM PROMPT =================
SYSTEM_PROMPT = f"""
You are {Assistantname}, a professional AI assistant speaking to {Username}.

Guidelines:
- Maintain a clear, concise, and authoritative tone.
- Avoid casual fillers.
- Deliver information with precision.
- Keep responses structured and professional.
- Reply only in English.
"""

# ================= CHAT LOG =================
def load_chatlog():
    try:
        with open(CHATLOG_FILE, "r") as f:
            return load(f)
    except (FileNotFoundError, JSONDecodeError):
        return []

def save_chatlog(messages):
    with open(CHATLOG_FILE, "w") as f:
        dump(messages, f, indent=4)

# ================= TIME INFO =================
def realtime_information():
    now = datetime.datetime.now()
    return (
        f"Day: {now.strftime('%A')}, "
        f"Date: {now.strftime('%d %B %Y')}, "
        f"Time: {now.strftime('%H:%M:%S')}"
    )

def needs_time_info(query: str) -> bool:
    keywords = ["time", "date", "day", "month", "year"]
    return any(word in query.lower() for word in keywords)

# ================= EDGE TTS (PROFESSIONAL VOICE) =================
async def speak_async(text: str):
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice="en-US-GuyNeural"   # Clean professional tone
        )

        await communicate.save("voice.mp3")

        # Open using Windows default media player
        subprocess.run(["start", "voice.mp3"], shell=True)

    except Exception as e:
        print(f"[VOICE ERROR] {e}")

# ================= CHATBOT CORE =================
def ChatBot(query: str) -> str:
    messages = load_chatlog()
    messages.append({"role": "user", "content": query})
    messages = messages[-MAX_HISTORY:]

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if needs_time_info(query):
        final_messages.append(
            {"role": "system", "content": realtime_information()}
        )

    final_messages.extend(messages)

    try:
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            max_tokens=1024,
            temperature=0.5,
            stream=True,
        )

        answer = ""
        print(f"{Assistantname}: ", end="", flush=True)

        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                answer += delta
                print(delta, end="", flush=True)

        print("\n")

        messages.append({"role": "assistant", "content": answer})
        save_chatlog(messages)

        return answer

    except Exception as e:
        print(f"[ERROR] {e}")
        return "An error occurred while processing your request."

# ================= MAIN LOOP =================
if __name__ == "__main__":
    print(f"{Assistantname} is running in Professional Mode. Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                sys.exit(0)

            reply = ChatBot(user_input)

            asyncio.run(speak_async(reply))

    except KeyboardInterrupt:
        print("\nSession interrupted. Exiting safely.")
