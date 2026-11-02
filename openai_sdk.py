

import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")

# wrap_openai sends each call to LangSmith.
# Needs LANGSMITH_TRACING, LANGSMITH_API_KEY, and LANGSMITH_PROJECT in .env.
openai_client = wrap_openai(OpenAI(api_key=api_key))

def chat_bot(user_input: str) -> None:
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
        temperature=1,
        max_completion_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        print(text, end="", flush=True)
    print()  

user_input = input("Enter a query: ")
chat_bot(user_input)
