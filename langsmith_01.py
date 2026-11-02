import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=api_key,
    temperature=0.7,
    max_tokens=1000,
)

user_input = "Explain E=Mc^2"
response = model.invoke(
    user_input,
    config={"run_name": "explain-emc2"},
)
print(response.content)
