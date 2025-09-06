import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_default_openai_client,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

load_dotenv()

open_ai_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

model = "gemini-2.5-flash-lite"

external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_key,
)

set_default_openai_client(external_client)

llm_model = OpenAIChatCompletionsModel(
    model=model,
    openai_client=external_client,
)

set_tracing_disabled(True)

agent = Agent(
    name="simple_agent",
    instructions="You are a helpful assistant. Please give answers in form of bullet points (max 3 points).",
    model=llm_model,
)

response = Runner.run_sync(agent, "What is wikipedia? Explain in detail.")
print(response.final_output)