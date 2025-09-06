from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import os

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
    instructions="You are a helpful assistant, specialized in providing single-line responses.",
    model=llm_model
)

def run_agent():
    response = Runner.run_sync(agent, "What is AI?")
    print(response.final_output)