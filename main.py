from agents import Agent,OpenAIChatCompletionsModel,Runner,set_tracing_disabled
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = os.getenv("GEMINI_BASE_PATH")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME")

set_tracing_disabled(True)

client = AsyncOpenAI(api_key=gemini_api_key,base_url=gemini_base_url)
model = OpenAIChatCompletionsModel(openai_client=client,model=str(gemini_model_name))

Command = "You are Helpful FAQ Agent Answer in concise and readable way"

agent:Agent = Agent(
    name="FAQ Agent",
    instructions=Command,
    model=model
)

for i in range(5):
    prompt = input(f"Ask FAQ {i+1}: -> ").strip()

    if not prompt:
        print("❗ Empty question, skipping.\n")
        continue
    
    result = Runner.run_sync(agent, prompt)
    print(f"Answer {i+1} -> {result.final_output}\n")