from openai import AzureOpenAI, APIConnectionError, AuthenticationError
from langsmith import traceable
import os
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT_NAME = os.getenv("OPEN_API_4_ENGINE")


@traceable(name="azure-openai-call", run_type="llm")
def llm(system_prompt, message, response_format={"type": "json_object"}, temperature=0):
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        response_format=response_format,
        temperature=temperature,
    )
    return response.choices[0].message.content