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
def llm(system_prompt, message, history=None, response_format={"type": "json_object"}, temperature=0):
    """Call the Azure OpenAI deployment.

    Parameters
    ----------
    system_prompt : str
        The system instruction prompt.
    message : str
        The current user message.
    history : list[dict], optional
        Prior conversation turns in ``[{"role": "user"|"system", "content": "..."}]``
        format.  Injected between the system prompt and the current user turn so
        the model has full context of the conversation so far.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Inject history turns (prior user inputs + chatbot system messages) if present
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )
    return response.choices[0].message.content