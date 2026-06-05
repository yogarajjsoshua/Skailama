import requests
import time 

API_URL_BASE = "http://127.0.0.1:8000/chat/"
LLM_INTENT_CLASSIFIER = "intent-classifier/llm"
LLM_INTENT_TRIGGERS_CLASSIFIER = "intent-classifier/triggers/llm"

def chat():
    print("CLI Chatbot started (FastAPI backend). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        print(user_input)
        if user_input.lower() == "exit":
            print("Bot: Bye!")
            break
        api_start = time.perf_counter()
        response = requests.post(
            API_URL_BASE+LLM_INTENT_TRIGGERS_CLASSIFIER,
            json={"message": user_input}
        )
        data = response.json()
        api_end = time.perf_counter()
        total_time = api_end - api_start
        print("Bot:", data["reply"],"\n",total_time)


if __name__ == "__main__":
    chat()