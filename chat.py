import requests
import time 

API_URL_BASE = "http://127.0.0.1:8000/chat/"
LLM_INTENT_CLASSIFIER = "intent-classifier/llm"
LLM_INTENT_TRIGGERS_CLASSIFIER = "intent-classifier/triggers/llm"
MINI_PROMOTION_AGENT = "mini-promotion-agent"

def chat():
    print("CLI Chatbot started (FastAPI backend). Type 'exit' to quit.\n")
    thread_id = None
    in_clarification = False

    while True:
        user_input = input("You: ")
        print(user_input)
        if user_input.lower() == "exit":
            print("Bot: Bye!")
            break
        
        api_start = time.perf_counter()
        if in_clarification and thread_id:
            response = requests.post(
                API_URL_BASE + MINI_PROMOTION_AGENT + "/clarify",
                json={
                    "thread_id": thread_id,
                    "clarification": user_input
                }
            )
        else:
            response = requests.post(
                API_URL_BASE + MINI_PROMOTION_AGENT,
                json={"message": user_input}
            )
            
        data = response.json()
        api_end = time.perf_counter()
        total_time = api_end - api_start
        
        reply = data.get("reply", {})
        print("Bot:", reply, "\n", total_time)
        
        if isinstance(reply, dict) and reply.get("status") == "clarification":
            in_clarification = True
            thread_id = reply.get("thread_id")
        else:
            in_clarification = False
            thread_id = None


if __name__ == "__main__":
    chat()