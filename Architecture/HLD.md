# High-Level Design (HLD)

![System Architecture Infographic](./architecture_infographic.png)

This document provides a high-level architectural overview of the **Skailama Mini Promotion Agent** application. It describes the system boundary, component design, core data flows, and interactions between runtime modules.

---

## 🏗️ System Architecture

The Skailama system is built around a **FastAPI** web layer, a state-driven workflow engine powered by **LangGraph**, **MongoDB** for state and message persistence, and **Azure OpenAI** for processing natural language merchant inputs.

```mermaid
graph TD
    %% Styling
    classDef client fill:#E1F5FE,stroke:#0288D1,stroke-width:2px;
    classDef backend fill:#E8F5E9,stroke:#388E3C,stroke-width:2px;
    classDef database fill:#FFF3E0,stroke:#F57C00,stroke-width:2px;
    classDef external fill:#FCE4EC,stroke:#C2185B,stroke-width:2px;
    
    %% Nodes
    Client["Merchant Client / CLI Chat Client<br>(chat.py)"]:::client
    
    subgraph FastAPI_Server ["FastAPI Application (app/)"]
        Endpoints["API Endpoints / Routing<br>(app/main.py)"]:::backend
        LangGraph_Engine["LangGraph State Machine<br>(app/graph.py)"]:::backend
        Nodes["Pipeline Nodes<br>(app/nodes.py)"]:::backend
        DB_Client["MongoDB Client<br>(app/db/mongo.py)"]:::backend
    end
    
    subgraph Storage_Layer ["Storage Layer"]
        MongoDB[("MongoDB Database")]:::database
    end
    
    subgraph External_Services ["External Services"]
        Azure_OpenAI["Azure OpenAI Service<br>(gpt-4o)"]:::external
        LangSmith["LangSmith Observability"]:::external
    end
    
    %% Connections
    Client <-->|HTTP POST Requests / JSON Responses| Endpoints
    Endpoints <-->|Orchestrates Execution| LangGraph_Engine
    LangGraph_Engine <-->|Executes Nodes| Nodes
    Nodes <-->|Calls via LLM Wrapper| Azure_OpenAI
    Endpoints <-->|Saves Messages & Contexts| DB_Client
    LangGraph_Engine <-->|Persists State via MongoEngineCheckpointer| MongoDB
    DB_Client <-->|Reads/Writes Chats & Messages| MongoDB
    
    %% Tracing
    Endpoints -.->|Traces Node & LLM Steps| LangSmith
    Nodes -.->|Traces Executions| LangSmith
```

### Component Details

1. **Merchant Client (chat.py / Frontend)**: The interface where merchants enter promotion rules in natural language.
2. **FastAPI Web Server (app/main.py)**: Exposes endpoints to start conversations, resume paused conversations (clarification requests), and fetch message histories.
3. **LangGraph Pipeline (app/graph.py, app/nodes.py)**: Manages state compilation and steps (nodes). It is a Directed Acyclic Graph (DAG) with a conditional routing branch and a human-in-the-loop loopback cycle for clarifications.
4. **MongoEngineCheckpointer (app/mongo_checkpointer.py)**: LangGraph checkpointer integration that serializes and stores task writes and graph checkpoints in MongoDB.
5. **MongoDB Client (app/db/mongo.py)**: Manages chat metadata and history collections. It logs every message (human prompts and agent replies) to provide a rich API audit trail.
6. **Azure OpenAI (app/llm.py)**: Provides LLM execution for classification, parsing, and validation using gpt-4o.
7. **LangSmith Tracing**: Captures telemetry for performance analysis, prompt evaluation, and error tracing.

---

## 🔄 Core Interactions & Data Flow

### 1. Happy Path: Fully Supported Promotion

Below is the standard data flow for a well-formed promotion request (e.g., *"Spend $100 get 10% off"*).

```mermaid
sequenceDiagram
    autonumber
    actor Merchant as Merchant Client
    participant API as FastAPI Router
    participant DB as MongoDB Client
    participant LG as LangGraph Engine
    participant LLM as Azure OpenAI LLM
    participant LS as LangSmith Tracing

    Merchant->>API: POST /chat/mini-promotion-agent {message: "..."}
    Note over API: Generate UUID thread_id
    API->>DB: create_chat_if_missing(user_id, thread_id)
    API->>DB: save_message(role: "user", content)
    API->>LG: invoke(initial_state, thread_id)
    LG->>LS: Start Trace
    
    LG->>LLM: Node: intent_classification (Call with history)
    LLM-->>LG: returns intent JSON (e.g. tiered_discount)
    
    LG->>LLM: Node: trigger_detection (Call with history)
    LLM-->>LG: returns structured tiers JSON
    
    LG->>LG: Node: state_assembly (Assembles attributes)
    
    LG->>LLM: Node: validation (Validate classifications)
    LLM-->>LG: returns validation JSON
    
    LG-->>API: returns final PromotionState
    API->>DB: save_chat_context(last_context)
    API->>DB: save_message(role: "assistant", final_reply)
    LG->>LS: End Trace
    API-->>Merchant: returns structured JSON reply + thread_id
```

---

### 2. Human-In-The-Loop: Clarification Request and Resume Flow

If the merchant supplies an ambiguous or incomplete promotion request, the system triggers a clarification process. This pauses the LangGraph execution using an **interrupt**, persists the thread state, and resumes it upon receiving input from the merchant.

```mermaid
sequenceDiagram
    autonumber
    actor Merchant as Merchant Client
    participant API as FastAPI Router
    participant DB as MongoDB Client
    participant LG as LangGraph Engine
    participant LLM as Azure OpenAI LLM
    
    Merchant->>API: POST /chat/mini-promotion-agent {message: "Give discount"}
    API->>DB: Save user message
    API->>LG: invoke(initial_state)
    LG->>LLM: Node: intent_classification
    LLM-->>LG: returns intent: "clarification"
    
    Note over LG: Routes to node "clarification"
    Note over LG: clarification_node executes
    LG->>LG: Append bot clarification question to state history
    Note over LG: Calls interrupt(question_payload) -> PAUSES
    LG->>DB: Save checkpoints to MongoDB (MongoEngineCheckpointer)
    LG-->>API: Yields Interrupt / state snapshot
    
    API->>DB: save_chat_context(state)
    API->>DB: save_message(role: "assistant", content: clarification_question)
    API-->>Merchant: Returns status "clarification", thread_id, and question
    
    Note over Merchant, API: -- User replies with detail --
    
    Merchant->>API: POST /chat/mini-promotion-agent/clarify {thread_id, clarification: "Spend $100 get $10"}
    API->>DB: Save clarification message
    API->>LG: invoke(Command(resume=clarification_text), thread_id)
    Note over LG: Loads checkpoint, resumes clarification_node
    LG->>LG: Append user reply to state history
    LG->>LG: Sets message = clarification_text, loops back to intent_classification
    
    LG->>LLM: Node: intent_classification
    LLM-->>LG: returns intent: "tiered_discount" (supported!)
    Note over LG: Proceeds through trigger_detection, state_assembly, validation
    LG-->>API: Returns final PromotionState
    API->>DB: Save final state context & message
    API-->>Merchant: Returns status "supported", final tiers & thread_id
```
