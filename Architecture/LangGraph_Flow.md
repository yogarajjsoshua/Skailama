# LangGraph Workflow Flowchart

This document details the state-transition graph and operational flow of the **LangGraph Mini Promotion Agent** (`mini_promotion_graph`).

---

## 🔁 Graph Flow Diagram

The flowchart below traces the execution lifecycle from a merchant's input text to the final structured classification response. It highlights the conditional branching, terminal ends, and the human-in-the-loop interruption cycle.

```mermaid
graph TD
    %% Styling
    classDef startNode fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,rx:10px;
    classDef stepNode fill:#E3F2FD,stroke:#1565C0,stroke-width:2px;
    classDef condNode fill:#FFF8E1,stroke:#F57F17,stroke-width:2px;
    classDef endNode fill:#FFEBEE,stroke:#C62828,stroke-width:2px,rx:10px;
    classDef interruptNode fill:#EDE7F6,stroke:#651FFF,stroke-width:2px,stroke-dasharray: 5 5;

    %% Nodes
    START([START]):::startNode
    IntentNode["intent_classification<br>(LLM Intent Classifier)"]:::stepNode
    Router{"_route_after_intent<br>(Conditional Router)"}:::condNode
    
    ClarifyNode["clarification<br>(Clarification Node)"]:::stepNode
    InterruptPoint[["interrupt(question_payload)<br>(Pause and Wait for Input)"]]:::interruptNode
    
    UnsupportedNode["unsupported<br>(Format Blocker & Reason)"]:::stepNode
    
    TriggerNode["trigger_detection<br>(LLM Trigger Parser)"]:::stepNode
    AssemblyNode["state_assembly<br>(Default State Ingestion)"]:::stepNode
    ValidationNode["validation<br>(LLM Strict Validation)"]:::stepNode
    
    END([END]):::endNode

    %% Transitions
    START --> IntentNode
    IntentNode --> Router
    
    Router -->|clarification| ClarifyNode
    Router -->|unsupported / invalid| UnsupportedNode
    Router -->|free_gift / buy_x_get_y / tiered_discount| TriggerNode
    
    %% Clarification Loop with Interrupt
    ClarifyNode --> InterruptPoint
    InterruptPoint -->|Command(resume=clarification_text)| IntentNode
    
    %% Happy Path Pipeline
    TriggerNode --> AssemblyNode
    AssemblyNode --> ValidationNode
    ValidationNode --> END
    
    %% Terminal Unsupported Flow
    UnsupportedNode --> END
```

---

## ⚙️ Node-by-Node Explanation

### 1. `intent_classification`
* **Purpose**: Identifies the primary intent of the promotion message.
* **Operation**:
  * Appends the current user message to `history`.
  * Invokes the LLM using the `INTENT_CLASSIFICATION_PROMPT` containing full historical turns.
  * Expects a JSON response with the classified intent field (e.g. `tiered_discount`, `free_gift`, `buy_x_get_y`, `clarification`, or `unsupported`).
  * Resets downstream state keys (`tiers`, `blockers`, etc.) to clear old values.

### 2. `_route_after_intent` (Conditional Router)
* **Purpose**: Evaluates `state["feature"]` to select the next step.
* **Edges**:
  * Directs to **`clarification`** if the LLM needs more details.
  * Directs to **`unsupported`** if the request falls outside of out-of-scope/unrecognized categories.
  * Directs to **`trigger_detection`** for supported promotion types.

### 3. `clarification` (Human-In-The-Loop)
* **Purpose**: Manages incomplete request prompt clarifications.
* **Operation**:
  * Asserts a maximum retry barrier (default: `3` attempts). If exceeded, changes state to `unsupported` and proceeds.
  * Appends a bot clarification question to the conversation history.
  * Triggers an `interrupt(question_payload)`, saving the state checkpoint to MongoDB and pausing thread execution.
  * **Resume phase**: When the client posts a clarification response to `/chat/mini-promotion-agent/clarify`, the graph resumes, appends the user's text to history, overrides `state["message"]`, and loops back to `intent_classification`.

### 4. `unsupported` (Terminal Error)
* **Purpose**: Constructs a standard rejection reply for unsupported promotion types.
* **Operation**:
  * Sets `state["status"]` to `"unsupported"`.
  * Logs the error message to `state["blockers"]`.
  * Updates `state["history"]` with the system rejection log and routes to `END`.

### 5. `trigger_detection`
* **Purpose**: Parses conditional limits and promotion discount brackets.
* **Operation**:
  * Passes history and current message context to the LLM using the `TRIGGER_ONLY_CLASSIFICATION_PROMPT`.
  * Extracts structured tiers (such as cart values, quantity operators, and value brackets) and loads them into `state["tiers"]`.

### 6. `state_assembly`
* **Purpose**: Ingests defaults and metadata for state consistency.
* **Operation**: Sets standard defaults like `tier_behavior = "best_tier_only"` and updates state status.

### 7. `validation` (Final Consistency Check)
* **Purpose**: Runs a sanity validation on the classified intent against the final tier structure.
* **Operation**:
  * Packages the user's text and feature label, passing it to the validation model via `VALIDATION_CLASSIFICATION_PROMPT`.
  * If the validator marks `is_correct: False`, the node overrides `state["status"] = "unsupported"` and registers the validation reason in `state["blockers"]`.
  * Otherwise, sets `state["status"] = "supported"`.
