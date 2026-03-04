# System Architecture Diagram

This file contains the Mermaid.js source for the Uncle Bao AI architecture.

```mermaid
graph TD
    %% Entry Point
    START((User Message)) --> Entry{route_entry}

    %% Perception Layer
    subgraph Perception_Layer [Perception Layer - Parallel Processing]
        Entry -->|New User| FG[first_greeting]
        Entry -->|Existing| CL[classifier_node]
        Entry -->|Existing| EX[extractor_node]
        
        CL --> WAIT[wait_node]
        EX --> WAIT[wait_node]
    end

    %% Decision Layer
    subgraph Decision_Layer [Decision Layer - Logic Routing]
        WAIT --> CR{core_router}
        FG --> END1((END))
    end

    %% Execution Layer
    subgraph Execution_Layer [Execution Layer - Specialized Agents]
        CR -->|High Value| HV[high_value_node]
        CR -->|Art| AD[art_director_node]
        CR -->|Low Budget| LB[low_budget_node]
        CR -->|Incomplete Profile| IV[interviewer_node]
        CR -->|Sales Ready| CS[consultant_node]
        CR -->|Chit Chat| CC[chit_chat_node]
        CR -->|Direct Handoff| HH[human_handoff_node]
    end

    %% Routing back to Handoff or End
    HV -->|Tool: Handoff| HH
    AD -->|Tool: Handoff| HH
    LB -->|Tool: Handoff| HH
    CS -->|Tool: Handoff| HH
    
    IV --> END2((END))
    CC --> END2
    HH --> END2
    HV -->|Done| END2
    AD -->|Done| END2
    LB -->|Done| END2
    CS -->|Done| END2

    %% Styling
    style Perception_Layer fill:#f5f5f5,stroke:#333,stroke-dasharray: 5 5
    style Decision_Layer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Execution_Layer fill:#f1f8e9,stroke:#33691e
    style Entry fill:#fff9c4
    style CR fill:#fff9c4
```

## Architecture Highlights for Interviews

1. **Layered Design**: Separation of concerns between Perception (Understanding), Decision (Routing), and Execution (Action).
2. **Parallel Execution**: `classifier` and `extractor` run in parallel to minimize LLM latency.
3. **Deterministic Routing**: The `core_router` uses structured state (Pydantic) rather than raw LLM prompts to decide transitions, ensuring system stability.
4. **Resilient State**: Custom `reduce_profile` logic handles incremental data updates and fuzzy matching for messy user inputs.
