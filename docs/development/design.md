# High Level Design

## Architecture Overview

### Core Components

1. Data Models (models.py)
 - Message: Single chat message (customer/agent roles) with mandatory created_at: datetime capturing the absolute (simulated or real) timestamp of the utterance.
 - Conversation: Complete chat session with messages and optional outcome
 - Outcome: Optional string-based conversation outcome (e.g., "resolved", "escalated", "sale_completed"). Can be provided by user or extracted from chat content. User will provide a dictionary between an outcome and description. When not provided, we can potentially learn about the existing outcomes from chats.
 - OutcomeSchema: Ordered mapping {id: str → description: str} that defines the legal Outcome labels for a simulation run. Keys are lowercase slugs (e.g., "resolved"); values hold human‑readable descriptions. This schema is injected into analytics and evaluators to keep metrics comparable across batches.

2. **Main Components**

**Participant** (base class): Abstract interface for any conversation participant with a `role: str` attribute (e.g., "customer", "agent", "supervisor"). Exposes:

- `get_next_message(conversation: Conversation) -> Message`

This method returns the participant's next utterance given the conversation history.

**CustomerParticipant** (subclass): Implementation of `Participant` instantiated with `role="customer"`. No additional behavior today, but provides a hook for future customer-specific logic (e.g., sentiment modeling, problem template selection).

**AgentParticipant** (subclass): Implementation of `Participant` instantiated with `role="agent"`. Future extensions may include tool-calling, policy compliance, etc.

**ConversationRunner**: Central orchestration engine with four internal responsibilities:

1. **Bootstrapper** – seeds the chat with a kickoff message, scenario, or system prompt based on a configured topic/start state.

2. **TurnScheduler** – selects which `Participant` speaks next. Strategies can be round-robin, random jitter, or learned from existing chats. It may schedule the same participant multiple times to emulate asynchronous customer replies or agent multi-step tool calls.

3. **OutcomeEvaluator** – owns the authoritative list of `Outcome` labels (e.g., resolved, escalated, sale_completed). Two supported modes:
   - *Centralized*: Runner inspects the full `Conversation` and decides when an outcome is reached.
   - *Distributed*: Each participant can emit an optional `suggested_outcome`; the Runner collects suggestions and applies an arbitration rule (e.g., majority, agent-weighted, confidence score) before committing the final outcome.

4. **Timing/Async Simulation** – optionally injects per-turn delays to approximate real-world latency. Delays can be logical (store a `timestamp_delta` on each `Message` without pausing execution) or wall-clock (actual `sleep()` when running in real-time demos).

By depending only on the `Participant` base type, `ConversationRunner` can later accommodate new roles (e.g., `SupervisorParticipant`) without modification.

3. **Simulation techniques**
   - Fine-tuned LLMs
   - RAG using existing chats as context

4. **Simulators**
   - CustomerSimulator - base class
   - AgentSimulator - base class


## Technical Considerations

### Performance
- Async LLM calls
- Caching of LLM calls
- Parallel chat execution

### Frameworks
- Use langchain for LLM integration (with async chains)
- Consider using Langchain for caching LLM results and embeddings
- All LLM calls through LangChain should use async methods

### Extensibility
- Plugin architecture for input/output formats (future feature)
- Configurable LLM providers
