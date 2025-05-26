# High Level Design

## Architecture Overview

Process: Chat Data → Clean → Summarize → Cluster → Generate SOPs

### Core Components

1. **Data Models** (`models.py`)
   - `Message`: Single chat message (customer/agent roles)
   - `Conversation`: Complete chat session with messages and optional outcome
   - `Outcome`: Optional string-based conversation outcome (e.g., "resolved", "escalated", "sale_completed"). Can be provided by user or extracted from chat content. User will provide a dictionary between and outcome and description. When not provided, we can potentially learn about the exsiting outcomes from chats.
   - `Customer Satisfaction`: Optional integer (1-10 scale) representing customer satisfaction for the conversation. Future support for other formats (e.g., categories, NPS, etc.) is planned. If not supplied, we can try to evaluate it automatically.

2. **Main components**
   - Customer
   - Agent
   - ConversationRunner

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
- Use langchain for LLM integration
- Consider using Langchain for caching LLM results and embeddings

### Extensibility
- Plugin architecture for input/output formats (future feature)
- Configurable LLM providers
