# Getting Started

This guide will help you get started with using the `chat_simulator` library.

## Installation

```bash
pip install chat_simulator
```

## Usage

### End-to-end flow

#### Train simulators

```python
# Load chats
customer_simulator_path = "simulators/example_customer_simulator"
agent_simulator_path = "simulators/example_agent_simulator"

loader = DCH2JsonLoader()
base_conversations = loader.load("path/to/chats.json")

# Init RAG based Customer simulator
rag_customer_simulator_builder = RAGCustomerSimulatorBuilder(base_conversations)
customer_simulator_factory = rag_customer_simulator_builder.build_factory()
customer_simulator_factory.save(customer_simulator_path)

# Init RAG based Agent simulator
rag_agent_simulator_builder = RAGAgentSimulatorBuilder(base_conversations)
agent_simulator_factory = rag_agent_simulator_builder.build_factory()
agent_simulator_factory.load(agent_simulator_path)
```

#### Simulate several chats

```python
# Load simulators
customer_simulator_factory = RAGCustomerSimulatorFactory.load(customer_simulator_path)
agent_simulator_factory = RAGAgentSimulatorFactory.load(agent_simulator_path)

# Setup conversation runner
conversation_runner = ConversationRunner(customer_simulator_factory, agent_simulator_factory)

# Get first rows, to initialize the simulation
loader = DCH2JsonLoader()
base_conversations = loader.load("path/to/chats.json")
conversations_to_simulate = base_conversations[:10]
conversations_first_messages = [c.messages[0] for c in conversations_to_simulate]

# Simulate several chats
simulation = conversation_runner.create_simulation(conversations_first_messages)
simulation.run()

# Save results
simulation.save("simulations/example_simulation")

# Show results
print(simulation.summary())
```

### Simulation with production Agent

```python
# Load customer simulator
customer_simulator_factory = RAGCustomerSimulatorFactory.load(customer_simulator_path)

# Init production agent
config = {
    "url": "http://clientDomain:port",
    "headers": {
        "Authorization": "Bearer your_token"
    }
}
agent_session_factory = RealAgentFactory.configure(config)

# Setup conversation runner
conversation_runner = ConversationRunner(customer_simulator_factory, agent_session_factory)

# Get first rows, to initialize the simulation
loader = DCH2JsonLoader()
base_conversations = loader.load("path/to/chats.json")
conversations_to_simulate = base_conversations[:10]
conversations_first_messages = [c.messages[0] for c in conversations_to_simulate]

# Simulate several chats
simulation = conversation_runner.create_simulation(conversations_first_messages)
simulation.run()

# Save results
simulation.save("simulations/example_production_simulation")

# Show results
print(simulation.summary())
```
