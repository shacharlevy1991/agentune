# Chat Simulator Design

## Overview

The chat simulator supports two primary flows:
1. **Full Simulation**: Both customer and agent are simulated
2. **Hybrid Simulation**: Simulated customer with a real agent

## Core Components

### 1. Participant Interface

The base interface for all simulated conversation participants (simulated agents and customers).

```python
class Participant(abc.ABC):
    @abstractmethod
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """
        Generate the next message based on conversation history.
        
        Args:
            conversation: The conversation history up to this point
            
        Returns:
            Message with timestamp, or None if participant is finished
        """
        ...
```

### 2. Message Structure

```python
@attrs.frozen
class MessageDraft:
    """Message content without timestamp - to be assigned during simulation."""
    content: str
    sender: ParticipantRole  # AGENT or CUSTOMER

@attrs.frozen
class Message:
    content: str
    timestamp: datetime
    sender: ParticipantRole  # AGENT or CUSTOMER
```

### 3. Channel Interface

Used for communication with real agents.

```python
class Channel(abc.ABC):
    @abstractmethod
    async def create_session(self) -> Session:
        """Create a new conversation session."""
        ...

class Session(abc.ABC):
    session_id: str
    
    @abstractmethod
    async def get_conversation(self) -> Conversation:
        """Get the conversation history so far."""
        ...
    
    @abstractmethod
    async def send(self, message: Message) -> None:
        """Send a message to the real agent."""
        ...
    
    @abstractmethod
    async def subscribe(self) -> AsyncIterator[Message]:
        """Subscribe to incoming messages from the agent."""
        ...
```

### 4. Simulation Result

Result object returned by conversation runners.

```python
@attrs.frozen
class SimulationResult:
    """Result of running a conversation simulation."""
    conversation: Conversation
    duration_seconds: float = 0.0
```

## Flow Implementations

### 1. Full Simulation Flow

Orchestrated by `FullSimulationRunner` - handles conversations between two simulated participants.


```python
@attrs.define
class FullSimulationRunner:
    """Runs conversations with both simulated customer and agent.
    
    Single-use runner that manages conversation state internally.
    Provides progress tracking capabilities for conversations.
    """
    
    customer: Participant
    agent: Participant
    initial_message: MessageDraft
    intent: Intent | None = None
    max_messages: int = 100
    base_timestamp: datetime | None = None  # If None, use current time when run() starts
    progress_callback: Callable[[Conversation, dict[str, Any]], None] | None = None
    
    # Private state - managed internally
    _conversation: Conversation = attrs.field(init=False)
    _is_complete: bool = attrs.field(init=False, default=False)
    _start_time: datetime | None = attrs.field(init=False, default=None)
    _current_timestamp: datetime = attrs.field(init=False)
    
    def __attrs_post_init__(self) -> None:
        """Initialize conversation with timestamped initial message."""
        # Will be set to base_timestamp or current time when run() starts
        self._current_timestamp = datetime.now()
        
        # Create initial message with timestamp
        initial_msg = Message(
            content=self.initial_message.content,
            sender=self.initial_message.sender,
            timestamp=self._current_timestamp
        )
        self._conversation = Conversation(messages=(initial_msg,))
    
    async def run(self) -> SimulationResult:
        """
        Execute the full simulation conversation.
        
        Returns:
            SimulationResult with conversation history and metadata
        """
        # Implementation details...
    
    @property
    def conversation(self) -> Conversation:
        """Get current conversation state (read-only access)."""
        return self._conversation
    
    @property
    def is_complete(self) -> bool:
        """Check if the simulation has completed."""
        return self._is_complete
    
    def get_progress(self) -> dict[str, Any]:
        """Get current progress information."""
        return {
            "message_count": len(self._conversation.messages),
            "max_messages": self.max_messages,
            "is_complete": self._is_complete,
            "elapsed_seconds": (
                (datetime.now() - self._start_time).total_seconds() 
                if self._start_time else 0
            ),
        }
```

**Key characteristics:**
- Simple alternating message exchange
- Deterministic message ordering based on simulated timestamps
- No real-time constraints - can run at maximum speed
- Both participants are AI-driven

**Usage Example:**
```python
# Create participants
customer = CustomerParticipant(...)
agent = AgentParticipant(...)

# Create and run simulation
runner = FullSimulationRunner(
    customer=customer, 
    agent=agent, 
    initial_message=MessageDraft(content="Hello, I need help", sender=ParticipantRole.CUSTOMER),
    intent=support_intent,
    max_messages=50
)
result = await runner.run()

print(f"Conversation completed in {result.duration_seconds:.2f}s")
print(f"Messages exchanged: {len(result.conversation.messages)}")

# Progress tracking example
def progress_handler(conversation: Conversation, progress: dict[str, Any]) -> None:
    print(f"Progress: {progress['message_count']}/{progress['max_messages']} messages")

runner_with_progress = FullSimulationRunner(
    customer=customer,
    agent=agent,
    initial_message=MessageDraft(content="Hello, I need help", sender=ParticipantRole.CUSTOMER),
    progress_callback=progress_handler
)
```

### 2. Hybrid Flow (Future Implementation)

Will be orchestrated by `HybridSimulationRunner` - handles conversations between simulated customer and real agent via channels.

**Key characteristics:**
- Real-time message synchronization
- Requires conversation termination detection
- Handles message timing conflicts
- Complex coordination between simulated and real participants

## Design Considerations

### 1. Conversation Termination

For real agent scenarios, implement multiple detection strategies:
- **Timeout-based**: No messages for X seconds
- **Pattern-based**: Detect closing phrases ("Have a nice day", "Goodbye", etc.)
- **LLM-based**: Analyze conversation completeness
- **Goal completion**: Check if customer intent was satisfied

### 2. Intent Handling

- Make intent optional for real agent scenarios
- Implement intent inference from conversation flow
- Support multiple intents per conversation

### 3. Message Timing and Conflicts

- Implement realistic typing delays for simulated participants
- Handle overlapping messages gracefully
- Consider message queuing for better realism

### 4. Error Handling

- Channel communication failures
- Participant response timeouts
- Message validation and formatting
- Graceful degradation for partial failures

## Implementation Notes

1. Use asyncio for all I/O operations
2. Implement proper logging for debugging
3. Make components testable in isolation
4. Support conversation persistence and replay
5. Consider metrics collection for analysis