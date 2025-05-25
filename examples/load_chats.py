"""Example script for loading chat data."""

from pathlib import Path

from chat_simulator.loader import DCH2JsonLoader


def main():
    # Path to the chat data file
    data_path = Path("chats/dch2_processed_train.json")
    
    # Load the conversations
    print(f"Loading conversations from {data_path}...")
    loader = DCH2JsonLoader()
    conversations = loader.load(data_path)
    
    # Print some stats
    print(f"\nLoaded {len(conversations)} conversations")
    if conversations:
        first = conversations[0]
        print(f"\nFirst conversation ({first.id}):")
        print(f"- Messages: {len(first.messages)}")
        print(f"- Customer messages: {len(first.customer_messages)}")
        print(f"- Agent messages: {len(first.agent_messages)}")
        
        print("\nFirst message:")
        if first.messages:
            print(f"{first.messages[0].role.value.upper()}: {first.messages[0].content[:100]}...")


if __name__ == "__main__":
    main()
