"""
- A node that retrieves and put them in state
- A Chatbot node, that responds using the retrieved memories
- A node that takes the current conversation, and extracts new memories
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.embeddings import init_embeddings
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import TypedDict, Annotated
from operator import add
from datetime import datetime

llm = ChatOpenAI(model="gpt-4o")

class ChatMessagesState(TypedDict):
    messages: Annotated[list, add]
    memory_context: str


# Memory Retrieval Node
def retrieve_memories(state: ChatMessagesState, config: RunnableConfig, store: BaseStore):

    # Get user_id from config
    user_id = config["configurable"].get("user_id", "default_user")

    print(f"\n{'='*70}")
    print(f"📝 MEMORY RETRIEVAL NODE")
    print(f"{'='*70}")
    print(f"Retrieving memories for user: {user_id}")

    # Search for memories
    user_memories_namespace = (user_id, "memories")

    memories = store.search(
        user_memories_namespace,
        query="What are the facts about this user?"
    )

    memory_context = ""
    if memories:
        print(f"✔️ Found {len(memories)} memories")
        memory_texts = []
        for i, memory in enumerate(memories, 1):
            text = memory.value.get("text", "")
            print(f"{i}. {text}")
            memory_texts.append(text)

        memory_context = "\n".join([f"- {text}" for text in memory_texts])
    else:
        print("ℹ️ No memories found (new user)")


    print(f"{'='*70}")

    return {
        "memory_context": memory_context
    }

# Chatbot Node
def chatbot (state: ChatMessagesState, config: RunnableConfig):

    user_id = config["configurable"].get("user_id", "default_user")

    print(f"\n{'='*70}")
    print(f"📝 CHATBOT NODE")
    print(f"{'='*70}")
    print(f"Generating response for user: {user_id}")

    memory_context = state.get("memory_context", "")

    print("\n💬 GENERATING RESPONSE...")

    if memory_context:
        print("✔️ Using retrieved memories for personalization")

        system_prompt = f"""You're a helpful assistant with memory of past conversations.

        What you remember about this user:
        {memory_context}

        Use this information to personalize your response. Be natural, and conversational
        """
    else:
        print("ℹ️ No memory context available")

        system_prompt = "You're a helpful assistant. This is your first conversation with this user"

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    # Generate response
    response = llm.invoke(messages)
    print(f"✅ Response Generated: {response.content[:80]}...")
    print(f"{'='*70}\n")

    return {
        "messages": [response]
    }


# Memory Extraction
def extract_and_save_memories(state: ChatMessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")

    print(f"\n{'='*70}")
    print(f"📝 MEMORY EXTRACTION NODE")
    print(f"{'='*70}")
    print(f"Extracting memories for user: {user_id}")

    if len(state["messages"]) >= 2:
        user_message = state["messages"][-2].content
        assistant_message = state["messages"][-1].content
    else:
        print("⚠️ Not enough messages to extract from")
        print(f"{'='*70}\n")
        return state
    
    print(f"Used said: {user_message[:60]}...")
    print(f"Assistant said: {assistant_message[:60]}...")

    print("\n🔍 EXTRACTING FACTS...")

    extract_prompt = f"""Look at this conversation and extract any facts worth remembering about the user.

    User: {user_message}
    Assistant: {assistant_message}

    List each fact on a new line starting with a dash (-).
    Only include clear, factual information about the USER (not about the assistant).
    If there are no facts to remember, respond with: NONE

    Examples of good facts:
    - User's name is Alice
    - User works as a teacher
    - User enjoys hiking
    - User is learning Python

    Examples of bad facts (don't include these):
    - The assistant was helpful
    - We had a conversation
    - The user asked a question"""

    extraction = llm.invoke(extract_prompt).content

    print(f"Extraction result: {extraction[:80]}...")

    print("\n💾 SAVING TO STORE...")

    if "NONE" not in extraction.upper():

        lines = [line.strip() for line in extraction.split("\n") if line.strip().startswith("-")]

        saved_count = 0
        for line in lines:
            fact = line[1:].strip()

            if fact and len(fact) > 5:
                memory_key = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

                store.put(
                    namespace = (user_id, "memories"),
                    key=memory_key,
                    value={
                        "text": fact,
                        "timestamp": datetime.now().isoformat(),
                        "source": "conversation"
                    }
                )
                print(f"✔️ Saved: {fact}")
                saved_count += 1

        if saved_count == 0:
            print("No valid facts to save")
    
    else:
        print("ℹ️ No new facts to save")

    print(f"{'='*70}\n")

    return state


builder = StateGraph(ChatMessagesState)

builder.add_node(retrieve_memories)
builder.add_node(chatbot)
builder.add_node("extract_memories", extract_and_save_memories)

builder.add_edge(START, "retrieve_memories")
builder.add_edge("retrieve_memories", "chatbot")
builder.add_edge("chatbot", "extract_memories")
builder.add_edge("extract_memories", END)

checkpointer = MemorySaver()

store_embeddings_model = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": store_embeddings_model,
        "dims": 1536,
        "fields": ["text", "$"]
    }
)

graph = builder.compile(
    checkpointer=checkpointer,
    store = store
)

config = {
    "configurable": {
        "thread_id": "chat_001",
        "user_id": "sarah"
    }
}

# Turn 1: Introduction
print("\n" + "-"*70)
print("TURN 1: Introduction")
print("-"*70)

sarah_message_1 = "Hi! My name is Sarah and I'm a data scientist"

result = graph.invoke(
    {
        "messages": [HumanMessage(content=sarah_message_1)]
    },
    config = config
)

print(f"\nUSER: {sarah_message_1}")
print(f"ASISTANT: {result['messages'][-1].content}")

# Turn 2: Share work info
print("\n" + "-"*70)
print("TURN 2: Sharing work information")
print("-"*70)

sarah_message_2 = "I'm currently working on a machine learning project using Python and TensorFlow."

result = graph.invoke(
    {"messages": [HumanMessage(content=sarah_message_2)]},
    config=config
)

print(f"\n📨 USER: {sarah_message_2}")
print(f"📤 ASSISTANT: {result['messages'][-1].content}")

# Turn 3: Share hobbies
print("\n" + "-"*70)
print("TURN 3: Sharing hobbies")
print("-"*70)

sarah_message_3 = "In my free time, I love playing guitar and going on weekend hikes."

result = graph.invoke(
    {"messages": [HumanMessage(content=sarah_message_3)]},
    config=config
)

print(f"\n📨 USER: {sarah_message_3}")
print(f"📤 ASSISTANT: {result['messages'][-1].content}")

# Turn 4: Share dietary preference
print("\n" + "-"*70)
print("TURN 4: Sharing preferences")
print("-"*70)

sarah_message_4 = "I'm vegetarian and I prefer coffee over tea."

result = graph.invoke(
    {"messages": [HumanMessage(content=sarah_message_4)]},
    config=config
)

print(f"\n📨 USER: {sarah_message_4}")
print(f"📤 ASSISTANT: {result['messages'][-1].content}")


""" print("\n\n" + "="*70)
print("INSPECTING STORED MEMORIES")
print("="*70)

memories = store.search(
    ("sarah", "memories"),
    query = "What are the facts about this user?"
)

print(f"\n Total memories stored for Sarah: {len(memories)}\n")

for i, memory in enumerate(memories, 1):
    print(f"{i}. {memory.value['text']}")
    print(f"Key: {memory.key}")
    print(f"Timestamp: {memory.value['timestamp']}")
    print(f"Source: {memory.value['source']}")
    print(f"Created at: {memory.created_at}\n") """


print("="*70)
print("DEMONSTRATION 2: Sarah Returns (Different Day)")
print("="*70)
print("New conversation thread - memories should persist!\n")

new_config = {
    "configurable": {
        "thread_id": "chat_002",
        "user_id": "sarah"
    }
}

print("-"*70)
print("DAY 2 - First Message")
print("-"*70)

sarah_day2_message_1 = "Good Morning! What do you remember about me?"

result = graph.invoke(
    {
       "messages": [HumanMessage(content=sarah_day2_message_1)]  
    },
    config = new_config
)

print(f"\nUSER: {sarah_day2_message_1}")
print(f"\n ASSISTANT: {result['messages'][-1].content}")


# Turn 2: Reference stored info
print("\n" + "-"*70)
print("DAY 2 - Second Message")
print("-"*70)

sarah_day2_message_2 = "Can you suggest a lunch place considering my dietary preferences?"

result = graph.invoke(
    {"messages": [HumanMessage(content=sarah_day2_message_2)]},
    new_config
)

print(f"\n📨 USER: {sarah_day2_message_2}")
print(f"📤 ASSISTANT: {result['messages'][-1].content}")

print("\n\n" + "="*70)
print("DEMONSTRATION 3: Different User - John")
print("="*70)
print("Testing memory isolation between users\n")

john_config = {
    "configurable": {
        "thread_id": "chat_011",
        "user_id": "john"
    }
}

print("-"*70)
print("JOHN'S FIRST MESSAGE")
print("-"*70)

john_message_1 = "Hey! I am John, What do you know about me?"

result = graph.invoke(
    {
        "messages": [HumanMessage(content=john_message_1)]
    },
    john_config
)

print(f"\nUSER (John): {john_message_1}")
print(f"\nASSISTANT: {result['messages'][-1].content}")
