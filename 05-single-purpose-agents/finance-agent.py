from langchain.agents import create_agent
from langchain.tools import tool

from dataclasses import dataclass
from langchain.tools import ToolRuntime

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    wrap_model_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    wrap_tool_call
)
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy

from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import AIMessage


basic_model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.5,
    max_tokens=512
)

premium_model = init_chat_model(
    "gpt-4o",
    max_tokens=2048
)

platinum_model = init_chat_model(
    "gpt-4o"
)

@dataclass
class UserContext:
    user_id: str
    user_name: str
    membership_tier: str #'basic', 'premium', 'platinum'
    preferred_currency: str

class FinancialResponse(BaseModel):

    summary: str = Field(description="A brief summary of the response (1 - 2 sentences)")
    details: str = Field(description= "Detailed explanation or data")

    action_items: list[str] = Field(
        default_factory=list,
        description="List of recommended actions the user should take"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or concerns to highlight"
    )

    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence level in the advice provided"
    )


USER_DATABASE = {
    "user_001": {
        "name": "Alice Johnson",
        "accounts": {
            "checking": 2500.00,
            "savings": 15000.00,
            "investment": 45000.00,
        },
        "transactions": {
            "checking": [
                {"date": "2025-01-15", "description": "Grocery Store", "amount": -85.50},
                {"date": "2025-01-14", "description": "Direct Deposit", "amount": 3200.00},
                {"date": "2025-01-13", "description": "Electric Bill", "amount": -120.00},
                {"date": "2025-01-12", "description": "Restaurant", "amount": -45.00},
                {"date": "2025-01-10", "description": "Gas Station", "amount": -55.00},
            ],
            "savings": [
                {"date": "2025-01-01", "description": "Interest Payment", "amount": 12.50},
                {"date": "2024-12-15", "description": "Transfer from Checking", "amount": 500.00},
            ],
            "investment": [
                {"date": "2025-01-14", "description": "Dividend - AAPL", "amount": 125.00},
                {"date": "2025-01-10", "description": "Buy - VTI", "amount": -1000.00},
            ],
        },
    },
    "user_002": {
        "name": "Bob Smith",
        "accounts": {
            "checking": 1200.00,
            "savings": 8000.00,
            "investment": 22000.00,
        },
        "transactions": {
            "checking": [
                {"date": "2025-01-15", "description": "Coffee Shop", "amount": -5.50},
                {"date": "2025-01-14", "description": "Freelance Payment", "amount": 1500.00},
            ],
            "savings": [
                {"date": "2025-01-01", "description": "Interest Payment", "amount": 6.50},
            ],
            "investment": [
                {"date": "2025-01-12", "description": "Dividend - VTI", "amount": 45.00},
            ],
        },
    },
}

"""
Define Tools
"""

@tool
def get_account_balance(
    account_type: str,
    runtime: ToolRuntime[UserContext]
    ) -> str:

    """
        Get the current balance for a specific account for a user

        Args:
            account_type: Type of account - 'checking', 'savings', or 'investment'
    """

    user_id = runtime.context.user_id
    currency = runtime.context.preferred_currency
    user_data = USER_DATABASE.get(user_id, {})

    balance = user_data.get("accounts", {}).get(account_type.lower())

    if balance is not None:
        if currency == "EUR":
            balance = balance * 0.92 
            return f"Your {account_type} account balance is €{balance:,.2f}"
        return f"Your {account_type} account balance is ${balance:,.2f}"
    
    return f"Unknown account type: {account_type}. Available: checking, savings, investment"


@tool
def get_recent_transactions(
    account_type: str, 
    limit: int = 5,
    runtime: ToolRuntime[UserContext] = None
    ) -> str:

    """
        Get recent transactions for an account of a user

        Args:
         account_type: Type of account - savings, investment, checking
         limit: Number of transactions to return (default: 5)
    """

    
    user_id = runtime.context.user_id
    user_data = USER_DATABASE.get(user_id, {})

    account_transactions = user_data.get("transactions", {}).get(account_type.lower(), [])[:limit]

    if not account_transactions:
        return f"No transactions found for {account_type}"
    
    result = f"Recent transactions for {account_type}:\n"

    for t in account_transactions:
        sign = "+" if t["amount"] > 0 else ""
        result += f"{t['date']}: {t['description']} ({sign}${t['amount']:,.2f})\n"

    return result

@tool
def calculate_budget(monthly_income: float, expense_category: str) -> str:
    
    """
        Calaculate recommended budget allocation for an expense category

        Args:
            monthly_income: User's monthly income
            expense_category: Categories like 'housing', 'food' etc
    """

    allocations = {
        "housing": 0.30,
        "food": 0.12,
        "transportation": 0.10,
        "utilities": 0.08,
        "savings": 0.20,
        "entertainment": 0.05,
        "healthcare": 0.05,
        "other": 0.10,
    }

    percentage = allocations.get(expense_category.lower())
    if percentage is None:
        return f"Unknown category: {expense_category}. Available {', '.join(allocations.keys())}"
    
    recommended = monthly_income * percentage

    return f"Recommended {expense_category} budget: ${recommended:,.2f}/month ({percentage*100:.0f}% of income)"

@tool
def get_personalized_greeting(runtime: ToolRuntime[UserContext]) -> str:
    """
        Get a personalized greeting for the user. No arguments required
    """

    name = runtime.context.user_name
    tier = runtime.context.membership_tier

    tier_benefits = {
        "basic": "You have access to standard features.",
        "premium": "As a premium member, you get priority support and advanced analytics!",
        "platinum": "Welcome, platinum member! You have access to all features including personal advisor consultations.",
    }

    benefit_msg = tier_benefits.get(tier, "")
    return f"Hello, {name}! {benefit_msg}"

@tool
def transfer_money(
    from_account: str,
    to_account: str,
    amount: float,
    runtime: ToolRuntime[UserContext]
) -> str:
    """
        Transfer money between accounts

        Args:
            from_account: Source account ('checking', 'savings', 'investment')
            to_account: Destination account ('checking', 'savings', 'investment')
            amount: Amount to transfer (must be positive)
    """

    if amount <= 0:
        raise ValueError("Transfer amount must be positive")
    
    if amount > 10000:
        raise ValueError("Transfer amount exceeds the daily limit of $10,000")
    
    if from_account.lower() == to_account.lower():
        raise ValueError("Cannot transfer to the same account")
    

    user_id = runtime.context.user_id
    user_data = USER_DATABASE.get(user_id, {})
    accounts = user_data.get("accounts", {})

    from_balance = accounts.get(from_account.lower())

    if from_balance is None:
        raise ValueError(f"Source account '{from_account}' not found")
    
    if to_account.lower() not in accounts:
        raise ValueError(f"Destination account '{to_account}' not found")
    
    if from_balance < amount:
        raise ValueError(f"Insufficient funds. {from_account} balance: ${from_balance: .2f}")
    
    # Simulate successful transfer
    return f"✅ Successfully transferred ${amount: .2f} from {from_account} to {to_account}"


"""
Define Middlewares
"""
@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler) -> ModelResponse:
    """ 
        Selects model based on user's membership tier
    """

    tier = request.runtime.context.membership_tier

    if tier == "platinum":
        request.override(model=platinum_model)
        print(f"[Middleware] Using PLATINUM model (gpt-4o, limitless)")
    elif tier == "premium":
        request.override(model=premium_model)
        print(f"[Middleware] Using PREMIUM model (gpt-4o, 2048 tokens)")

    else:
        request.override(model=basic_model)
        print(f"[Middleware] Using BASIC model (gpt-4o-mini, 512 token)")

    return handler(request)

@dynamic_prompt
def tier_based_prompt(request: ModelRequest) -> str:
    """
        Generate system prompt based on user's membership tier
    """

    tier = request.runtime.context.membership_tier
    user_name = request.runtime.context.user_name

    base_prompt = f"""You are a personal finance assistant helping {user_name}.

            Your capabilities:
            - Check account balances (checking, savings, investment)
            - View recent transactions
            - Calculate budget recommendations
            - Provide personalized greetings"""
		
    if tier == "premium":
        return base_prompt + """

            PREMIUM MEMBER BENEFITS:
            - Provide helpful explanations with your responses
            - Offer occasional tips for financial improvement
            - Be friendly and informative
            - Balance detail with brevity"""

    elif tier == "platinum":
        return base_prompt + """

            PLATINUM MEMBER BENEFITS:
            - Provide detailed, comprehensive financial analysis
            - Offer proactive suggestions for wealth growth
            - Include market insights when relevant
            - Be thorough and consultative in your responses
            - Take extra time to explain complex concepts"""

    else:  # basic
        return base_prompt + """

            Guidelines:
            - Be concise and direct
            - Answer questions efficiently
            - Focus on the specific request
            - Keep responses brief but helpful"""
    
@wrap_tool_call
def handle_tool_errors(request: ToolCallRequest, handler) -> ToolMessage:
    """
        Gracefully handles tool execution errors
    """

    tool_name = request.tool_call['name']

    try:
        # Attempt to execute the tool
        return handler(request)

    except ValueError as e:
        error_message = f"⚠️ {tool_name} failed: {str(e)}"
        print(f"[Error Handler] Caught ValueError: {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"]
        )
    
    except KeyError as e:
        error_message = f"⚠️ {tool_name} error: Required data not found - {str(e)}"
        print(f"[Error Handler] Caught KeyError: {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"]
        )
    
    except Exception as e:
        error_message = f"⚠️ {tool_name} encountered an error. Please try again or contact support"
        print(f"[Error Handler] Caught Unexpected error: {type(e).__name__}- {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"]
        )
    


SYSTEM_PROMPT = """You are a helpful personal finance assistant.

Your capabilities:
- Check account balances (checking, savings, investment)
- View recent transactions
- Calculate budget recommendations
- Provide personalized greetings

Guidelines:
- Be helpful and informative
- Always start by greeting the user 
- Provide clear, actionable advice
- Use tools to get accurate, user-specific information
- Format monetary values clearly
- Tailor advice based on the user's membership tier"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=basic_model,
    tools=[
        get_account_balance,
        get_recent_transactions,
        calculate_budget,
        get_personalized_greeting,
        transfer_money
    ],
    #system_prompt=SYSTEM_PROMPT,
    context_schema=UserContext,
    middleware=[
        dynamic_model_selector,
        tier_based_prompt,
        handle_tool_errors,
    ],
    response_format=ToolStrategy(FinancialResponse),
    checkpointer=checkpointer
)

agent_for_streaming = create_agent(
    model=basic_model,
    tools=[
        get_account_balance,
        get_recent_transactions,
        calculate_budget,
        get_personalized_greeting,
        transfer_money
    ],
    #system_prompt=SYSTEM_PROMPT,
    context_schema=UserContext,
    middleware=[
        dynamic_model_selector,
        tier_based_prompt,
        handle_tool_errors,
    ],
    checkpointer=checkpointer
)

def main():
    print("=" * 60)
    print("Stage 1: Simple Finance Assistant")
    print("=" * 60)

    alice_context = UserContext(
        user_id="user_001",
        user_name="Alice Johnson",
        membership_tier="platinum",
        preferred_currency="USD"
    )

    bob_context = UserContext(
        user_id="user_002",
        user_name="Bob Smith",
        membership_tier="basic",
        preferred_currency="EUR"
    )

    """ # Test 1: Check balance
    balance_message = "What's my checking account balance?"

    print(f"\nQuery: {balance_message}")
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content":balance_message }]
        },
        context=alice_context
    )

    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 2: Multi-tool query
    multi_tool_prompt = "Show me my savings balance and recent transactions"
    print(f"\n📝 Query: {multi_tool_prompt}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": multi_tool_prompt}]},
        context=bob_context
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 3: Budget calculation
    budget_prompt = "I make $5000/month. How much should I spend on housing?"
    print(f"\n📝 Query: {budget_prompt}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": budget_prompt}]}
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """


    """ # Test 4: Financial situation and advice
    financial_situation_query = "What's my financial situation? Check all my accounts and give me advice"
    print("\n👥 Same query, different treatment")
    print("-"* 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": financial_situation_query}]},
        context=bob_context
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 5: successful transfer
    successful_transfer_prompt = "Transfer $500 from checking to savings"
    print(f"\n📝 Query: '{successful_transfer_prompt}'")
    print("-" * 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": successful_transfer_prompt}]},
        context=alice_context,
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 6: Error handling - insufficient funds
    insufficient_amount_prompt = "Transfer $5000 from checking to savings"
    print(f"\n📝 Query: '{insufficient_amount_prompt}' (should fail)")
    print("-" * 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": insufficient_amount_prompt}]},
        context=alice_context,
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 7: Error handling - same account
    print("\n📝 Query: 'Transfer $100 from checking to checking' (should fail)")
    print("-" * 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Transfer $100 from checking to checking"}]},
        context=bob_context,
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 8: Structured Response
    query = "What's my financial situation? Check all my accounts and give me advice"
    print("\n👥 Alice - Financial Breakdown")
    print("-"* 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        context=alice_context
    )
    
    structured: FinancialResponse = response["structured_response"]

    print("\nSTRUCTURED RESPONSE")
    print(f"\n Summary:\n {structured.summary}")
    print(f"\n Details:\n {structured.details}")

    print("\n✅ Action Items:")
    for item in structured.action_items:
        print(f"* {item}")

    print("\n⚠️ Warnings:")
    for warning in structured.warnings:
        print(f"* {warning}")

    print(f"\n Confidence: {structured.confidence}") """

    """ # Test 9: Memory Test
    memory_config = {"configurable": {"thread_id": "alice-memory-test"}}

    # Turn 1
    turn1_message = "What are my account balances?"
    print(f"\n💬 Turn 1: {turn1_message}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": turn1_message}]},
        context=alice_context,
        config=memory_config
    )
    print(f"🤖 Agent: {response['structured_response'].details}")

    # Turn 2
    turn2_message = "Which account has the most money?"
    print(f"\n💬 Turn 2: {turn2_message}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": turn2_message}]},
        context=alice_context,
        config=memory_config
    )
    print(f"🤖 Agent: {response['structured_response'].summary}")

    # Turn 3
    turn3_message = "Based on what we discussed, should I move money to my savings?"
    print(f"\n💬 Turn 3: {turn3_message}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": turn3_message}]},
        context=alice_context,
        config=memory_config
    )
    print(f"🤖 Agent Summary: {response['structured_response'].summary}")
    print(f"🤖 Agent Recommendations: {response['structured_response'].action_items}") """

    # Test 10: Streaming
    bob_config = {"configurable": {"thread_id": "bob-002"}}

    query = "What's my financial situation? Check all my accounts and give me advice"

    """ for chunk in agent_for_streaming.stream(
        {"messages": [
            {"role": "user", "content": query }
        ]},
        context=bob_context,
        config=bob_config, 
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            print(f"Step: {step}")
            print(f"Content: {data['messages'][-1].content_blocks}") """
    
    for chunk in agent_for_streaming.stream(
        {"messages": [
            {"role": "user", "content": query }
        ]},
        context=bob_context,
        config=bob_config, 
        stream_mode="values"
    ):
        lastest_message = chunk['messages'][-1]

        if lastest_message.content:
            print(f"Agent: {lastest_message.content}")
        elif lastest_message.tool_calls:
            print(f"Calling tools: {[tool_call['name'] for tool_call in lastest_message.tool_calls]}")



if __name__ == "__main__":
    main()