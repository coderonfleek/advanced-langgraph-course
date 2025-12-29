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
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from pydantic import BaseModel, Field 
from typing import Literal 
from langchain.agents.structured_output import ToolStrategy

"""
Define Models
"""

basic_model = init_chat_model(
    "gpt-4o-mini",  # Faster, cheaper model for basic users
    temperature=0.5,
    max_tokens=512,
)

premium_model = init_chat_model(
    "gpt-4o",  # Balanced model for premium users
    temperature=0.7,
    max_tokens=1024,
)

platinum_model = init_chat_model(
    "gpt-4o",  # Best model with higher limits for platinum
    temperature=0.7,
)

"""
Define Context Schema
"""
@dataclass
class UserContext:
    """User-specific context injected at runtime."""
    user_id: str
    user_name: str
    membership_tier: str  # 'basic', 'premium', 'platinum'
    preferred_currency: str

"""
Define the structured output
"""
class FinancialResponse(BaseModel):
    """Structured response from the finance assistant."""
    
    summary: str = Field(
        description="A brief summary of the response (1-2 sentences)"
    )
    
    details: str = Field(
        description="Detailed explanation or data"
    )
    
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



"""
Add the Mock User Database
"""
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
def get_account_balance(account_type: str, runtime: ToolRuntime[UserContext]) -> str:
    """Get the current balance for a specific account.
    
    Args:
        account_type: Type of account - 'checking', 'savings', or 'investment'
    """
    user_id = runtime.context.user_id
    currency = runtime.context.preferred_currency
    user_data = USER_DATABASE.get(user_id, {})
    
    balance = user_data.get("accounts", {}).get(account_type.lower())
    
    if balance is not None:
        if currency == "EUR":
            balance = balance * 0.92  # Simplified conversion
            return f"Your {account_type} account balance is €{balance:,.2f}"
        return f"Your {account_type} account balance is ${balance:,.2f}"
    
    return f"Unknown account type: {account_type}. Available: checking, savings, investment"

@tool
def get_recent_transactions(account_type: str, limit: int = 5, runtime: ToolRuntime[UserContext] = None) -> str:
    """Get recent transactions for an account.
    
    Args:
        account_type: Type of account - 'checking', 'savings', or 'investment'
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
    """Calculate recommended budget allocation for an expense category.
    
    Args:
        monthly_income: User's monthly income
        expense_category: Category like 'housing', 'food', 'transportation', 'savings', 'entertainment'
    """
    # Standard budget percentages (50/30/20 rule inspired)
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
        return f"Unknown category: {expense_category}. Available: {', '.join(allocations.keys())}"
    
    recommended = monthly_income * percentage
    return f"Recommended {expense_category} budget: ${recommended:,.2f}/month ({percentage*100:.0f}% of income)"

@tool
def get_personalized_greeting(runtime: ToolRuntime[UserContext]) -> str:
    """Get a personalized greeting for the user. No arguments needed."""
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
    runtime: ToolRuntime[UserContext],
) -> str:
    """Transfer money between accounts.
    
    Args:
        from_account: Source account ('checking', 'savings', 'investment')
        to_account: Destination account ('checking', 'savings', 'investment')
        amount: Amount to transfer (must be positive)
    """
    # Validation - these will raise errors caught by middleware
    if amount <= 0:
        raise ValueError("Transfer amount must be positive")
    
    if amount > 10000:
        raise ValueError("Transfer amount exceeds daily limit of $10,000")
    
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
        raise ValueError(f"Insufficient funds. {from_account} balance: ${from_balance:.2f}")
    
    # Simulate the transfer (in production, this would update the database)
    return f"✓ Successfully transferred ${amount:.2f} from {from_account} to {to_account}"



"""
Middleware
"""
@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler) -> ModelResponse:
    """Select model based on user's membership tier."""
    
    tier = request.runtime.context.membership_tier
    
    if tier == "platinum":
        request.override(model=platinum_model)
        print(f"  [Middleware] Using PLATINUM model (gpt-4o, limitless tokens)")
    elif tier == "premium":
        request.override(model=premium_model)
        print(f"  [Middleware] Using PREMIUM model (gpt-4o, 1024 tokens)")
    else:
        request.override(model=basic_model)
        print(f"  [Middleware] Using BASIC model (gpt-4o-mini, 512 tokens)")
    
    return handler(request)

@dynamic_prompt
def tier_based_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user's membership tier."""
    
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
    """Gracefully handle tool execution errors."""
    tool_name = request.tool_call["name"]
    
    try:
        # Attempt to execute the tool
        return handler(request)
    
    except ValueError as e:
        # Handle validation errors (user-friendly)
        error_message = f"⚠️ {tool_name} failed: {str(e)}"
        print(f"  [Error Handler] Caught ValueError: {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"],
        )
    
    except KeyError as e:
        # Handle missing data errors
        error_message = f"⚠️ {tool_name} error: Required data not found - {str(e)}"
        print(f"  [Error Handler] Caught KeyError: {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"],
        )
    
    except Exception as e:
        # Handle unexpected errors
        error_message = f"⚠️ {tool_name} encountered an unexpected error. Please try again or contact support."
        print(f"  [Error Handler] Caught unexpected error: {type(e).__name__}: {e}")
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"],
        )



"""
Define System Prompt for the agent
"""

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

"""
Agent Creation
"""
agent = create_agent(
    # model="gpt-4o",
    model= basic_model, # MODIFIED FOR STAGE 3
    tools=[
        get_account_balance, 
        get_recent_transactions, 
        calculate_budget,
        get_personalized_greeting, # ===== STAGE 2: NEW TOOL ADDED =====
        transfer_money, # ===== STAGE 4: NEW TOOL =====
    ],
    #system_prompt=SYSTEM_PROMPT, # MODIFIED FOR STAGE 3
    context_schema=UserContext, # ===== STAGE 2: NEW PARAMETER =====
    response_format= ToolStrategy(FinancialResponse), 
    middleware=[  # ===== STAGE 3: NEW PARAMETER =====
        dynamic_model_selector,
        tier_based_prompt,
        handle_tool_errors, # ===== STAGE 4: NEW MIDDLEWARE =====
    ], 
)

"""
Test the Agent
"""

def main():
    print("=" * 60)
    print("Stage 1: Simple Finance Assistant")
    print("=" * 60)

    # ===== STAGE 2: NEW ADDITION - Create user contexts =====
    alice_context = UserContext(
        user_id="user_001",
        user_name="Alice Johnson",
        membership_tier="platinum",
        preferred_currency="USD",
    )
    
    bob_context = UserContext(
        user_id="user_002",
        user_name="Bob Smith",
        membership_tier="basic",
        preferred_currency="EUR",
    )

    """ # Test 1: Check balance
    balance_message = "What's my checking account balance?"
    print(f"\n📝 Query: {balance_message}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": balance_message}]},
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
        {"messages": [{"role": "user", "content": budget_prompt}]},
        context=alice_context
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    """ # Test 4: Financial Situation and advice
    financial_situation_query = "What's my financial situation? Check all my accounts and give me advice."
    
    # Test with Alice (Platinum - gets best model + detailed prompt)
    print("\n👤 Same query, different treatment")
    print("-" * 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": financial_situation_query}]},
        context=bob_context,
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
        context=alice_context,
    )
    print(f"🤖 Agent: {response['messages'][-1].content}") """

    # Test 8: Structured Response
    query = "What's my financial situation? Check all my accounts and give me advice."
    
    print("\n👤 Alice (Platinum) - Financial Condition")
    print("-" * 40)
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        context=alice_context,
    )

    # Access the structured response (Pydantic model)
    structured: FinancialResponse = response["structured_response"]
    
    print("\n📊 STRUCTURED RESPONSE (Pydantic Model):")
    print(f"\n📌 Summary:\n   {structured.summary}")
    print(f"\n📋 Details:\n   {structured.details}")
    print(f"\n✅ Action Items:")
    for item in structured.action_items:
        print(f"   • {item}")
    print(f"\n⚠️ Warnings:")
    for warning in structured.warnings:
        print(f"   • {warning}")
    print(f"\n🎯 Confidence: {structured.confidence}")


if __name__ == "__main__":
    main()
