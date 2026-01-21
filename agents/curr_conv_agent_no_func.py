"""
Currency Conversion Agent using smolagents with CodeAgent and LiteLLMModel.
This agent can fetch live exchange rates and perform currency conversions.
"""

import requests
from smolagents import CodeAgent, LiteLLMModel, tool

# =============================================================================
# SMOLAGENTS TOOLS
# =============================================================================

@tool
def fetch_live_rate(from_currency: str, to_currency: str) -> float:
    """
    Retrieve a live exchange rate from a public exchange rate API.

    Args:
        from_currency (str): 3-letter source currency code, e.g. "USD"
        to_currency (str): 3-letter target currency code, e.g. "EUR"

    Returns:
        float: Exchange rate (target units per one source unit)

    Raises:
        RuntimeError: if the rate cannot be fetched.
    """
    base = from_currency.lower()
    target = to_currency.lower()
    
    # Try multiple API endpoints for reliability
    urls = [
        f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base}.json",
        f"https://latest.currency-api.pages.dev/v1/currencies/{base}.json",
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            rates = data.get(base, {})
            
            if target in rates:
                return float(rates[target])
        except Exception:
            continue
    
    raise RuntimeError(
        f"Failed to fetch exchange rate from {from_currency} to {to_currency}"
    )


@tool
def calculate(expression: str) -> float:
    """
    Evaluate a basic arithmetic expression safely.

    Args:
        expression (str): e.g. "100 * 0.85"

    Returns:
        float: numeric result

    Raises:
        RuntimeError: if the expression is invalid.
    """
    try:
        # Safe eval with restricted builtins
        result = eval(expression, {"__builtins__": {}})
        return float(result)
    except Exception as e:
        raise RuntimeError(f"Calculation error: {e}")


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

def create_agent(model_id: str = "gpt-4o-mini", api_base: str = None):
    """
    Create and configure the currency conversion agent.

    Args:
        model_id (str): LiteLLM model identifier (default: "gpt-4o-mini")
        api_base (str): API base URL for LiteLLM (default: None for OpenAI)

    Returns:
        CodeAgent: Configured agent instance
    """
    model = LiteLLMModel(
        model_id=model_id,
        api_base=api_base,
        temperature=0.0,  # deterministic tool use
    )

    agent = CodeAgent(
        tools=[fetch_live_rate, calculate],
        model=model,
        max_steps=10,
        add_base_tools=False,
    )

    return agent


# =============================================================================
# DEMONSTRATION QUERIES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Currency Conversion Agent - smolagents + CodeAgent + LiteLLMModel")
    print("=" * 70)

    # Create the agent
    agent = create_agent()

    # Example 1: Simple exchange rate query
    print("\n[Example 1] Get current exchange rate")
    print("-" * 70)
    query1 = "What is the exchange rate from USD to EUR?"
    print(f"Query: {query1}")
    try:
        result1 = agent.run(query1)
        print(f"Result: {result1}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: Convert a specific amount
    print("[Example 2] Convert a specific amount")
    print("-" * 70)
    query2 = "Convert 100 USD to EUR"
    print(f"Query: {query2}")
    try:
        result2 = agent.run(query2)
        print(f"Result: {result2}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Multiple currency conversions
    print("[Example 3] Convert to multiple currencies")
    print("-" * 70)
    query3 = "Convert 1000 USD to EUR, GBP, and JPY. Which gives the highest value?"
    print(f"Query: {query3}")
    try:
        result3 = agent.run(query3)
        print(f"Result: {result3}\n")
    except Exception as e:
        print(f"Error: {e}\n")
