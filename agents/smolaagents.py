"""
Simple Smolagents Agent - Weather Assistant
A basic demonstration of how to create a smolagents ToolCallingAgent
"""

from smolagents import ToolCallingAgent, LiteLLMModel, tool
import random


MODEL = "ollama/llama3.2:latest"


# ========== DEFINE TOOLS ==========

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The name of the city
    
    Returns:
        A string describing the current weather
    """
    weather_conditions = ["sunny", "rainy", "cloudy", "snowy", "windy"]
    temperatures = [15, 20, 25, 30, 5, 10]
    
    condition = random.choice(weather_conditions)
    temp = random.choice(temperatures)
    
    return f"The weather in {city} is {condition} with a temperature of {temp}°C"


@tool
def get_temperature(city: str) -> int:
    """
    Get the temperature for a city.
    
    Args:
        city: The name of the city
    
    Returns:
        The temperature in Celsius
    """
    return random.randint(-10, 40)


@tool
def get_humidity(city: str) -> int:
    """
    Get the humidity level for a city.
    
    Args:
        city: The name of the city
    
    Returns:
        The humidity percentage (0-100)
    """
    return random.randint(30, 90)


# ========== INITIALIZE MODEL & AGENT ==========

def create_weather_agent():
    """
    Create and return a weather assistant agent.
    
    Returns:
        A ToolCallingAgent configured with weather tools
    """
    # Initialize the LLM model
    llm = LiteLLMModel(
        model_id=MODEL,
        api_base="http://localhost:11434"
    )
    
    # Create agent with weather tools
    agent = ToolCallingAgent(
        tools=[get_weather, get_temperature, get_humidity],
        model=llm,
    )
    
    return agent


# ========== MAIN DEMO ==========

def main():
    """Run the weather assistant agent."""
    
    print("\n" + "="*70)
    print("SMOLAGENTS - Simple Weather Assistant Agent")
    print("="*70)
    
    print("\n⏳ Initializing agent...")
    agent = create_weather_agent()
    print("✓ Agent ready!\n")
    
    # Example queries
    queries = [
        "What's the weather like in Paris?",
        "Tell me about the weather in Tokyo and its humidity",
        "Is it hot in New York today?",
        "Compare the temperature in London and Berlin"
    ]
    
    print("="*70)
    print("Example Queries:")
    print("="*70)
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "-"*70)
    print("Or enter your own query:")
    print("-"*70)
    
    user_query = input("\n🗣️  Enter your weather query: ").strip()
    
    if not user_query:
        user_query = queries[0]
        print(f"\n[Using default query]: {user_query}")
    
    print(f"\n📝 Query: {user_query}")
    print("\n⏳ Agent is processing... (this takes 1-2 minutes)\n")
    print("-"*70)
    
    try:
        response = agent.run(user_query)
        print("\n" + "="*70)
        print("🤖 AGENT RESPONSE:")
        print("="*70)
        print(f"\n{response}\n")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
