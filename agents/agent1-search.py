from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from duckduckgo_search import DDGS

# 1. 实例化本地模型
llm = OllamaLLM(model="llama3.2", temperature=0)

# 2. 写一个同步搜索函数
def web_search(query: str) -> str:
    return " ".join([r["body"] for r in DDGS().text(query, max_results=3)])

# 3. 包装成 LangChain Tool
search_tool = Tool(
    name="web_search",
    description="Search the web for up-to-date information.",
    func=web_search
)

# 4. 构建 ReAct Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 5. 运行
if __name__ == "__main__":
    print("search agent (type 'exit' to quit)\n")
    while True:
        question = input("whatever (or 'exit'): ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            answer = agent.run(question)
            print(f"\n✓ {answer}\n")
        except Exception as e:
            print(f"⚠️  Error: {e}\n")