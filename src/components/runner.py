from langchain_core.runnables.base import RunnableSerializable

class ChainRunner:
    def __init__(self):
       pass

    def run(self, rag_chain: RunnableSerializable,
            USER_QUERY: str) -> str:
        with open(USER_QUERY, "r", encoding="utf-8") as f:
            query = f.read()
        response = rag_chain.invoke(query)

        return response

