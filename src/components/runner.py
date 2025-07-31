from langchain.chains import RetrievalQA

class ChainRunner:
    def __init__(self):
       pass

    def run(self, qa_chain: RetrievalQA, USER_QUERY: str) -> str:
        with open(USER_QUERY, "r", encoding="utf-8") as f:
            prompt = f.read()
        response = qa_chain.run(prompt)
        return response

