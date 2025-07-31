from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

class ComponentChainer:
    def __init__(self, ollama_model: str):
        self.llm = ChatOllama(model=ollama_model)

    def chain(self, retriever: VectorStoreRetriever) -> RetrievalQA:
        qa_chain =RetrievalQA.from_chain_type(llm=self.llm,
                                              retriever=retriever)
        return qa_chain

