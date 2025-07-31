from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class ComponentChainer:
    def __init__(self, ollama_model: str,
                        temperature: float,
                        top_k: int,
                        top_p: float):
        self.llm = ChatOllama(model=ollama_model,
                              temperature=temperature,
                              top_k=top_k,
                              top_p=top_p)

    def chain(self, retriever: VectorStoreRetriever,
              SYSTEM_PROMPT: str) -> RetrievalQA:
        with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            prompt = f.read()

        rag_prompt = {"prompt": PromptTemplate.from_template(prompt)}
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                               retriever=retriever,
                                               chain_type_kwargs=rag_prompt,
                                               chain_type='stuff',
                                               return_source_documents=True)
        return qa_chain

