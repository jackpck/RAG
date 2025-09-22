# RAG

## Introduction

In this repo, I have developed a modulized RAG chatbot. In MVP1, user can ask questions about a topic from the
Wikipedia page. In future MVPs, the plan is to develop a chatbot that reference difference sources. This can 
easily be done due to the highly modulized structure of the code.

**Highlight**: 

the structure of the repo allows for easy customized experimentation, including addition of new components,
  parameter tuning, prompt engineering etc. Using the langchain tech stack provides a seamless connection to the
  langsmith UI for tracing and evaluation capabilities.

## Tech stack
- **Langchain**, specifically **LCEL** (LangChain Expression Language) as the main development framework.
  Modularity is further enhanced using **dynamic chaining**
- **Langsmith** for prompt versioning, tracing, evaluation and experimentation

## Instructions
In MVP1, there are the following components for the RAG under `src/components`
- `loader.py`: make API call to Wikipedia and fetch the article of the given topic. Can expand capability to fetch 
  other document sources
- `chunker.py`: recursively chunk the document. Can expand to other more sophisticated chunking techniques
- `embedder.py`: embedding chunks into the FAISS vector store. Can expand to other vector store such as *weaviate*
  that support hybrid search (cosine similarity + BM25)
- `retriever.py`: retrieve top chunks given the user query
- `reranker.py`: rerank retrieved chunks by their relevancy to the user query using a LLM
- `generator.py`: generate the final answer based on the reranked, retrieved chunks and the user query
 
All components are highly modulized. User specifies the parameters of each component and how they link together
in `configs/pipeline_config.yaml`. The `src/components/runner.py` script call the `chain_from_yaml` method to
build the LCEL chain based on the config yaml file.

All LLM are initialized by the `init_chat_model` method from langchain. Inference is made via making API call
to the selected model (gemini-2.5-flash for MVP1)

## MVP2

- Add abstention
- Add metadata to document


