# RAG

## Introduction

Build a modulized RAG. Components are modulized and linked together
via dynamic chaining. Modular component can be found under ``src/components``.
The ``run()`` function in ``main.py`` chain all the components via the DAG
specified by ``configs/pipeline_config.yaml``


For starter, FAISS is used for the vector database. This does not 
support hybrid search (e.g. BM25 + cosine similarity). If hybrid search
is needed, consider using weaviate.

llama3 (7B) was used as the first LLM. Performance deteriorates as the size of the document
increases. Recommend using gpt-oss (20B) to improve performance.

A two-stage retrieval is used. A reranker is used to enhance retrieval precision.

Citation was also added to provide source of the retrieved document.


## Development history
1. Build simple RAG using Ollama
2. Modulize code using dynamic chaining
3. Add reranker and citation
4. Add evaluation module [TODO]
5. Add prompt template [TODO]

## Tracing using Langsmith
- since not all components used are from langchain (e.g. FAISS vectorstore), the tracable decorate is 
  needed to trace those components
- Langsmith is not showing a chain of flow of the RAG due to the dynamic chaining structure is separating
  components from each other. Use native langchain class to chain together components to enhance tracing
