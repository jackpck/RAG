SYSTEM_PROMPT = """
You are an AI assistant. Answer the question using only the provided context.

Context:
{context}

Question: {question}
Answer:
"""

RERANKER_PROMPT = """
Rate the relevance of the following document to the query on a scale
of 1 (irrelevant) to 10 (highly relevant).

Query: '{0}'

Document:
\"\"\"
{1}
\"\"\"

Respond with only the number
"""

LLMJUDGE_PROMPT = """
Determine if the following response contains the true answer with 0 (not 
contain) and 1 (contain).

True answer:
\"\"\"
{0}
\"\"\"

Response:
\"\"\"
{1}
\"\"\"

Respond with only the number.
"""
