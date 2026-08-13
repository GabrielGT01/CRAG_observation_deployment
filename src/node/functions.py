
"""Chains and tools shared across the CRAG graph's nodes."""

import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

load_dotenv()  # pulls OPENAI_API_KEY / TAVILY_API_KEY from .env



# Helper: squash a list of Document objects into one context string

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# Document relevance grader
class GradeDocuments(BaseModel):
    """Grade whether a retrieved document actually answers the user's question —
    not just whether it shares a topic or keywords with it."""

    binary_score: str = Field(
        description="Whether the document is relevant to the question: 'yes' or 'no'"
    )


grader_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,  # deterministic yes/no grading
)
structured_llm_grader = grader_llm.with_structured_output(GradeDocuments)

grader_system = """You are a grader assessing whether a retrieved document actually answers a user's \
question — not just whether it shares keywords or topic with it.

Grade 'yes' only if the document contains the specific information needed to answer the question. \
Grade 'no' if the document is only topically related (same general subject, overlapping terms or \
numbers) but doesn't contain the actual answer, or if it describes a different specific policy that \
happens to use similar language.

A document should never be marked 'yes' just because it mentions the same category of thing \
(e.g. a dollar amount, a time window, a department name) as the question — check that it's the same \
policy, not just the same shape of fact."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader



# Answer generation chain
generation_system_prompt = """
You are an assistant for question-answering tasks.

Use the following retrieved context and conversation history to answer the question.
If the answer is not contained in the context, say that you don't know.
Use no more than three sentences and keep the answer concise.

Chat history:
{chat_history}

Context:
{context}
"""

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generation_system_prompt),
        ("human", "{question}"),
    ]
)

# Separate LLM instance from the grader's — generation doesn't need structured output
generation_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = generation_prompt | generation_llm | StrOutputParser()



# Query rewriter (used before falling back to web search)

rewriter_system = """You a question re-writer that converts an input question to a better version that \
is optimized for web search. Look at the input and try to reason about the underlying semantic \
intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewriter_system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

rewriter_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
question_rewriter = re_write_prompt | rewriter_llm | StrOutputParser()



# Web search fallback tool
web_search_tool = TavilySearch(k=3)


