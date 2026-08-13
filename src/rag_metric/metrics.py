
import os
from dotenv import load_dotenv
load_dotenv()  # loading all the environment variables

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "TESTING-phase"

from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable


# ============================================================
# Faithfulness: is every claim in the answer backed by context?
# ============================================================

class FaithfulnessGrade(TypedDict):
    # Explanation first -- forces the model to reason before it commits to a verdict.
    explanation: Annotated[str, ..., "Step-by-step reasoning, claim by claim"]
    faithful: Annotated[bool, ..., "True if every claim in the answer is supported "
        "by the retrieved context, False otherwise"]

faithfulness_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
faithfulness_grader = faithfulness_llm.with_structured_output(
    FaithfulnessGrade, method="json_schema", strict=True
)

faithfulness_instructions = """You are grading whether a STUDENT ANSWER is faithful
to the provided CONTEXT.

You will be given a CONTEXT (the retrieved documents) and a STUDENT ANSWER generated
from that context.

Grading criteria:
(1) Break the STUDENT ANSWER down into its individual factual claims.
(2) For each claim, check whether it can be directly inferred from the CONTEXT.
(3) A claim that is not supported by the CONTEXT, or that contradicts it, counts as
    unfaithful -- even if the claim happens to be true in general.
(4) The answer does not need to use every part of the CONTEXT. It only needs to be
    the case that whatever the answer DOES say is backed by the CONTEXT.

Faithfulness:
A faithfulness value of True means every claim in the STUDENT ANSWER is supported
by the CONTEXT.
A faithfulness value of False means at least one claim is unsupported or contradicts
the CONTEXT (a hallucination).

Explain your reasoning step by step, claim by claim, before giving your final answer.
Avoid stating your verdict at the outset."""

faithfulness_prompt = ChatPromptTemplate.from_messages([
    ("system", faithfulness_instructions),
    ("human", "CONTEXT: {context}\n\nSTUDENT ANSWER: {answer}"),
])

@traceable(name="score_faithfulness")
def score_faithfulness(question: str, answer: str, chunks: list[dict]) -> bool:
    """chunks: list of {"content": ...} dicts"""
    context = " ".join(c["content"] for c in chunks)
    chain = faithfulness_prompt | faithfulness_grader
    grade = chain.invoke({"context": context, "answer": answer})
    return grade["faithful"]


# ============================================================
# Answer relevancy: does the answer address the question asked?
# ============================================================

class AnswerRelevancyGrade(TypedDict):
    explanation: Annotated[str, ..., "Step-by-step reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer directly and completely "
        "addresses the question, False otherwise"]

relevancy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
relevancy_grader = relevancy_llm.with_structured_output(
    AnswerRelevancyGrade, method="json_schema", strict=True
)

answer_relevancy_instructions = """You are grading whether a STUDENT ANSWER is
relevant to the QUESTION that was asked.

Grading criteria:
(1) The answer should directly address what was asked, not a related but different
    question.
(2) The answer should not be incomplete -- dodging part of the question the context
    would have supported answering.
(3) The answer should not pad with information disconnected from the question.
(4) Do NOT grade factual correctness here. Only grade whether the response is on-topic
    and reasonably complete relative to the question.

Relevancy:
A relevant value of True means the answer is on-topic, addresses the question
directly, and is reasonably complete.
A relevant value of False means the answer is off-topic, evasive, or leaves out
something the question clearly asked for.

Explain your reasoning step by step before giving your final answer."""

answer_relevancy_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_relevancy_instructions),
    ("human", "QUESTION: {question}\n\nSTUDENT ANSWER: {answer}"),
])

@traceable(name="score_answer_relevancy")
def score_answer_relevancy(question: str, answer: str) -> bool:
    chain = answer_relevancy_prompt | relevancy_grader
    grade = chain.invoke({"question": question, "answer": answer})
    return grade["relevant"]


# ============================================================
# Context precision: are the retrieved chunks actually needed?
# ============================================================

class ContextPrecisionGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for each chunk, in order"]
    chunk_relevance: Annotated[list[bool], ..., "One True/False per retrieved chunk, "
        "in the same order as given, indicating whether that chunk is relevant/necessary "
        "to answer the QUESTION"]

precision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
precision_grader = precision_llm.with_structured_output(
    ContextPrecisionGrade, method="json_schema", strict=True
)

context_precision_instructions = """You are grading the PRECISION of retrieved
context for a RAG system.

You will be given a QUESTION and a numbered list of RETRIEVED CHUNKS returned by the
retriever.

Grading criteria:
(1) For each chunk, decide whether it is actually relevant and necessary for
    answering the QUESTION.
(2) A chunk on the same general topic but not needed to answer THIS question should
    be marked not relevant.
(3) Judge each chunk independently of the others and independently of its retrieval
    rank/position.
(4) If the QUESTION asks about something the source material does not cover, a chunk
    that explicitly states the requested information is absent or not specified
    counts as relevant -- it is the necessary evidence for a correct "not stated"
    answer, not noise.

Return one True/False verdict per chunk, in the same order as given.

Explain your reasoning for each chunk before giving your final verdicts."""

context_precision_prompt = ChatPromptTemplate.from_messages([
    ("system", context_precision_instructions),
    ("human", "QUESTION: {question}\n\nRETRIEVED CHUNKS:\n{chunks}"),
])

@traceable(name="score_context_precision")
def score_context_precision(question: str, chunks: list[dict]) -> float:
    chunks_string = "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks))
    chain = context_precision_prompt | precision_grader
    grade = chain.invoke({"question": question, "chunks": chunks_string})

    # LLM only judges each chunk's relevance; precision@k weighting is computed here,
    # matching RAGAS's actual formula
    flags = grade["chunk_relevance"]
    precisions_at_k, num_relevant = [], 0
    for k, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            num_relevant += 1
            precisions_at_k.append(num_relevant / k)

    return sum(precisions_at_k) / len(precisions_at_k) if precisions_at_k else 0.0


