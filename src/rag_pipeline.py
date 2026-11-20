
from src.config import supabase, embeddings
from langchain_core.runnables import RunnableLambda

from src.chains import (
    contextualize_chain,
    document_answer_chain,
    intent_chain,
    history_answer_chain,
    general_chain
)


def retrieve_documents(inputs):

    # 1. Rewrite the question
    standalone_question = contextualize_chain.invoke(
        {
            "input": inputs["input"],
            "chat_history": inputs["chat_history"]
        }
    )

    print("\nRewritten Question:", standalone_question)

    # 2. Generate query embedding
    query_embedding = embeddings.embed_query(standalone_question)

    # 3. Retrieve relevant documents from Supabase
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.0,
            "match_count": 3
        }
    ).execute()

    documents = response.data

    print("\nRetrieved documents:", len(documents))

    # 4. Combine document chunks
    context = "\n\n".join(
        [doc["content"] for doc in documents]
    )

    # 5. Handle empty retrieval
    if not context.strip():
        return "I don't know based on the provided documents."

    # 6. Generate document-based answer
    answer = document_answer_chain.invoke(
        {
            "input": standalone_question,
            "context": context
        }
    )

    return answer


def route_request(inputs):

    # 1. Get user question
    question = inputs["input"]
    history = inputs["chat_history"]

    # 2. Identify intent
    intent = intent_chain.invoke(
        {
            "input": question,
            "chat_history": history
        }
    ).strip().upper()

    print("\nDetected Intent:", intent)

    # 3. Route request
    if intent == "DOCUMENT":
        return retrieve_documents(inputs)

    elif intent == "HISTORY":
        return history_answer_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )

    elif intent == "GENERAL":
        return general_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )

    else:
        return general_chain.invoke(
            {
                "input": question,
                "chat_history": history
            }
        )


# Create RAG chain
rag_chain = RunnableLambda(route_request)
