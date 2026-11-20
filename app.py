import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Company Policy RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# CUSTOM TITLE
st.title("🤖 Company Policy Assistant")

st.caption(
    "Ask questions about your company policy documents."
)

st.divider()



# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []




# SIDEBAR
with st.sidebar:
    st.header("Settings")

    st.write("Company Policy RAG Assistant")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.info(
        "Powered by LangChain, FAISS and Groq."
    )





# DISPLAY PREVIOUS MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# USER INPUT
user_input = st.chat_input(
    "Ask something about your company policy..."
)


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = rag_chain.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    }
                )

                # Handle string or dictionary output
                if isinstance(response, dict):
                    answer = (
                        response.get("answer")
                        or response.get("output")
                        or response.get("response")
                        or str(response)
                    )
                else:
                    answer = str(response)

                st.markdown(answer)

                # Update conversation history
                st.session_state.chat_history.extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content=answer)
                    ]
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Something went wrong: {e}")