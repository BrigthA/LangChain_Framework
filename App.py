import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# 2. Validate API keys
# ---------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing in the .env file.")
    st.stop()

if not os.getenv("LANGSMITH_API_KEY"):
    st.warning(
        "LANGSMITH_API_KEY is missing. "
        "LangSmith tracing will not work."
    )


# ---------------------------------------------------------
# 3. Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LangChain LLM Model",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------------------------------
# 4. Application title
# ---------------------------------------------------------

st.title("🤖 LangChain LLM Model")

st.caption(
    "LangChain + OpenAI + LangSmith + Local Vector Storage"
)


# =========================================================
# 5. Local files
# =========================================================

LOG_FILE = "chat_history.txt"
VECTOR_FILE = "vectors.json"


# =========================================================
# 6. Save chat input/output to TXT file
# =========================================================

def save_chat_to_file(user_input, ai_response):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write("\n")
        file.write("=" * 80 + "\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write("=" * 80 + "\n")

        file.write("\nUSER INPUT:\n")
        file.write(user_input)

        file.write("\n\nAI RESPONSE:\n")
        file.write(ai_response)

        file.write("\n")
        file.write("=" * 80 + "\n")


# =========================================================
# 7. Create OpenAI embedding model
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# =========================================================
# 8. Convert text into vector and save locally
# =========================================================

def save_vector_to_file(user_input):

    # Generate embedding vector
    vector = embeddings.embed_query(user_input)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Create vector record
    vector_record = {
        "timestamp": timestamp,
        "text": user_input,
        "dimension": len(vector),
        "vector": vector
    }

    # Read existing vectors
    if os.path.exists(VECTOR_FILE):

        try:

            with open(
                VECTOR_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                vector_list = json.load(file)

        except json.JSONDecodeError:

            vector_list = []

    else:

        vector_list = []


    # Add new vector
    vector_list.append(vector_record)


    # Save updated vector list
    with open(
        VECTOR_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vector_list,
            file,
            indent=2
        )

    return vector


# ---------------------------------------------------------
# 9. Create OpenAI model
# ---------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)


# ---------------------------------------------------------
# 10. Create prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the user's question clearly and accurately.

            If you are unsure, say that you are unsure.
            """
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ---------------------------------------------------------
# 11. Create LangChain chain
# ---------------------------------------------------------

chain = prompt | llm | StrOutputParser()


# ---------------------------------------------------------
# 12. Streamlit input
# ---------------------------------------------------------

question = st.text_area(
    "Ask your question:",
    placeholder="Example: Explain LangChain in simple terms",
    height=120
)


# ---------------------------------------------------------
# 13. Run chain
# ---------------------------------------------------------

if st.button(
    "🚀 Ask AI",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    with st.spinner(
        "Thinking..."
    ):

        try:

            # -------------------------------------------------
            # Generate AI response
            # -------------------------------------------------

            response = chain.invoke(
                {
                    "question": question
                }
            )


            # -------------------------------------------------
            # Display response
            # -------------------------------------------------

            st.subheader(
                "🤖 AI Response"
            )

            st.write(
                response
            )


            # -------------------------------------------------
            # Save conversation
            # -------------------------------------------------

            save_chat_to_file(
                question,
                response
            )


            # -------------------------------------------------
            # Generate and save vector
            # -------------------------------------------------

            vector = save_vector_to_file(
                question
            )


            # -------------------------------------------------
            # Success messages
            # -------------------------------------------------

            st.success(
                "Conversation saved to chat_history.txt"
            )

            st.success(
                f"Vector saved to vectors.json "
                f"({len(vector)} dimensions)"
            )


        except Exception as e:

            st.error(
                f"An error occurred:\n\n{str(e)}"
            )


# =========================================================
# 14. Sidebar
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    st.write(
        f"**Model:** "
        f"{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
    )

    st.write(
        f"**Embedding Model:** "
        f"text-embedding-3-small"
    )

    st.write(
        f"**LangSmith Project:** "
        f"{os.getenv('LANGSMITH_PROJECT', 'Not configured')}"
    )

    st.divider()

    # -----------------------------------------------------
    # Show local files
    # -----------------------------------------------------

    st.subheader(
        "📁 Local Storage"
    )

    st.write(
        f"💬 `{LOG_FILE}`"
    )

    st.write(
        f"🔢 `{VECTOR_FILE}`"
    )

    # -----------------------------------------------------
    # Number of stored vectors
    # -----------------------------------------------------

    if os.path.exists(VECTOR_FILE):

        try:

            with open(
                VECTOR_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                stored_vectors = json.load(file)

            st.write(
                f"**Stored vectors:** "
                f"{len(stored_vectors)}"
            )

        except Exception:

            st.write(
                "**Stored vectors:** 0"
            )

    else:

        st.write(
            "**Stored vectors:** 0"
        )

    st.divider()

    st.info(
        """
        LangSmith tracing is controlled through
        the LANGSMITH_TRACING environment variable.
        """
    )

