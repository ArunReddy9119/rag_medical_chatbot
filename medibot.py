import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt():
    # Mimic OpenAI's clear, instruction-following prompt style
    custom_prompt_template = """
    You are a helpful assistant. Use the provided context to answer the question accurately and concisely.
    If the context doesn't contain the answer, say "I don't have enough information to answer this."
    Do not generate answers outside the given context.

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",  # Explicitly set task
            temperature=0.7,  # Adjusted for more natural responses
            max_new_tokens=512,  # Use max_new_tokens instead of max_length
            huggingfacehub_api_token=hf_token
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def main():
    st.title("Chatbot Powered by Mistral-7B")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # User input
    prompt = st.chat_input("Ask me anything!")

    if prompt:
        # Add user message to chat history
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Load configurations
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.getenv("HF_TOKEN")

        if not HF_TOKEN:
            st.error("Hugging Face API token not found. Please set HF_TOKEN in your .env file.")
            return

        try:
            # Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return

            # Load LLM
            llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
            if llm is None:
                return

            # Set up QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            # Get response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"].strip()
            source_documents = response["source_documents"]

            # Format response like OpenAI: clean and direct
            if not result:
                result = "I don't have enough information to answer this."
            
            # Optionally show source documents (comment out if not needed)
            # result_to_show = f"{result}\n\n**Source Documents:**\n{source_documents}"
            result_to_show = result  # Show only the answer for a cleaner UI

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()