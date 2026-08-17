import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag.auth import Auth
from rag.chat_store import ChatStore
from rag.document_service import DocumentService
from rag.keyword_search import KeywordRetriever
from rag.pipeline import generate_rag_answer
from rag.supabase_store import SupabaseVectorStore


load_dotenv()

st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="🤖",
    layout="wide"
)

auth = Auth()

if "access_token" in st.session_state and "refresh_token" in st.session_state:
    try:
        auth.restore_session(
            st.session_state.access_token,
            st.session_state.refresh_token
        )
    except Exception:
        st.session_state.clear()
        st.warning("Your session has expired. Please log in again.")
        st.rerun()

if "user" not in st.session_state:
    st.session_state.user = None


if st.session_state.user is None:
    st.title("🤖 Technical Documentation Assistant")
    st.caption("Sign in to access your technical documentation assistant.")

    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                try:
                    response = auth.sign_in(email, password)

                    if response.user and response.session:
                        st.session_state.user = response.user
                        st.session_state.access_token = (
                            response.session.access_token
                        )
                        st.session_state.refresh_token = (
                            response.session.refresh_token
                        )

                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed.")

                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tab_signup:
        signup_email = st.text_input(
            "Email",
            key="signup_email"
        )
        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )
        signup_password_confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_password_confirm"
        )

        if st.button("Create Account", use_container_width=True):
            if not signup_email or not signup_password:
                st.error("Please fill in all fields.")
            elif signup_password != signup_password_confirm:
                st.error("Passwords do not match.")
            elif len(signup_password) < 6:
                st.error("Password must contain at least 6 characters.")
            else:
                try:
                    response = auth.sign_up(
                        signup_email,
                        signup_password
                    )

                    if response.user:
                        if response.session:
                            st.session_state.user = response.user
                            st.session_state.access_token = (
                                response.session.access_token
                            )
                            st.session_state.refresh_token = (
                                response.session.refresh_token
                            )

                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.success(
                                "Account created. Please check your email "
                                "to confirm your account, then log in."
                            )
                    else:
                        st.error("Could not create account.")

                except Exception as e:
                    st.error(f"Registration failed: {e}")

    st.stop()


user = st.session_state.user
user_id = user.id


@st.cache_resource
def load_resources():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    store = SupabaseVectorStore()
    document_service = DocumentService(client, store)

    return client, store, document_service


client, store, document_service = load_resources()

chat_store = ChatStore(user_id, auth.client)

keyword_retriever = KeywordRetriever(
    store.get_chunks_for_keyword_search()
)


if "chat_id" not in st.session_state:
    chats = chat_store.get_chats()

    if chats:
        st.session_state.chat_id = chats[0]["id"]
    else:
        st.session_state.chat_id = chat_store.create_chat()


with st.sidebar:
    st.title("💬 Chats")
    st.caption(f"Logged in as: {user.email}")

    if st.button("🚪 Logout", use_container_width=True):
        try:
            auth.sign_out()
        except Exception:
            pass

        st.session_state.clear()
        st.rerun()

    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.chat_id = chat_store.create_chat()
        st.rerun()

    st.divider()

    st.subheader("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    result = document_service.process_pdf(uploaded_file)

                    st.success(
                        f"Stored {result['filename']} "
                        f"with {result['chunks']} chunks."
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Document processing failed: {e}")

    st.divider()

    st.subheader("Previous Chats")

    chats = chat_store.get_chats()

    if not chats:
        st.caption("No previous chats.")

    for chat in chats:
        if chat["id"] == st.session_state.chat_id:
            label = "🟢 " + chat["title"]
        else:
            label = "💬 " + chat["title"]

        if st.button(
            label,
            key=f"chat_{chat['id']}",
            use_container_width=True
        ):
            st.session_state.chat_id = chat["id"]
            st.rerun()

    st.divider()

    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        chat_store.delete_chat(st.session_state.chat_id)

        chats = chat_store.get_chats()

        if chats:
            st.session_state.chat_id = chats[0]["id"]
        else:
            st.session_state.chat_id = chat_store.create_chat()

        st.rerun()


st.title("🤖 Technical Documentation Assistant")
st.caption("Ask questions about your technical documents.")

messages = chat_store.get_messages(
    st.session_state.chat_id
)

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question...")

if question:
    chat_store.add_message(
        st.session_state.chat_id,
        "user",
        question
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            try:
                answer, results = generate_rag_answer(
                    client,
                    question,
                    store,
                    keyword_retriever,
                    top_k=3
                )
            except Exception as e:
                st.error(f"RAG error: {e}")
                answer = "Sorry, I could not process your question."
                results = []

        st.markdown(answer)

        if results:
            with st.expander("📚 Sources"):
                for i, result in enumerate(results, 1):
                    st.markdown(f"**Source {i}**")
                    st.caption(
                        f"Chunk: {result['chunk_id']} | "
                        f"Score: {result['score']:.4f}"
                    )
                    st.write(result["chunk"])

                    if i < len(results):
                        st.divider()

    chat_store.add_message(
        st.session_state.chat_id,
        "assistant",
        answer
    )

    current_chats = chat_store.get_chats()

    current_chat = next(
        (
            chat
            for chat in current_chats
            if chat["id"] == st.session_state.chat_id
        ),
        None
    )

    if current_chat and current_chat["title"] == "New Chat":
        title = question[:40]

        if len(question) > 40:
            title += "..."

        chat_store.update_title(
            st.session_state.chat_id,
            title
        )

    st.rerun()