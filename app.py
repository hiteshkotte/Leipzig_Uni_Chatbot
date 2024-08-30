import streamlit as st
import base64
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from chatbot import get_executor, generate_response
import os

st.set_page_config(
    page_title="BiWi AI Tutor", page_icon="📚", layout="wide"
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        return base64.b64encode(file.read()).decode()

logo_base64 = get_base64_of_bin_file('images/Both_logos.png')

st.markdown(
    f"""
    <style>
    .logo {{
        position: relative;
        top: 0;
        right: 0;
        border: none;
        width: 250px;  /* Adjust the width as needed */
        height: auto; /* This will maintain the aspect ratio */
    }}
    </style>
    <img class="logo" src="data:image/png;base64,{logo_base64}">
    """,
    unsafe_allow_html=True
)

"# 👋 BiWi AI Tutor"

introduction = st.markdown("""
Welcome to the demonstration of our AI Tutor Chatbot. This tool is designed to assist with answering queries related to BiWi course.

Happy Learning! 😊
"""
)

st.markdown("<hr>", unsafe_allow_html=True)

if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets.OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
    st.sidebar.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    os.environ['OPENAI_API_KEY'] = openai_api_key

if "COHERE_API_KEY" in st.secrets:
    cohere_api_key = st.secrets.COHERE_API_KEY
    os.environ['COHERE_API_KEY'] = cohere_api_key
else:
    cohere_api_key = st.sidebar.text_input(
        "Cohere API Key", type="password"
    )
    st.sidebar.write("[Get a Cohere API key](https://dashboard.cohere.com/api-keys)")
    os.environ['COHERE_API_KEY'] = cohere_api_key


if not openai_api_key or not cohere_api_key:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    if not cohere_api_key:
        st.info("Please add your Cohere API key to continue.")
    st.stop()


@st.cache_resource(ttl="1h")
def get_executor_cached():
    agent_executor, conversational_memory = get_executor("All Material")

    return agent_executor, conversational_memory


msgs = StreamlitChatMessageHistory()


types = {"human": "user", "ai": "assistant"}
avatars = {"human": "🧑‍💻", "ai": "😊"}


if prompt := st.chat_input(placeholder="Add your question here ..."):

    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(types[msg.type], avatar=avatars[msg.type]):
            st.write(msg.content)

    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    with st.chat_message("assistant", avatar="😊"):

        agent_executor, conversational_memory = get_executor_cached()

        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False,
                                             collapse_completed_thoughts=False)

        response, explanation, openai_callback = generate_response(prompt, agent_executor, conversational_memory, st_callback)

        msgs.messages = conversational_memory.chat_memory.messages

        st.write(response["output"])
