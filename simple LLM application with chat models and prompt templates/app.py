import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Gemini Chat App", page_icon="🤖")
st.title("🤖 Gemini Chat with Streamlit")

# 1) Get API key from Streamlit Secrets (preferred) or environment
API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# 2) Choose model (you can change this)
MODEL_NAME = st.sidebar.text_input("Model name", value="gemini-1.5-flash")

# 3) Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4) Render history so far
for role, content in st.session_state.messages:
    st.chat_message(role).write(content)

# 5) Require API key
if not API_KEY:
    st.info("Add your GOOGLE_API_KEY in Streamlit Cloud → Settings → Secrets (or set it locally).")
    st.stop()

# 6) Init model (using the key from secrets/env)
model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY)

# 7) Simple system+human prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

# 8) Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # show user message immediately
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # run chain (prompt → model)
    chain = prompt | model
    response = chain.invoke({"user_input": user_input})
    reply = response.content

    # show assistant reply
    st.session_state.messages.append(("assistant", reply))
    st.chat_message("assistant").write(reply)
