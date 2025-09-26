import streamlit as st
from openai import OpenAI

client = OpenAI()

st.set_page_config(page_title="Hello INFO 5940", layout="centered")

st.title("Let's Cook French Cuisine!")

system_prompt = """
    You are an expert at suggesting french cuisine recipes. Be encouraging and informative to help
    people learn more about cooking french dishes. Keep you responses concise, do not use emojis, and
    do not help with any other topics.
    If you suggest a recipe use this structure:
    1. Name of the dish
    2. Ingredients
    3. Cooking instructionsz
    """
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hello, what are you interested in?"}
        ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(model="openai.gpt-4o",
                                                messages=st.session_state.messages,
                                                stream=True)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})