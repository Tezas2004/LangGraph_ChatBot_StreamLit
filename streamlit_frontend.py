import streamlit as st
import time
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict ->
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # add user message to history
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.markdown(user_input)

    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )

    ai_message = response['messages'][-1].content

    # add assistant message to history
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message
    })

    with st.chat_message('assistant'):
        placeholder = st.empty()

        full_response = ""

        for word in ai_message.split():
            full_response += word + " "
            placeholder.markdown(full_response)
            time.sleep(0.05)