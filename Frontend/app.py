import streamlit as st
from rag_functionality import rag_func

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant" , "content": "Hello, I am here to assist you with any queries related to the Mines and Minerals (Development and Regulation) Act. Please feel free to ask your questions, and I am here to help you."}
    ]
    
    
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user" , "content": user_prompt})
    
    
    with st.chat_message("user"):
        st.write(user_prompt)
        
if st.session_state.message[-1]["role"]!="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = rag_func(user_prompt)
            st.write(ai_response)
            
    
    new_ai_message = {"role": "assistant" , "content": ai_response}
    st.session_state.messages.append(new_ai_message)