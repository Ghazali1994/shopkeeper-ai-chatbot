# app.py
import streamlit as st
from transformers import pipeline

# ---- Streamlit page setup ----
st.set_page_config(page_title="Shopkeeper AI Chatbot", page_icon="🛒")
st.title("🛍️ Shopkeeper AI Chatbot")
st.write("Chat with your AI shop assistant for free!")

# ---- Load free open-source model ----
@st.cache_resource
def load_model():
    chatbot_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)
    return chatbot_pipeline

chatbot = load_model()

# ---- Product knowledge ----
products = [
    {"name": "Red T-Shirt", "desc": "Cotton, comfortable", "price": "$25"},
    {"name": "Sneakers", "desc": "Lightweight and stylish", "price": "$80"},
]
product_info = "\n".join([f"{p['name']}: {p['desc']} ({p['price']})" for p in products])

# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- User input ----
user_input = st.text_input("You:", "")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"You are a helpful shopkeeper AI. Products available:\n{product_info}\nCustomer: {user_input}\nShopkeeper:"

    with st.spinner("Thinking..."):
        response = chatbot(prompt, max_length=150, do_sample=True)[0]["generated_text"]

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---- Display chat ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Shopkeeper AI:** {msg['content']}")
