import streamlit as st
from transformers import pipeline
import pandas as pd

# ---- Streamlit page config ----
st.set_page_config(page_title="Shopkeeper AI Chatbot", page_icon="🛒")
st.title("🛍️ Shopkeeper AI Chatbot")
st.write("Talk to your AI shop assistant for free!")

# ---- Load product knowledge ----
products = pd.read_csv("products.csv")
product_info = "\n".join([f"{row['Name']}: {row['Description']} ({row['Price']})" for _, row in products.iterrows()])

# ---- Load a small, free open-source model ----
@st.cache_resource
def load_model():
    # Small model for zero-cost deployment
    chatbot_pipeline = pipeline("text-generation", model="NousResearch/Nous-Hermes-13b-mini", device=-1)
    return chatbot_pipeline

chatbot = load_model()

# ---- Initialize chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- User input ----
user_input = st.text_input("You:", "")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"You are a friendly shopkeeper AI assistant. Here are the available products:\n{product_info}\nCustomer: {user_input}\nShopkeeper:"

    with st.spinner("Thinking..."):
        response = chatbot(prompt, max_length=150, do_sample=True)[0]["generated_text"]

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---- Display chat ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Shopkeeper AI:** {msg['content']}")
