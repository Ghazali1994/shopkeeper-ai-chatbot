import streamlit as st
from transformers import pipeline
import pandas as pd

# ---- Streamlit page config ----
st.set_page_config(page_title="Shopkeeper AI Chatbot", page_icon="🛒")
st.title("🛍️ Shopkeeper AI Chatbot")
st.write("Talk to your AI shop assistant for free!")

# ---- Load product knowledge ----
@st.cache_data
def load_products():
    products = pd.read_csv("products.csv")
    return products

products = load_products()
product_info = "\n".join(
    [f"{row['Name']}: {row['Description']} ({row['Price']})" for _, row in products.iterrows()]
)

# ---- Load a small model safely ----
@st.cache_resource
def load_model():
    try:
        # Replace "gpt2" with "NousResearch/Nous-Hermes-13b-mini" if you have token & enough memory
        model_name = "gpt2"
        # If using a private model:
        # model_name = "NousResearch/Nous-Hermes-13b-mini"
        # return pipeline("text-generation", model=model_name, device=-1, use_auth_token="YOUR_HF_TOKEN")
        return pipeline("text-generation", model=model_name, device=-1)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

chatbot = load_model()

# ---- Initialize chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- User input ----
user_input = st.text_input("You:", "")

if user_input and chatbot:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Limit product info to first 1000 chars to prevent huge prompts
    prompt = (
        f"You are a friendly shopkeeper AI assistant. "
        f"Here are the available products:\n{product_info[:1000]}\n"
        f"Customer: {user_input}\nShopkeeper:"
    )

    with st.spinner("Thinking..."):
        try:
            response = chatbot(prompt, max_length=150, do_sample=True)[0]["generated_text"]
        except Exception as e:
            response = f"Error generating response: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---- Display chat ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Shopkeeper AI:** {msg['content']}")
