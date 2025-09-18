# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set the title of your web app
st.title("💬 Personal Finance Chatbot")
st.caption("🚀 A simple prototype for your hackathon with IBM Granite AI")

# Load the IBM Granite model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your financial question?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response with IBM Granite model
    with st.spinner("Thinking..."):
        try:
            # Load model and tokenizer
            tokenizer, model = load_model()
            
            # Format the conversation history for the model
            formatted_messages = []
            
            # Add system message to establish context
            system_message = """You are a helpful and friendly personal finance assistant. 
            Your goal is to provide accurate and practical advice on savings, investments, 
            taxes, and budgeting. Keep your responses concise and to the point."""
            
            formatted_messages.append({"role": "system", "content": system_message})
            
            # Add conversation history
            for msg in st.session_state.messages[-6:]:  # Keep last 6 messages for context
                formatted_messages.append(msg)
            
            # Format input using the tokenizer's chat template
            inputs = tokenizer.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Generate response
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract and decode the response
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Fallback response if model fails
            prompt_lower = prompt.lower()
            if "save" in prompt_lower:
                response = "That's a great goal! A good starting point is to try saving 20% of your income each month."
            elif "tax" in prompt_lower:
                response = "Tax rules can be complex. For basic guidance, remember that income from a job is usually taxable. I can help you find resources to learn more."
            elif "invest" in prompt_lower:
                response = "Investing is key to building wealth long-term. Popular starting points are low-cost index funds or ETFs. Remember, all investments carry risk."
            elif "budget" in prompt_lower:
                response = "I can generate a budget summary for you! To do that, I would need some data on your income and expenses first."
            else:
                response = "I'm a simple prototype and still learning about finance. Could you ask me about savings, taxes, investments, or budgets?"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})