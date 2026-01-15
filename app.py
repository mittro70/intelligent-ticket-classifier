import streamlit as st
import joblib

# 1. Load the Model
model = joblib.load('quest_model.pkl')

# 2. "Pro Move": Intent Detection Logic
# Instead of High/Low priority, we detect the TYPE of question.
def detect_intent(text):
    text = text.lower()
    trouble_keywords = ['error', 'fail', 'fix', 'broken', 'bug', 'crash', 'problem']
    howto_keywords = ['how', 'install', 'create', 'build', 'tutorial', 'guide']
    
    if any(w in text for w in trouble_keywords):
        return "Troubleshooting", "red"
    elif any(w in text for w in howto_keywords):
        return "How-To / Guide", "blue"
    else:
        return "General Discussion", "green"

# 3. UI Layout
st.set_page_config(page_title="Smart Content Classifier", page_icon="🧠")

st.title("🧠 Google Quest Classifier")
st.markdown("Enter a question title to predict its topic category and intent.")

# User Input
user_input = st.text_area("Enter Question:", placeholder="e.g., How do I install Pandas on Python 3.8?")

if st.button("Analyze"):
    if user_input:
        # Prediction
        category_pred = model.predict([user_input])[0]
        
        # Intent Rule
        intent, color = detect_intent(user_input)

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.info("📂 Predicted Category")
            st.write(f"**{category_pred}**")

        with col2:
            st.info("🔍 Detected Intent")
            if color == "red":
                st.error(f"**{intent}**")
            elif color == "blue":
                st.info(f"**{intent}**")
            else:
                st.success(f"**{intent}**")
    else:
        st.warning("Please type a question first!")
