import streamlit as st
import joblib
import pandas as pd

# Load assets (cached so they don't reload on every interaction)
@st.cache_resource
def load_assets():
    try:
        vec = joblib.load('vectorizer.pkl')
        mods = joblib.load('models.pkl')
        return vec, mods
    except FileNotFoundError:
        return None, None

vectorizer, models = load_assets()

# UI Layout
st.title("🛡️ Toxic Comment Classifier")
st.markdown("Enter a comment below to check its toxicity levels.")

if vectorizer is None or models is None:
    st.error("Error: Model files not found. Please run 'train_model.py' first to generate .pkl files.")
else:
    # Text Input
    user_input = st.text_area("Comment Text", placeholder="Type something here...")
    
    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            # Vectorize input
            text_vec = vectorizer.transform([user_input])
            
            # Predict
            results = []
            for label, model in models.items():
                prob = model.predict_proba(text_vec)[0][1]
                results.append({"Label": label, "Probability": prob})
            
            # Display Results
            results_df = pd.DataFrame(results)
            
            # Show as a table with progress bars
            st.subheader("Analysis Results")
            
            # Custom formatting for probability
            st.dataframe(
                results_df.style.background_gradient(cmap='Reds', subset=['Probability'])
                                .format({'Probability': '{:.2%}'})
            )

            # Highlight specific flags
            st.subheader("Flags Triggered")
            triggered = [r['Label'] for r in results if r['Probability'] > 0.5]
            if triggered:
                for t in triggered:
                    st.error(f"⚠️ {t.upper()}")
            else:
                st.success("✅ This comment appears safe.")