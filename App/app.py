# Core Packages
import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import plotly.express as px 

# Load the pre-trained model
pipe_lr = joblib.load(open("models/text_emotion_classifier.pkl", "rb"))

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
    "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", 
    "surprise": "😮"
}

# Main Application
def main():
    st.title("Emotion Classifier App 🎭")
    st.markdown("Welcome to the Emotion Classifier app! Enter text to detect emotions and view predictions.")
    
    # Navigation Menu
    st.sidebar.title("Navigation")
    menu = ["Home", "About"]
    choice = st.sidebar.radio("Select Page", menu)

    if choice == "Home":
        # Text input form for Home page
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Enter text for emotion analysis:")
            submit_text = st.form_submit_button(label="Submit")

        # Display results after form submission
        if submit_text:
            col1, col2 = st.columns(2)

            # Apply prediction function
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            # Display results
            with col1:
                st.write("### Original Text")
                st.info(raw_text)

                st.write("### Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")  # Default to question mark if emotion not found
                st.success(f"{prediction}: {emoji_icon}")
                st.write(f"**Confidence:** {np.max(probability):.2f}")

            # Prediction probability pie chart
            with col2:
                st.write("### Prediction Probability Distribution")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]

                # Pie chart with plotly displaying probability of each emotion
                fig = px.pie(
                    proba_df_clean,
                    values='Probability',
                    names='Emotion',
                    title="Probability Distribution of Emotions",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

    elif choice == "About":
        st.header("About")
        st.markdown("""
            This application predicts emotions based on the text you enter, using a pre-trained machine learning model. 
            The model classifies the text into various emotions, and the application displays the predicted emotion along with its confidence level.
            \n
            **Features:**
            - Text input for emotion analysis
            - Visual representation of emotion probabilities
            \n
            **Built with:**
            - Streamlit for the web interface
            - Scikit-learn for machine learning
            - Plotly for data visualization
            \n
            We hope you find this application useful!
        """)

if __name__ == '__main__':
    main()
