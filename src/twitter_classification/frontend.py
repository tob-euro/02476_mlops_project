import streamlit as st
import requests

BACKEND_URL = "https://twitter-classification-api-791862686266.europe-west1.run.app"  

def get_prediction(text):
    """Send text to the backend and return prediction results."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    return None

def main():
    """Streamlit app for Twitter Disaster Classification."""
    st.title("Twitter Disaster Classification")
    st.write(
        "This app uses a machine learning model to classify tweets "
        "as disaster-related or not disaster-related."
    )

    # Input text
    text = st.text_area("Enter a tweet for classification:", height=150)

    if st.button("Classify"):
        if text.strip():
            result = get_prediction(text)
            if result:
                st.write("### Prediction Results:")
                st.write(f"**Label:** {'Disaster' if result['label'] == 1 else 'Not Disaster'}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main()


# to run locally: streamlit run src\twitter_classification\frontend.py
