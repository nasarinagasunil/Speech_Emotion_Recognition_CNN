import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import model_from_json
import pickle
import librosa
import numpy as np
import os

# Set up page configuration
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")

# Load the model and resources
@st.cache_resource
def load_model_and_resources():
    with open(r"C:\Users\laksh\OneDrive\Desktop\jupyternotebook\CNN_model.json", 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(r"C:\Users\laksh\OneDrive\Desktop\jupyternotebook\best_model1_weights.keras")
    
    with open(r"C:\Users\laksh\OneDrive\Desktop\jupyternotebook\scaler2.pickle", 'rb') as f:
        scaler2 = pickle.load(f)
    with open(r"C:\Users\laksh\OneDrive\Desktop\jupyternotebook\encoder2.pickle", 'rb') as f:
        encoder2 = pickle.load(f)
    
    return loaded_model, scaler2, encoder2

model, scaler2, encoder2 = load_model_and_resources()

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr=22050, frame_length=2048, hop_length=512, flatten=True):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_predict_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data)
    features = np.reshape(features, newshape=(1, 1620))
    scaled_features = scaler2.transform(features)
    final_features = np.expand_dims(scaled_features, axis=2)
    return final_features

# Emotion dictionary with emojis
emotions_with_emoji = {
    'Neutral': '😐',
    'Calm': '😌',
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😡',
    'Fear': '😨',
    'Disgust': '🤢',
    'Surprise': '😲'
}

def prediction(path):
    res = get_predict_feat(path)
    predictions = model.predict(res)
    predicted_emotion = encoder2.inverse_transform(predictions)
    # Normalize the predicted emotion for matching
    return predicted_emotion[0][0].strip().capitalize()

# Streamlit Interface
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Upload Audio", "About"],
        icons=["house", "cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Home Section
if selected == "Home":
    st.title("🎙️ Speech Emotion Recognition")
    st.subheader("Detect emotions from speech audio files using AI.")
    st.markdown(
        """
        - Upload your audio to identify the emotion conveyed.
        - Works on short audio clips (2.5 seconds recommended).
        - Supports WAV format for input files.
        """
    )

# Upload Audio Section
elif selected == "Upload Audio":
    st.title("🎵 Upload a Speech Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Analyze Emotion"):
        if uploaded_file is not None:
            temp_filename = "temp.wav"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                emotion = prediction(temp_filename)
                # Match the normalized emotion to the emoji
                emoji = emotions_with_emoji.get(emotion, "❓")
                st.success(f"Predicted Emotion: **{emotion}** {emoji} 🎉")
                st.markdown(f"### **Emotion Analysis Complete!** {emoji}")
            except Exception as e:
                st.error("Error occurred while predicting emotion.")
                st.error(str(e))
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        else:
            st.warning("Please upload a file first.")

# About Section
elif selected == "About":
    st.title("About")
    st.markdown(
        """
        **Speech Emotion Recognition** is an advanced AI tool that uses deep learning to analyze speech signals 
        and predict the emotional state of the speaker. 

        ### Features:
        - Utilizes MFCC, RMSE, and ZCR features for analysis.
        - Supports real-time audio analysis.
        - Predicts emotions like Happy, Sad, Angry, and more.

        **Developed by:** [The Emotion Engineers]
        """
    )
    st.image("img.jpg", caption="AI in Action")

# Footer
st.sidebar.info("Developed with ❤️ using Streamlit.")
