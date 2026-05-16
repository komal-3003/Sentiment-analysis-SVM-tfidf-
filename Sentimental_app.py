import streamlit as st
import pickle
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK files
nltk.download('punkt')
nltk.download('stopwords')

# Load model
model = pickle.load(
    open('svm_model.pkl', 'rb')
)

# Load vectorizer
vectorizer = pickle.load(
    open('tfidf_vectorizer.pkl', 'rb')
)

# Load emotion labels
emotion_labels = pickle.load(
    open('emotion_labels.pkl', 'rb')
)

# Reverse labels
reverse_labels = {}

for key, value in emotion_labels.items():
    reverse_labels[value] = key

# Stopwords
stop_words = set(stopwords.words('english'))

keep_words = ['not', 'no', 'never']

for word in keep_words:
    stop_words.discard(word)

# Text cleaning function
def clean_text(txt):

    txt = txt.lower()

    txt = txt.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = word_tokenize(txt)

    cleaned = []

    for word in words:

        if word not in stop_words:
            cleaned.append(word)

    return " ".join(cleaned)

# Streamlit page settings
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

# Title
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>
😊 AI Emotion Detection App
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align: center; color: gray;'>
Detect human emotions from text using NLP + Machine Learning
</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 About")

st.sidebar.info("""
This project uses:

✅ TF-IDF Vectorization  
✅ LinearSVC Model  
✅ NLP Text Preprocessing  
✅ Streamlit Deployment  

Built using Python and Machine Learning.
""")

# Sample examples
st.subheader("✨ Try Examples")

col1, col2 = st.columns(2)

with col1:
    st.info("I am very happy today")

with col2:
    st.info("I feel nervous about exams")

# User input
user_input = st.text_area(
    "Enter your text below",
    height=150
)

# Predict button
if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        # Clean text
        cleaned_input = clean_text(user_input)

        # Vectorize
        vector_input = vectorizer.transform(
            [cleaned_input]
        )

        # Predict
        # Manual negation handling
        lower_text = user_input.lower()

        if "not happy" in lower_text:
            emotion = "sadness"

        elif "not good" in lower_text:
            emotion = "sadness"

        elif "not joyful" in lower_text:
            emotion = "sadness"
        
        elif "not excited" in lower_text:
            emotion = "sadness"

        else:

            prediction = model.predict(vector_input)[0]

            emotion = reverse_labels[prediction]

        # Output
        st.subheader("Prediction Result")

        if emotion == "joy":
            st.success(f"😊 Emotion Detected: {emotion}")

        elif emotion == "sadness":
            st.error(f"😢 Emotion Detected: {emotion}")

        elif emotion == "anger":
            st.warning(f"😠 Emotion Detected: {emotion}")

        elif emotion == "love":
            st.balloons()
            st.success(f"❤️ Emotion Detected: {emotion}")

        elif emotion == "fear":
            st.warning(f"😨 Emotion Detected: {emotion}")

        elif emotion == "surprise":
            st.info(f"😲 Emotion Detected: {emotion}")

        else:
            st.write(f"Emotion Detected: {emotion}")


# Custom CSS
# Custom CSS
st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #0E1117;
}

/* Text Area */
.stTextArea textarea {
    background-color: white;
    color: black;
    border-radius: 12px;
    border: 2px solid #ccc;
    padding: 12px;
    font-size: 16px;
    transition: all 0.3s ease;
}

/* Text Area Hover */
.stTextArea textarea:hover {
    border: 2px solid #FF4B4B;
    box-shadow: 0px 0px 15px rgba(255,75,75,0.3);
}

/* Text Area Focus */
.stTextArea textarea:focus {
    border: 2px solid #FF4B4B !important;
    box-shadow: 0px 0px 20px rgba(255,75,75,0.5) !important;
    background-color: white !important;
    color: black !important;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}

/* Button Hover */
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff2e2e, #ff5c5c);
    box-shadow: 0px 0px 20px rgba(255,75,75,0.6);
    color: white;
}

/* Prediction Result Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    background-color: #262730;
    color: white;
    border: 2px solid #FF4B4B;
    box-shadow: 0px 0px 20px rgba(255,75,75,0.3);
}

</style>
""", unsafe_allow_html=True)