import streamlit as st
from PIL import Image
import numpy as np
import cv2
from fer import FER
from io import BytesIO
import matplotlib.pyplot as plt

#Function to save the analysis result to session history
def save_to_history(image, emotion, score):
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_data = buffered.getvalue()

    if not st.session_state.get('last_uploaded') == img_data:
        st.session_state['history'].append({
            'image': img_data,
            'emotion': emotion,
            'score': score
        })
        st.session_state['last_uploaded'] = img_data

#Function to retrieve history from session
def get_user_history():
    return st.session_state.get('history', [])

#Clear history function
def clear_user_history():
    st.session_state['history'] = [] 
    st.session_state['last_uploaded'] = None 
    st.rerun()  

#Function to plot emotion confidence over time
def plot_emotion_confidence_over_time(history):
    if history:
        timestamps = list(range(1, len(history) + 1))
        emotions = [record['emotion'] for record in history]
        scores = [record['score'] for record in history]

        fig, ax = plt.subplots()
        ax.plot(timestamps, scores, marker='o')
        ax.set_xticks(timestamps)
        ax.set_xticklabels(emotions, rotation=45)
        ax.set_xlabel('Time (Image Uploads)')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Emotion Confidence Over Time')

        st.pyplot(fig)

#Function to plot emotion proportions
def plot_emotion_proportions(history):
    if history:
        emotions = [record['emotion'] for record in history]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        fig, ax = plt.subplots()
        ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Emotion Proportions')

        st.pyplot(fig)

#Function to plot emotion score trends
def plot_emotion_score_trends(history):
    if history:
        timestamps = list(range(1, len(history) + 1))
        emotion_scores = {}

        for record in history:
            if record['emotion'] not in emotion_scores:
                emotion_scores[record['emotion']] = []
            emotion_scores[record['emotion']].append((timestamps.pop(0), record['score']))

        fig, ax = plt.subplots()
        for emotion, scores in emotion_scores.items():
            times, vals = zip(*scores)
            ax.plot(times, vals, marker='o', label=emotion)

        ax.set_xlabel('Time (Image Uploads)')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Emotion Score Trends Over Time')
        ax.legend()

        st.pyplot(fig)

#Show history
def display_history_in_sidebar():
    history = get_user_history() 
    st.sidebar.title("Your History")

    if history:
        for idx, record in enumerate(history):
            st.sidebar.write(f"Image {idx + 1}")
            img = Image.open(BytesIO(record['image']))
            st.sidebar.image(img, width=100, caption=f"Emotion: {record['emotion']}")
            st.sidebar.write(f"Sentiment: {record['emotion']}")
            st.sidebar.write(f"Confidence: {record['score']:.2f}")
            st.sidebar.write("---")

        if st.sidebar.button("Clear History", type="primary"):
            clear_user_history()
    else:
        st.sidebar.write("No history yet.")

#Function to handle sentiment analysis
def sentiment_analysis():
    st.title("HappyPics")
    st.subheader("Discover your sentiments together!")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image uploaded.', use_column_width=True)
            st.write("")
            st.write("Searching for sentiments...")

            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            detector = FER(mtcnn=True)
            emotions = detector.detect_emotions(opencv_image)
            if emotions:
                emotion, score = detector.top_emotion(opencv_image)
                st.write(f"Detected emotion: {emotion} with confidence {score:.2f}")
                save_to_history(image, emotion, score)
            else:
                st.warning("No face detected or unable to detect emotions")
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")

    display_history_in_sidebar()

    st.sidebar.title("Graph Options")
    show_confidence_plot = st.sidebar.checkbox("Show Emotion Confidence Over Time", key='confidence_plot')
    show_proportion_plot = st.sidebar.checkbox("Show Emotion Proportions", key='proportion_plot')
    show_trend_plot = st.sidebar.checkbox("Show Emotion Score Trends", key='trend_plot')

    history = get_user_history()

    if history:
        if show_confidence_plot:
            st.subheader("Emotion Confidence Over Time")
            plot_emotion_confidence_over_time(history)

        if show_proportion_plot:
            st.subheader("Emotion Proportions")
            plot_emotion_proportions(history)

        if show_trend_plot:
            st.subheader("Emotion Score Trends")
            plot_emotion_score_trends(history)

st.header("Welcome to :blue[HappyPics] :wave:", divider='rainbow')
sentiment_analysis()
