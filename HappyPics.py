import streamlit as st
import pyrebase
from PIL import Image
import numpy as np
import cv2
from fer import FER
from io import BytesIO
from sqlalchemy import create_engine, Column, String, Float, LargeBinary, Integer, MetaData, Table, desc
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Firebase configuration
firebase_config = {
    "apiKey": os.getenv("API_KEY"),
    "authDomain": os.getenv("AUTH_DOMAIN"),
    "databaseURL": os.getenv("DATABASE_URL"),
    "projectId": os.getenv("PROJECT_ID"),
    "storageBucket": os.getenv("STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("MESSAGING_SENDER_ID"),
    "appId": os.getenv("APP_ID")
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Database Configuration
DATABASE_URL = "sqlite:///user_data.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Table to store the history of the images the user shared with HappyPics
user_history_table = Table(
    'user_history', metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String, nullable=False),
    Column('image', LargeBinary, nullable=False),
    Column('emotion', String, nullable=False),
    Column('score', Float, nullable=False)
)

# To create tables
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to save the analysis result to history
def save_to_history(email, image, emotion, score):
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        new_record = user_history_table.insert().values(email=email, image=img_data, emotion=emotion, score=score)
        session.execute(new_record)
        session.commit()
    except Exception as e:
        st.error(f"Error saving to history: {str(e)}")

# Function to retrieve user history from the database
def get_user_history(email):
    try:
        query = user_history_table.select().where(user_history_table.c.email == email).order_by(desc(user_history_table.c.id))
        result = session.execute(query).fetchall()
        history = []
        for row in result:
            history.append({
                'image': row[2],
                'emotion': row[3],
                'score': row[4]
            })
        return history
    except Exception as e:
        st.error(f"Error retrieving user history: {str(e)}")
        return []

# Function to clear user history from the database
def clear_user_history(email):
    try:
        delete_query = user_history_table.delete().where(user_history_table.c.email == email)
        session.execute(delete_query)
        session.commit()
        st.success("History cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")

# Function to show graph emotion confidence over time
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

# Function to show graph emotion proportions
def plot_emotion_proportions(history):
    if history:
        emotions = [record['emotion'] for record in history]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        fig, ax = plt.subplots()
        ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Emotion Proportions')
        
        st.pyplot(fig)

# Function to show graph emotion score trends
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

# Function to handle sentiment analysis process for images
def sentiment_analysis():
    st.title("HappyPics")
    st.subheader("Discover your sentiments together!")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image uploaded.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            detector = FER(mtcnn=True)
            emotions = detector.detect_emotions(opencv_image)
            if emotions:
                emotion, score = detector.top_emotion(opencv_image)
                st.write(f"Detected emotion: {emotion} with confidence {score:.2f}")
                save_to_history(st.session_state['user_info']['email'], image, emotion, score)
            else:
                st.warning("No face detected or unable to detect emotions")
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
    # Sidebar username and logout button
    st.sidebar.title(f"Hi, {st.session_state['user_info']['email']}")

    if st.sidebar.button("Logout", type="primary"):
        st.session_state['authenticated'] = False
        st.session_state['user_info'] = None
        st.rerun()

    st.sidebar.subheader("""________________________________""")

    if 'user_info' in st.session_state:
        history = get_user_history(st.session_state['user_info']['email'])
        if history:
            # Sidebar graph view options
            st.sidebar.subheader("Graph Options")
            if 'show_confidence' not in st.session_state:
                st.session_state['show_confidence'] = False
            if 'show_proportions' not in st.session_state:
                st.session_state['show_proportions'] = False
            if 'show_trends' not in st.session_state:
                st.session_state['show_trends'] = False

            if st.sidebar.checkbox("Show Emotion Confidence Over Time", value=st.session_state['show_confidence']):
                st.session_state['show_confidence'] = True
                plot_emotion_confidence_over_time(history)
            else:
                st.session_state['show_confidence'] = False

            if st.sidebar.checkbox("Show Emotion Proportions", value=st.session_state['show_proportions']):
                st.session_state['show_proportions'] = True
                plot_emotion_proportions(history)
            else:
                st.session_state['show_proportions'] = False

            if st.sidebar.checkbox("Show Emotion Score Trends", value=st.session_state['show_trends']):
                st.session_state['show_trends'] = True
                plot_emotion_score_trends(history)
            else:
                st.session_state['show_trends'] = False
            
            # Sidebar user history section
            st.sidebar.subheader("""________________________________""")
            st.sidebar.subheader("Your History")
            if st.sidebar.button("Clear History", type="primary"):
                clear_user_history(st.session_state['user_info']['email'])

            for record in history:
                img = Image.open(BytesIO(record['image']))
                st.sidebar.image(img, caption=f"Emotion: {record['emotion']}, Confidence: {record['score']:.2f}", use_column_width=True)
        else:
            st.sidebar.write("No history yet")

# Function to handle Firebase Authentication
def handle_firebase_auth():
    st.subheader("Sign in to explore your emotions with just a single picture!")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    login_clicked = st.button("Login", type="primary")

    st.subheader("New to HappyPics?")
    signup_clicked = st.button("Sign Up", type="primary")

    if login_clicked:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state['authenticated'] = True
            st.session_state['user_info'] = user
            st.session_state['login_error'] = None
            st.success("Logged in successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.session_state['login_error'] = "Invalid credentials or error occurred. Please try again."

    if signup_clicked:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("Account created successfully! Please login.")
        except:
            st.error("Error creating account. Please try again.")

    if 'login_error' in st.session_state and st.session_state['login_error']:
        st.error(st.session_state['login_error'])

# Main logic with session
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if st.session_state['authenticated']:
    sentiment_analysis()
else:
    st.header("Welcome to :blue[HappyPics] :wave:", divider='rainbow')
    handle_firebase_auth()
