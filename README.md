# HappyPics

HappyPics is a Streamlit web application that uses facial emotion detection to analyze uploaded images and display the detected emotions with confidence scores. It integrates Firebase Authentication for user login and sign-up and stores analysis results in a SQLite database. This repository contains the code for the HappyPics app, built with Python, Streamlit, Matplotlib, OpenCV, and the Facial Emotion Recognition (FER) library.

## Key Features:
1. Upload images (JPEG, PNG) to analyze facial expressions.
2. Detect emotions (e.g., happiness, sadness) with confidence scores using FER.
3. User authentication via Firebase Authentication.
4. Store and display analysis history for each user.
5. View plots based on the user's analysis history.
6. Deployed on Streamlit Community Cloud for easy access and usage.
   
## Technologies Used:
- Python
- Streamlit
- OpenCV
- Facial Emotion Recognition (FER) library
- Firebase Authentication
- SQLite
- Matplotlib
  
## Deployment:
The app is deployed on Streamlit Community Cloud and can be accessed [here](https://happypics-qtrplvnyyzzdtbeboj7kks.streamlit.app/).

## Usage:
1. Sign in with your email and password to start analyzing images.
2. Upload an image and see real-time emotion detection results.
3. View your analysis history and choose plots in the sidebar.
4. Logout securely when done.
   
## Contributing:
Contributions are welcome! Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
