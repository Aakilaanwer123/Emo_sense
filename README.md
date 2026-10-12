# Smart Human Emotion Detection — Streamlit

## Files
- `streamlit_emotion_app.py`: Streamlit app for face-photo and voice emotion prediction, plus an emotion gallery.
- `requirements.txt`: Python packages needed by Streamlit Community Cloud.

## Deploy on Streamlit Community Cloud
1. Upload both files to the root of your GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create/deploy an app using your repository and branch.
4. Set **Main file path** to `streamlit_emotion_app.py`.
5. Deploy.

The first launch may take several minutes because models and packages need to download. Voice predictions are estimates; the selected audio model supports four labels (angry, happy, neutral, sad).
