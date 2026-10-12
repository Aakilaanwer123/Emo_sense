"""
Smart Human Emotion Detection - Streamlit App
Features:
- Upload a face photo and estimate facial emotion
- Upload WAV/FLAC audio and estimate voice emotion
- Separate Emotion Gallery section
- Shows confidence scores

Install in VS Code terminal:
python -m pip install streamlit opencv-python numpy fer torch transformers soundfile pillow scipy

Run:
streamlit run streamlit_emotion_app.py

Note: The voice model supports four labels: angry, happy, neutral, sad.
Emotion predictions are estimates, not a diagnosis.
"""

import io
import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    from fer import FER
except ImportError:
    FER = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


st.set_page_config(
    page_title="Smart Human Emotion Detection",
    page_icon="😊",
    layout="wide",
)

EMOTION_INFO = {
    "happy": {
        "title": "Happy",
        "emoji": "😊",
        "color": "#DDF7C8",
        "message": "Positive and cheerful expression/voice estimate.",
    },
    "sad": {
        "title": "Sad",
        "emoji": "😔",
        "color": "#D9ECFF",
        "message": "Sadness-related expression/voice estimate.",
    },
    "ang": {
        "title": "Angry",
        "emoji": "😠",
        "color": "#FFE0DE",
        "message": "Anger-related voice estimate.",
    },
    "angry": {
        "title": "Angry",
        "emoji": "😠",
        "color": "#FFE0DE",
        "message": "Anger-related expression/voice estimate.",
    },
    "hap": {
        "title": "Happy",
        "emoji": "😊",
        "color": "#DDF7C8",
        "message": "Happiness-related voice estimate.",
    },
    "neu": {
        "title": "Neutral",
        "emoji": "😐",
        "color": "#EEEEEE",
        "message": "Neutral voice estimate.",
    },
    "neutral": {
        "title": "Neutral",
        "emoji": "😐",
        "color": "#EEEEEE",
        "message": "Neutral expression/voice estimate.",
    },
    "fear": {
        "title": "Fear",
        "emoji": "😨",
        "color": "#FFE9CC",
        "message": "Fear-related facial expression estimate.",
    },
    "surprise": {
        "title": "Surprise",
        "emoji": "😲",
        "color": "#FFF2BF",
        "message": "Surprise-related facial expression estimate.",
    },
    "disgust": {
        "title": "Disgust",
        "emoji": "🤢",
        "color": "#D9F7E8",
        "message": "Disgust-related facial expression estimate.",
    },
}

VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"


@st.cache_resource(show_spinner="Loading face emotion model...")
def load_face_model():
    if FER is None:
        raise RuntimeError("FER package is not installed. Run: pip install fer")
    return FER(mtcnn=False)


@st.cache_resource(show_spinner="Loading voice emotion model (first run needs internet)...")
def load_voice_model():
    if pipeline is None:
        raise RuntimeError("Transformers is not installed. Run: pip install transformers")
    return pipeline(
        task="audio-classification",
        model=VOICE_MODEL_NAME,
        top_k=None,
    )


def normalize_label(label):
    label = str(label).strip().lower()
    aliases = {
        "ang": "angry",
        "hap": "happy",
        "neu": "neutral",
        "sad": "sad",
    }
    return aliases.get(label, label)


def show_emotion_card(label, score=None, source=""):
    key = normalize_label(label)
    info = EMOTION_INFO.get(key, {
        "title": key.title(),
        "emoji": "🙂",
        "color": "#EEEEEE",
        "message": "Model prediction estimate.",
    })
    score_text = f"{float(score) * 100:.1f}%" if score is not None else "N/A"
    st.markdown(
        f"""
        <div style="
            background:{info['color']};
            border-radius:16px;
            padding:18px;
            color:#172033;
            border:1px solid #d4d9e2;
            min-height:145px;">
            <div style="font-size:38px;">{info['emoji']}</div>
            <div style="font-size:22px;font-weight:700;">{info['title']}</div>
            <div style="font-size:16px;">Confidence: <b>{score_text}</b></div>
            <div style="font-size:13px;">{source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_audio(uploaded_file):
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")

    audio_bytes = uploaded_file.getvalue()
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    # Remove DC offset and avoid invalid values.
    audio = np.nan_to_num(audio)
    audio = audio - np.mean(audio)

    # The selected model expects 16 kHz audio.
    target_rate = 16000
    if sample_rate != target_rate:
        if resample_poly is None:
            raise RuntimeError("scipy is required to resample audio. Run: pip install scipy")
        from math import gcd
        common = gcd(int(sample_rate), target_rate)
        audio = resample_poly(audio, target_rate // common, int(sample_rate) // common)
        sample_rate = target_rate

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-5:
        raise ValueError("Audio is silent or too quiet. Please upload a clear voice recording.")
    audio = audio / max(peak, 1e-8)

    # Very short clips often produce unreliable results.
    if len(audio) < int(0.5 * sample_rate):
        raise ValueError("Please upload at least 0.5 seconds of voice audio.")

    return audio.astype(np.float32), int(sample_rate)


st.title("😊 Smart Human Emotion Detection")
st.caption("Estimate emotion from a face photo or a voice recording.")
st.info(
    "This demo estimates emotion from visible facial expression or vocal tone. "
    "It may be wrong and should not be used to diagnose anyone."
)

tab_face, tab_voice, tab_gallery = st.tabs(
    ["📷 Face Photo", "🎙️ Voice Upload", "🖼️ Emotion Gallery"]
)

with tab_face:
    st.subheader("Upload a face photo")
    photo = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="face_upload",
    )

    if photo is not None:
        image = Image.open(photo).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded photo", use_container_width=True)

        if st.button("Detect Face Emotion", type="primary", key="detect_face"):
            try:
                detector = load_face_model()
                with st.spinner("Detecting facial expression..."):
                    results = detector.detect_emotions(np.array(image))

                if not results:
                    st.warning("No face detected. Try a clear, front-facing photo with good lighting.")
                else:
                    # Use the face with the largest bounding box.
                    chosen = max(
                        results,
                        key=lambda item: item["box"][2] * item["box"][3],
                    )
                    scores = chosen.get("emotions", {})
                    if not scores:
                        st.warning("The model did not return emotion scores.")
                    else:
                        best_label = max(scores, key=scores.get)
                        best_score = float(scores[best_label])
                        with right:
                            show_emotion_card(best_label, best_score, "Source: uploaded face photo")
                            st.write("**All facial emotion scores**")
                            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                                st.write(f"{label.title()}: {score * 100:.1f}%")
                                st.progress(max(0.0, min(1.0, float(score))))
            except Exception as exc:
                st.error(f"Face detection error: {exc}")

with tab_voice:
    st.subheader("Upload a voice recording")
    st.write("Supported formats: WAV or FLAC. Use a clear voice recording with minimal background noise.")
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac"],
        key="voice_upload",
    )

    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Detect Voice Emotion", type="primary", key="detect_voice"):
            try:
                audio, sr = read_audio(audio_file)
                model = load_voice_model()
                with st.spinner("Analysing voice tone..."):
                    predictions = model({"raw": audio, "sampling_rate": sr})

                # transformers versions can return a list of dicts or nested list.
                if predictions and isinstance(predictions[0], list):
                    predictions = predictions[0]
                predictions = sorted(predictions, key=lambda item: float(item.get("score", 0)), reverse=True)

                if not predictions:
                    st.warning("The model did not return a prediction.")
                else:
                    best = predictions[0]
                    best_label = normalize_label(best.get("label", "unknown"))
                    best_score = float(best.get("score", 0))
                    a, b = st.columns([1, 1])
                    with a:
                        show_emotion_card(best_label, best_score, "Source: uploaded voice audio")
                    with b:
                        st.markdown("#### Full voice emotion scores")
                        for item in predictions:
                            label = normalize_label(item.get("label", "unknown"))
                            score = float(item.get("score", 0))
                            title = EMOTION_INFO.get(label, {}).get("title", label.title())
                            st.write(f"{title}: {score * 100:.1f}%")
                            st.progress(max(0.0, min(1.0, score)))

                    st.caption(
                        "Voice model labels are limited to angry, happy, neutral, and sad. "
                        "The highest score is the model's best estimate, not a guaranteed feeling."
                    )
            except Exception as exc:
                st.error(f"Voice detection error: {exc}")

with tab_gallery:
    st.subheader("Emotion Photos / Cards")
    st.write("Separate visual cards for the emotion categories supported by this demo.")
    gallery_items = [
        ("happy", "Happy"),
        ("sad", "Sad"),
        ("angry", "Angry"),
        ("neutral", "Neutral"),
        ("fear", "Fear (face model only)"),
        ("surprise", "Surprise (face model only)"),
        ("disgust", "Disgust (face model only)"),
    ]
    cols = st.columns(3)
    for index, (key, title) in enumerate(gallery_items):
        info = EMOTION_INFO[key]
        with cols[index % 3]:
            st.markdown(
                f"""
                <div style="
                    background:{info['color']};
                    border-radius:16px;
                    padding:20px;
                    margin-bottom:12px;
                    text-align:center;
                    color:#172033;
                    border:1px solid #d4d9e2;">
                    <div style="font-size:54px;">{info['emoji']}</div>
                    <div style="font-size:20px;font-weight:700;">{title}</div>
                    <div style="font-size:13px;">{info['message']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption("Project demo only • Predictions may be inaccurate • No uploaded files are saved by this app.")
