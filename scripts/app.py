import streamlit as st
import os
import ffmpeg
import whisper
from transformers import pipeline
from gtts import gTTS
from googletrans import Translator
import yt_dlp

# App Title
st.title("AI-Powered Speech Analysis Tool")

# Description
st.write("This app helps convert speech to text, analyze sentiment, summarize content, and convert summaries back to speech. It supports English, Spanish, and French users. You can upload files or provide online video links.")

# Create directories to save uploaded files
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/downloads", exist_ok=True)

# Function to download video from URL
def download_video(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'data/downloads/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict).replace('.webm', '.wav').replace('.m4a', '.wav')
        st.success(f"Video downloaded and audio extracted successfully: {downloaded_file}")
        return downloaded_file
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Function to extract audio from a video file
def extract_audio(video_path, output_path):
    try:
        ffmpeg.input(video_path).output(output_path, format="wav").overwrite_output().run()
        st.success(f"Audio extracted and saved at: {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(audio_path, language="en"):
    try:
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(audio_path, language=language)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Function to perform sentiment analysis
def analyze_sentiment(text):
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        chunk_size = 512
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        results = [sentiment_pipeline(chunk)[0] for chunk in chunks]
        sentiment_labels = [res["label"] for res in results]
        sentiment_scores = [res["score"] for res in results]
        aggregated_label = max(set(sentiment_labels), key=sentiment_labels.count)
        aggregated_score = sum(sentiment_scores) / len(sentiment_scores)
        return {"label": aggregated_label, "score": aggregated_score}
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return None

# Function to summarize text
def summarize_text(text):
    try:
        summarizer = pipeline("summarization")
        chunk_size = 1024
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

# Function to translate text
def translate_text(text, target_language="en"):
    try:
        translator = Translator()
        if text:
            translation = translator.translate(text, dest=target_language)
            return translation.text
        else:
            st.error("No text available for translation.")
            return None
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text, language="en"):
    try:
        tts = gTTS(text=text, lang=language)
        audio_path = "data/output_audio.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

# Main logic
source_option = st.radio("Choose your input source:", ("Upload File", "Online Video Link"))

if source_option == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload an audio or video file", 
        type=["mp3", "wav", "mp4"],
        help="Upload audio or video files for transcription and analysis."
    )

    if uploaded_file is not None:
        file_path = os.path.join("data/uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write(f"File saved at: {file_path}")

        if uploaded_file.name.endswith(".mp4"):
            audio_path = file_path.replace(".mp4", ".wav")
            audio_path = extract_audio(file_path, audio_path)
        else:
            audio_path = file_path

elif source_option == "Online Video Link":
    video_url = st.text_input("Enter the video URL:")

    if video_url:
        downloaded_video = download_video(video_url)
        if downloaded_video:
            audio_path = downloaded_video

if 'audio_path' in locals() and audio_path:
    st.write(f"Audio ready for processing: {audio_path}")

    transcription_language = st.selectbox(
        "Select the language for transcription:",
        options=[("English", "en"), ("Spanish", "es"), ("French", "fr")],
        format_func=lambda lang: lang[0]
    )[1]

    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(audio_path, language=transcription_language)

    if transcription:
        st.subheader("Transcription")
        st.write(transcription)

        sentiment_result = analyze_sentiment(transcription)
        if sentiment_result:
            st.subheader("Sentiment Analysis")
            st.write(f"Sentiment: **{sentiment_result['label']}**")
            st.write(f"Confidence: **{sentiment_result['score']:.2f}**")

        summary = summarize_text(transcription)
        if summary:
            st.subheader("Text Summarization")
            st.write(summary)

            target_language = st.selectbox(
                "Select the language for translation:",
                options=[("English", "en"), ("Spanish", "es"), ("French", "fr")],
                format_func=lambda lang: lang[0]
            )[1]

            translated_summary = translate_text(summary, target_language=target_language)
            if translated_summary:
                st.subheader("Translated Summary")
                st.write(translated_summary)

                tts_audio_path = text_to_speech(translated_summary, language=target_language)
                if tts_audio_path:
                    st.audio(tts_audio_path, format="audio/mp3", start_time=0)
else:
    st.info("Please upload a file or enter a video link to begin.")

