import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import speech_recognition as sr
import tempfile
import io

# --------------------------
# Initialize session
# --------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello 👋, how can I help you today?")
        ]
    if "recording" not in st.session_state:
        st.session_state.recording = False


# --------------------------
# Convert audio to text
# --------------------------
def transcribe_audio(file_bytes):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(file_bytes)) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn’t understand that."
    except sr.RequestError:
        return "Speech recognition service unavailable."


# --------------------------
# Streamlit App
# --------------------------
def run():
    st.set_page_config(page_title="ChatGPT Voice App", layout="centered")
    st.title("💬 ChatterHub")

    init_session()
    llm = ChatOllama(model="tinydolphin:1.1b", temperature=0.1)

    system_prompt = (
        "You are a helpful and concise AI assistant. "
        "Respond naturally and briefly to user queries. "
        "Keep answers under 100 words unless asked for more detail."
    )

    # --------------------------
    # Display chat messages
    # --------------------------
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("ai"):
                st.markdown(msg.content)

    # --------------------------
    # Input section
    # --------------------------
    col1, col2 = st.columns([6, 1])
    with col1:
        user_prompt = st.chat_input("Type your message...")

    with col2:
        # Toggle voice mode
        if st.button("🎤 Voice"):
            st.session_state.recording = not st.session_state.recording

    # --------------------------
    # Handle voice input
    # --------------------------
    voice_prompt = None
    if st.session_state.recording:
        st.info("🎙️ Recording... Press the 🎤 button again to stop.")
        audio_data = st.audio_input("Speak now (click stop when done)", key="voice_recorder")

        # User pressed stop and uploaded audio
        if audio_data is not None:
            st.session_state.recording = False
            st.success("Recording stopped! Processing...")
            voice_prompt = transcribe_audio(audio_data.getvalue())
            if "Sorry" in voice_prompt:
                st.error(voice_prompt)
                voice_prompt = None
            else:
                st.success(f"You said: {voice_prompt}")

    # Determine user input
    prompt = user_prompt or voice_prompt

    # --------------------------
    # Generate AI Response
    # --------------------------
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        messages = [SystemMessage(content=system_prompt)] + st.session_state.messages
        with st.chat_message("ai"):
            stream = llm.stream(messages)
            response = st.write_stream(stream)

        st.session_state.messages.append(AIMessage(content=response))
        st.rerun()


# --------------------------
# Run app
# --------------------------
if __name__ == "__main__":
    run()
