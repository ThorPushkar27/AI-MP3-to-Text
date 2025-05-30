# Importing the dependencies..
import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import soundfile as sf
from pydub import AudioSegment
import torch
import os


# To avoid reloading the model again and again, cache it. Used specific streamlit function for that.
@st.cache(allow_output_mutation=True)
def load_whisper_model():
    model_id = "openai/whisper-base"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True,  #Primary uses accelerator for low cpu usage.
        use_safetensors=True
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Hugging face pipeline
    pipe = pipeline(
        "automatic-speech-recognition",  # It's an speech recognition model
        model=model,
        tokenizer=processor.tokenizer,  #Transformer architecture, where input is tokenized before processing
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )
    
    return pipe

def transcribe_mp3(file_path, pipe):
    #Since whishper takes file in .wav format for mel spectrogram. So, we need to convert mp3 into .wav
    
    # Convert MP3 to WAV for Whisper
    wav_path = file_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(file_path)
    audio.export(wav_path, format="wav")
    
    print("Transcribing audio...")
    result = pipe(wav_path, return_timestamps=True)
    
    # Remove the wav file, we don't need it now
    if os.path.exists(wav_path):
        os.remove(wav_path)
    
    return result

# Streamlit App 
st.title("AI MP3 to Text Converter")

# Load model once
with st.spinner("Loading the model... (first run may take a moment)"):
    pipe = load_whisper_model()

#Upload the mp3 file.
uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Display Uploaded file name 
    st.write(f"**File:** {uploaded_file.name}")
    uploaded_file.seek(0)
    
    # Displaying the audio
    st.write("**Audio preview:**")
    st.audio(uploaded_file, format="audio/mp3")
    
    if st.button("Transcribe Audio"):
        try:
            with st.spinner("Transcribing audio... Please wait."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Transcribe the audio
                result = transcribe_mp3(tmp_file_path, pipe)
                
                # Display results
                st.success("Transcription completed!")
                st.subheader("Transcribed text:")
                st.write(result["text"])
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please try with a different audio file or check the file format.")
        
       
