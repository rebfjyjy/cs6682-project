from moviepy.editor import AudioFileClip
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def convert_mp3_to_wav(mp3_path, wav_path):
    audio_clip = AudioFileClip(mp3_path)
    audio_clip.write_audiofile(wav_path, codec='pcm_s16le')  # pcm_s16le is the codec for .wav format
    audio_clip.close()
    print(f"Converted {mp3_path} to {wav_path}")



# Example usage:
# convert_mp3_to_wav('./audio.mp3', './output_audio.wav')

