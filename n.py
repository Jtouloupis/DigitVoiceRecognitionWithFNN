import pyaudio
import keyboard
import wave as wv
import librosa
import soundfile as sf
import os
import shutil
from recordings_to_spectograms import dir_to_spectrogram
import numpy as np
from tensorflow import keras
import keras.utils as image
import pickle

import librosa.display
import matplotlib.pyplot as plt

from pydub import AudioSegment
from pydub.silence import split_on_silence

dir_path = os.path.dirname(os.path.realpath(__file__))
tmp_path = os.path.join(dir_path,"tmp")

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
else:
    shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)

numbers_tmp_path = os.path.join(tmp_path,"num_tmp")

if not os.path.exists(numbers_tmp_path):
    os.makedirs(numbers_tmp_path)

spectrograms_tmp_path = os.path.join(tmp_path,"spect_tmp")

if not os.path.exists(spectrograms_tmp_path):
    os.makedirs(spectrograms_tmp_path)





def record():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 22050  # Record at 22050 samples per second
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print ("Press enter to start:")


    keyboard.wait('enter')
    print ("Press 'ESC' to stop:")

    frames = []  # Initialize array to store frames
    while True: # Loop until a key has been pressed
        data = stream.read(chunk) # Record data
        frames.append(data) # Add the data to a buffer (a list of chunks)
        if keyboard.is_pressed('esc'):
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wv.open(os.path.join(tmp_path, filename), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_file = "output.wav"
    #read wav data
    audio, sr = librosa.load(os.path.join(tmp_path, audio_file))


    S_full, phase = librosa.magphase(librosa.stft(audio))

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 1,2
    power = .01

    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    y_foreground = librosa.istft(S_foreground * phase)

    audio_file = "cleared.wav"


    sf.write(os.path.join(tmp_path, audio_file), y_foreground, sr, subtype='PCM_24')


    audio, sr = librosa.load(os.path.join(tmp_path, audio_file))

    #segment_audio(os.path.join(tmp_path, audio_file),numbers_tmp_path)
    #clips = librosa.effects.split(audio,top_db=70)



    sound = AudioSegment.from_file(os.path.join(tmp_path, audio_file), format=".wav")
    chunks = split_on_silence(
        sound,

        # split on silences longer than 1000ms (1 sec)
        min_silence_len=800,

        # anything under -16 dBFS is considered silence
        silence_thresh=-65,

        # keep 200 ms of leading/trailing silence
        keep_silence=200
    )

    for i, chunk in enumerate(chunks):
       output_file = "{0}.wav".format(i)
       print("Exporting file", output_file)
       chunk.export(os.path.join(numbers_tmp_path,output_file), format="wav")


    dir_to_spectrogram(numbers_tmp_path+"/",spectrograms_tmp_path+"/")

def load_image(filename):
    img = image.load_img(os.path.join(spectrograms_tmp_path, filename))
    img = image.img_to_array(img)
    img = img.reshape(1,128,128,3)
    img = img.astype('float32')
    img = img / 255.0
    return img


def predict():
    #load model
    #model0 = pickle.load(open('spoken_digit_recognition_.h5', 'rb'))

    model0 = keras.models.load_model('best_model.h5')
    print(model0.input_shape)
    for filename in os.listdir(spectrograms_tmp_path):
        if filename.endswith(".png"):
            img = load_image(filename)
            print(img.shape)
            #nsamples, nx, ny, nf = img.shape
            #img2D = img.reshape((nsamples,nx*ny*nf))
            print('Prediction' ,np.argmax(model0.predict(img)))

            continue

record()
predict()
