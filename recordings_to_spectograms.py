from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import matplotlib as mpl
import os



def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(128, 128), noverlap=16, cmap='viridis'):
  sample_rate, samples = wav.read(audio_path)
  fig = plt.figure()
  fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.specgram(samples, cmap=cmap, Fs=22050, noverlap=noverlap)
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())
  fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

def dir_to_spectrogram(audio_dir, spectrogram_dir, spectrogram_dimensions=(128, 128), noverlap=16, cmap='viridis'):
  file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
  for file_name in file_names:
    print(file_name)
    audio_path = os.path.join(audio_dir,file_name)
    spectogram_path = os.path.join(spectrogram_dir,file_name.replace('.wav', '.png'))
    wav_to_spectrogram(audio_path, spectogram_path, spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)


def trasnform_train_data(data):
    mpl.use('Agg')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path,data)
    spectrogram_folder = os.path.join(data_path,"spectrograms")
    audio_folder = os.path.join(data_path,"recordings")
    dir_to_spectrogram(audio_folder, spectrogram_folder)
