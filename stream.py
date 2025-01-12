import torch
import os
import time
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torchaudio
import json
import sounddevice as sd
import soundfile as sf
from parts.text.cleaners import english_cleaners
from datetime import datetime
from absl import app, flags
from augmentation import ConcatFeature
from parts.features import AudioPreprocessing


import av
import torch
import torchaudio
from absl import app, flags

from rnnt.args import FLAGS
from rnnt.stream import PytorchStreamDecoder, OpenVINOStreamDecoder

import tempfile
import queue
import sys

import numpy as np
import socketio

av.logging.set_level(av.logging.ERROR)

# PytorchStreamDecoder
flags.DEFINE_string('model_name', "english_43_medium.pt", help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 2, help='input frame(stacked)')
flags.DEFINE_enum('stream_decoder', 'torch', ['torch', 'openvino'],
                  help='stream decoder implementation')
flags.DEFINE_string('url', 'https://www.youtube.com/watch?v=2EppLNonncc',
                    help='youtube live link')
flags.DEFINE_integer('reset_step', 500, help='reset hidden state')
flags.DEFINE_string('path', None, help='path to .wav')
'''
REAL Time Speech To Text

'''
# transforms = torch.nn.Sequential(AudioPreprocessing(
#         normalize='per_feature', sample_rate=16000, window_size=0.02, 
#         window_stride=0.015, features= 801 , n_fft=512, log=True, pad_to = 1,
#         feat_type='logfbank', trim_silence = False,  window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
#     ))
global blank_counter
blank_counter = 0
global buffer
buffer = []

global encoder_h
encoder_h = None

sd.default.samplerate = 16000

global counter
counter = 0

global seq_list
seq_list = []


# transforms = torch.nn.Sequential(AudioPreprocessing(
#                 normalize='none', sample_rate=16000, window_size=window_size, 
#                 window_stride=window_stride, features=args.audio_feat, n_fft=512, log=True,
#                 feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
#             ), ConcatFeature(merge_size=3))


def callback(raw_indata, out_data,frames, time, status):
    global buffer
    global encoder_h
    global blank_counter
    global seq_list
    global counter
    seq_list.append("power on earth")
    # print ("In Callback")
    if status: # usually something bad
        print("X",  end=" ", flush = True)
    else:
        indata = raw_indata.copy()
        buffer.append(indata)
        buffer = buffer[-2:]
        # indata = indata / (1 << 9)
        indata = indata / (1 << 10)
        waveform = torch.from_numpy(indata.flatten()).float()

        waveform = waveform.unsqueeze(0)
        seq = stream_decoder.decode(waveform)
        if seq == "":
            blank_counter += 1
            if blank_counter == 35:
                print(' [Background]')
                stream_decoder.reset()
        else:
            blank_counter = 0
            seq_list.append(seq)
            print(seq, end='', flush=True)
            print("Seq_list--->", seq_list)
        
    #     # sd.sleep(80000)
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def words():
    global seq_list
    yield seq_list

def test_wav(wav_file):
    import torchaudio

    data, sr = torchaudio.load(wav_file, normalization=True)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        data = resample(data)
        sr = 16000
    data_ = data[0]
    data_ = data_.unsqueeze(0)
    print(data_.shape)
    seq = stream_decoder.decode(data_)
    print (seq)


def main(argv):
    global stream_decoder
    stream_decoder = PytorchStreamDecoder(FLAGS)
    duration = 80
    if FLAGS.path is not None:
        return test_wav(FLAGS.path)
    else:
        with sd.Stream(channels=1,dtype='float32', samplerate= 16000, 
            blocksize=FLAGS.win_length*FLAGS.step_n_frame + (FLAGS.step_n_frame - 1),
            # blocksize=500,
            # never_drop_input = True,
            callback=callback, 
            latency='high'):
            sd.sleep(duration * 100000)

import threading
import streamlit as st
import time
def drive():
    app.run(main)

def main1():
    global seq_list
    global counter
    st.title("AI Model Word Generator")

    # Define the callback function
    def button_callback():
        # thread1.start()
        st.write('Start Speak!')

    # Create the button
    if st.button('Click me', on_click=button_callback):
        word_generator = words()

        word_container = st.empty()

        for word in word_generator:
            word_container.write(word)
if __name__ == "__main__":
    app.run(main)
    thread1 = threading.Thread(target = main1)
    thread1.start()
    # print("calling main1")
    # main1()

