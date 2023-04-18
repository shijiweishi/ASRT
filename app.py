#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
用于通过ASRT语音识别系统预测一次语音文件的程序
"""

import os
import pyaudio
import wave
from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
    )
feat = Spectrogram()
ms = ModelSpeech(sm251bn, feat, max_label_length=64)

ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')


# python录音
CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 单声道
RATE = 16000  # 采样频率
def get_audio(wave_out_path,record_second):
    # 创建对象
    p = pyaudio.PyAudio()
    #创建流:采样位，声道数，采样频率，缓冲区大小input=True;# 每个缓冲区的帧数
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # 创建打开音频文件
    wf = wave.open(wave_out_path, 'wb')  # 打开 wav 文件。
    wf.setnchannels(CHANNELS)  # 声道设置
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样位数设置
    wf.setframerate(RATE)  # 采样频率设置
    # 开始录音
    print('开始录音')
    for _ in range(0, int(RATE * record_second / CHUNK)):
        data = stream.read(CHUNK)
        wf.writeframes(data)  # 写入数据
    # 录音结束
    print('录音结束')
    stream.stop_stream()  # 关闭流
    stream.close()
    p.terminate()
    wf.close()


wave_out_path = r'D:\data\audio\test.wav'
# wave_out_path = r'D:\data\audio\ing{}.wav'.format(10)
record_second = 5
get_audio(wave_out_path,record_second)

res = ms.recognize_speech_from_file(wave_out_path)

# res = ms.recognize_speech_from_file('filename.wav')
print('*[提示] 声学模型语音识别结果：\n', res)

ml = ModelLanguage('model_language')
ml.load_model()
str_pinyin = res
res = ml.pinyin_to_text(str_pinyin)
print('语音识别最终结果：\n',res)


# from flask import Flask
# app = Flask(__name__) # 实例化类flask
# @app.route('/getQuesFromfile', methods=['POST'])
# def hello_world():
#
#     return 'Hello World!'
#
# if __name__ == '__main__':
#     app.run(host="localhost", port=8900)
