import os
import pyaudio
import wave
from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def recongnize():
    # 读入参数
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
    print(ms)
    ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')
    print(ms)

    # 录音
    # 创建对象
    CHUNK = 1024  # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16  # 采样位数
    CHANNELS = 1  # 单声道
    RATE = 16000  # 采样频率
    wave_out_path = r'D:\data\audio\test.wav'
    record_second = 6
    p = pyaudio.PyAudio()
    # 创建流:采样位，声道数，采样频率，缓冲区大小input=True;# 每个缓冲区的帧数
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

    # 开始识别
    res1 = ms.recognize_speech_from_file(wave_out_path)

    # res = ms.recognize_speech_from_file('filename.wav')
    print('*[提示] 声学模型语音识别结果：\n', res1)

    ml = ModelLanguage('model_language')
    ml.load_model()
    str_pinyin = res1
    res2 = ml.pinyin_to_text(str_pinyin)
    print('语音识别最终结果：\n', res2)
    return res1, res2
# recongnize()