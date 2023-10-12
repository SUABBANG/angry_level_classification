import joblib
import librosa
import numpy as np
import soundfile as sf
import pickle


from transformers import pipeline
from flask import Flask, jsonify, request, flash, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn import metrics
from transformers import pipeline
from sklearn.preprocessing import StandardScaler


def calculate_pitch(audio_path):
    import librosa
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    import numpy as np

    pitch_list=[] 
    audio_sample, sampling_rate = librosa.load(audio_path, sr = None)

    S = np.abs(librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sampling_rate)

    shape = np.shape(pitches)
    nb_samples = shape[0]
    nb_windows = shape[1]

    for i in range(0, nb_windows):
        index = magnitudes[:,i].argmax()
        pitch = pitches[index,i]
        pitch_list.append(pitch)
        mean_pitch = np.mean(pitch_list)
    return mean_pitch

def calculate_speech_rate(audio_path):
    import librosa

    audio, sr = librosa.load(audio_path)

    non_silent_intervals = librosa.effects.split(audio, top_db=20)

    speech_durations = [librosa.get_duration(y=audio[start:end], sr=sr) for start, end in non_silent_intervals]

    average_speech_rate = len(speech_durations) / sum(speech_durations)
  
    return average_speech_rate

def calculate_decibel(audio_path):
    # 오디오 파일을 로드합니다.
    audio, sr = librosa.load(audio_path)

    # 시간-주파수 분석을 수행합니다.
    stft = np.abs(librosa.stft(audio))

    # 파워 스펙트럼을 계산합니다.
    power_spec = librosa.power_to_db(stft**2)

    # 최대 데시벨 값을 계산합니다.
    max_db = np.max(power_spec)
    
    return max_db

# 화자인식
model = joblib.load('saved_model.pkl')
# 분노 중립 분류
classifier = pipeline("audio-classification", model="SUABBANG/my_voice_classification_model")

# 단계분류
# filename = '/mnt/c/Users/user/angry_level_classification/angry_level_model/angry_level_main/backend/xgb_model_ver4.sav'
# loaded_model = pickle.load(open(filename, 'rb'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = './files/'
ALLOWED_EXTENSIONS = {'wav'}

#-------------

app=Flask(__name__)

# main update
@app.route('/main', methods=['GET','POST'])

def main():
    # print(request.method) # GET or POST
    # UPLOAD SOUND FILE
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        # print(f.filename)
        global filename
        filename = f.filename + '.wav'
        if filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(filename):
            # print(filename)
            f.save(UPLOAD_FOLDER + secure_filename(filename))
            sound_file = UPLOAD_FOLDER + filename
    #         print(sound_file) # wav 파일 경로

    # print("파일경로:",sound_file)
    #화자인식
    y, sr = librosa.load(sound_file) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
    y_test_estimated = model.predict(mfcc)
    test_label = np.full(len(mfcc), 5) # test 파일에 해당하는 레이블 -> 정답값 (한별 : 5)
    ac_score = metrics.accuracy_score(y_test_estimated, test_label)
    # print("정답률 =", ac_score)

    if ac_score >= 0.7:
        print("측정하지않습니다")
        # 싱딤원 -> 모델 측정 X
        result_lavel = 0

    else: # 고객
        # sampling rate 재지정
        # new_sr_audio_path = './files/new_sr/blob.wav'
        audio, sr = librosa.load(sound_file, sr=None)
        # sf.write(new_sr_audio_path, audio, sr)

        result_percent = classifier(sound_file)
        # print(result_percent)

        # result_percent.sort(key=lambda x: x['score'], reverse=True)
        # highest_score_label = result_percent[0]['label']
        # print(highest_score_label)

        for item in result_percent:
            if item['label'] == 'angry':
                angry_score = item['score']
                break
        print("angry_score",angry_score)

        decibel = calculate_decibel(sound_file)
        print("decibel:", decibel)
        pitch = calculate_pitch(sound_file)
        print("pitch:", pitch)
        speech_rate = calculate_speech_rate(sound_file)
        print("speech_rate:", speech_rate)

        # 분노 단계 분류
        # if highest_score_label == 'angry' :
        if angry_score >= 0.6 : # 소음처리
            result_lavel = 0

        elif angry_score >= 0.35 or decibel >= 70 :
            # print("분노함")

            if decibel >= 45:
                #여기서 분노 분류 코드 이따 넣기
                if decibel >= 70 and pitch >= 600:
                    result_lavel = 3
                elif  decibel >= 70:
                    result_lavel = 2
                else:
                    result_lavel = 1 
            else:
                result_lavel = 0


        else:
            result_lavel = 0


    return {'result_lavel':result_lavel}

if __name__ == "__main__":
    app.run(debug=True)

# 추가
CORS(
    app, 
    resources={r'*': {'origins': '*'}}, 
    supports_credentials=True)