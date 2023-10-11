import joblib
import librosa
import numpy as np
import soundfile as sf


from transformers import pipeline
from flask import Flask, jsonify, request, flash, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn import metrics
from transformers import pipeline

# 화자인식
model = joblib.load('saved_model.pkl')
# 분노 중립 분류
classifier = pipeline("audio-classification", model="SUABBANG/my_voice_classification_model")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = './files/'
ALLOWED_EXTENSIONS = {'wav'}

app=Flask(__name__)

# main update
@app.route('/main', methods=['GET','POST'])

def main():
    print(request.method) # GET or POST
    # UPLOAD SOUND FILE
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        print(f.filename)
        global filename
        filename = f.filename + '.wav'
        if filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(filename):
            print(filename)
            f.save(UPLOAD_FOLDER + secure_filename(filename))
            sound_file = UPLOAD_FOLDER + filename
            print(sound_file) # wav 파일 경로

    print("파일경로:",sound_file)
    #화자인식
    y, sr = librosa.load(sound_file) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
    y_test_estimated = model.predict(mfcc)
    test_label = np.full(len(mfcc), 5) # test 파일에 해당하는 레이블 -> 정답값 (한별 : 5)
    ac_score = metrics.accuracy_score(y_test_estimated, test_label)
    print("정답률 =", ac_score)

    if ac_score >= 70:
        # 싱딤원 -> 모델 측정 X
        result_lavel = 0
    else: # 고객
        # sampling rate 재지정
        new_sr_audio_path = './files/new_sr/blob.wav'
        audio, sr = librosa.load(sound_file, sr=16000)
        sf.write(new_sr_audio_path, audio, sr)

        result_percent = classifier(new_sr_audio_path)
        print(result_percent)

        result_percent.sort(key=lambda x: x['score'], reverse=True)
        highest_score_label = result_percent[0]['label']

        # 분노 단계 분류
        if result_percent == 'angry' :
            db = librosa.amplitude_to_db(librosa.feature.rms(y=audio), ref=np.max)
            pitch, _ = librosa.effects.pitch(audio)

            db_mean = db.mean()
            pitch_mean = pitch.mean()

            if db_mean > -20 and pitch_mean > 200:  # 높은 데시벨 및 높은 피치
                result_lavel = 3
            elif db_mean > -20:  # 높은 데시벨, 낮은 피치
                result_lavel = 2
            else:  # 낮은 데시벨
                result_lavel = 1
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