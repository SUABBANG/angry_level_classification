import joblib
import librosa


from transformers import pipeline
from flask import Flask, jsonify, request, flash, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn import metrics

# 화자인식
model = joblib.load('saved_model.pkl')


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
    if ac_score >= 69:
        # 싱딤원 -> 모델 측정 X
        result_lavel = 0
    else: # 고객
        result_lavel = 1 #가상값
     
    

    return {'result_lavel':result_lavel}

if __name__ == "__main__":
    app.run(debug=True)

# 추가
CORS(
    app, 
    resources={r'*': {'origins': '*'}}, 
    supports_credentials=True)