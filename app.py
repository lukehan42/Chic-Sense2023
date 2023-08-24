from flask import Flask, render_template, request, redirect, url_for,jsonify
import os,time
import cv2
from werkzeug.utils import secure_filename
import shutil,StartACGPN
from flask import Flask, render_template, send_file, Response, request
import pymysql
import requests
from gtts import gTTS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

camera = cv2.VideoCapture(0)

app = Flask(__name__)

api_endpoint = "https://api.openweathermap.org/data/2.5/weather"
api_key = "e988dcd5742e41a5b9ca696d6d207115"


model1 = keras.models.load_model('static/detection2.h5')
model2 = keras.models.load_model('static/color3.h5')
model3 = keras.models.load_model('static/pattern2.h5')

class_names_model1 = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names_model2 = ['Black', 'Blue', 'Green', 'Pink', 'Red', 'White', 'Yellow']

class_names_model3 = ['Dot', 'Floral', 'Plain', 'Striped']



def preprocess_image(image):
    # 이미지 전처리
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    # 이미지 전처리
    input_image = preprocess_image(image)

    # 각 모델 예측
    predictions_model1 = model1.predict(input_image)
    predicted_class_model1 = np.argmax(predictions_model1)
    class_name_model1 = class_names_model1[predicted_class_model1]

    predictions_model2 = model2.predict(input_image)
    predicted_class_model2 = np.argmax(predictions_model2)
    class_name_model2 = class_names_model2[predicted_class_model2]

    predictions_model3 = model3.predict(input_image)
    predicted_class_model3 = np.argmax(predictions_model3)
    class_name_model3 = class_names_model3[predicted_class_model3]

    return class_name_model1, class_name_model2, class_name_model3

def run_prediction(image_path):
    image = cv2.imread(image_path)
    input_frame = cv2.resize(image, (28, 28))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    predictions1 = model1.predict(input_frame)
    predicted_class1 = class_names_model1[np.argmax(predictions1)]

    input_frame_model2 = cv2.resize(image, (224, 224))
    input_frame_model2 = cv2.cvtColor(input_frame_model2, cv2.COLOR_BGR2RGB)
    input_frame_model2 = input_frame_model2 / 255.0
    input_frame_model2 = np.expand_dims(input_frame_model2, axis=0)
    predictions2 = model2.predict(input_frame_model2)
    predicted_class2 = class_names_model2[np.argmax(predictions2)]

    input_frame_model3 = cv2.resize(image, (224, 224))
    input_frame_model3 = cv2.cvtColor(input_frame_model3, cv2.COLOR_BGR2RGB)
    input_frame_model3 = input_frame_model3 / 255.0
    input_frame_model3 = np.expand_dims(input_frame_model3, axis=0)
    predictions3 = model3.predict(input_frame_model3)
    predicted_class3 = class_names_model3[np.argmax(predictions3)]

    return predicted_class1, predicted_class2, predicted_class3

@app.route('/image')
def image_prediction():
    image_path = "static/captured_image.jpg"  # 이미지 파일 경로
    predicted_class1, predicted_class2, predicted_class3 = run_prediction(image_path)
    return render_template('image.html', image_path=image_path,
                           predicted_class1=predicted_class1,
                           predicted_class2=predicted_class2,
                           predicted_class3=predicted_class3)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        input_frame = cv2.resize(frame, (28, 28))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        input_frame = input_frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)
        predictions1 = model1.predict(input_frame)
        predicted_class1 = np.argmax(predictions1)
        class_name1 = class_names_model1[predicted_class1]

        input_frame_model2 = cv2.resize(frame, (224, 224))
        input_frame_model2 = cv2.cvtColor(input_frame_model2, cv2.COLOR_BGR2RGB)
        input_frame_model2 = input_frame_model2 / 255.0
        input_frame_model2 = np.expand_dims(input_frame_model2, axis=0)
        predictions2 = model2.predict(input_frame_model2)
        predicted_class2 = np.argmax(predictions2)
        class_name2 = class_names_model2[predicted_class2]

        input_frame_model3 = cv2.resize(frame, (224, 224))
        input_frame_model3 = cv2.cvtColor(input_frame_model3, cv2.COLOR_BGR2RGB)
        input_frame_model3 = input_frame_model3 / 255.0
        input_frame_model3 = np.expand_dims(input_frame_model3, axis=0)
        predictions3 = model3.predict(input_frame_model3)
        predicted_class3 = np.argmax(predictions3)
        class_name3 = class_names_model3[predicted_class3]

        cv2.putText(frame, f'Model 1: {class_name1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Model 2: {class_name2}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Model 3: {class_name3}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/ACGPN')
def ACGPN():
    for folder in ['./ACGPN/results/test/warped_cloth/test_label',
                       './ACGPN/results/test/try-on/test_label',
                       './ACGPN/results/test/refined_cloth/test_label',
                       './static/result',
                       './static/cloth',
                       ]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    return render_template('ACGPN.html')

@app.route('/capture_body', methods=['POST'])
def capture_body():
    
    # 사진 촬영 (몸을 찍기)
    return_code, frame = camera.read()
    if not return_code:
        return jsonify(message="사진 촬영 실패.")



    # 사진 저장 (몸을 찍기)
    folder = app.config['UPLOAD_FOLDER'] = "./static/cloth"
    body_photo_name = 'body_photo.jpg'
    body_photo_path = os.path.join(folder, body_photo_name)
    cv2.imwrite(body_photo_path, frame)
    

    return jsonify(message="몸을 찍은 사진이 저장되었습니다.", photo_url=body_photo_name)





@app.route('/result')
def result():
        StartACGPN.main()

        # 새 파일 저장
        cloth_image_path = './ACGPN/results/test/try-on/test_label/img_result.png'
        if os.path.exists(cloth_image_path):
            shutil.move(cloth_image_path, "./static/result")
            return render_template('result.html')
        else:
            return "오류"

        



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/main1')
def main1():
    return render_template('main1.html')



@app.route('/main2')
def main2():
    return render_template('main2.html')

@app.route('/detection')
def main():
    return render_template('detection.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

def get_weather_code(city):
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr"
    }

    response = requests.get(api_endpoint, params=params)
    data = response.json()

    if data.get("cod") == 200:
        weather_code = data["weather"][0]["id"]
    else:
        weather_code = None

    return weather_code

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/weather', methods=['GET'])
def weather():
    city = "Seoul"  # 날씨를 확인할 도시 이름을 설정합니다.
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr"
    }

    response = requests.get(api_endpoint, params=params)
    data = response.json()

    if data.get("cod") == 200:
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
    else:
        weather_description = "날씨 정보 없음"
        temperature = 0
        humidity = 0
        
    # TTS로 음성 생성

    if temperature > 25:
        codi = "오늘은 더운 날씨입니다. 반팔과 반바지를 입고. 밝은 색상을 추천드립니다."
    elif 20 <= temperature <= 25:
        codi = "오늘은 쾌적한 날씨입니다. 얇은 가디건과 바지를 입고. 옅은 파스텔 톤의 색상을 추천드립니다."
    elif 15 <= temperature < 20:
        codi = "오늘은 약간 서늘한 날씨입니다. 가디건과 슬랙스를 추천합니다. 어두운 컬러를 추천드립니다."
    else:
        codi = "오늘은 춥습니다. 따뜻한 코트와 니트를 입고. 따뜻한 컬러를 추천드립니다."

    text = f"현재 {city}의 날씨는 {weather_description}이며, 온도는 {temperature} ℃이고 습도는 {humidity} %입니다. {codi}"
    tts = gTTS(text=text, lang='ko', slow=False)

    # 생성된 TTS를 파일로 저장
    audio_file_path = "static/weather_tts.mp3"
    tts.save(audio_file_path)

    weather_code = get_weather_code(city)

    return render_template('weather.html', tts_file=audio_file_path, city=city, description=weather_description, temperature=temperature, humidity=humidity)

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image_path = os.path.join('static', 'captured_image.jpg')
            image.save(image_path)

            # 이미지 전처리 및 모델 예측
            input_frame = cv2.imread(image_path)
            input_frame = cv2.resize(input_frame, (28, 28))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            input_frame = input_frame / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)
            predictions1 = model1.predict(input_frame)
            predicted_class1 = np.argmax(predictions1)
            class_name1 = class_names_model1[predicted_class1]

            input_frame_model2 = cv2.imread(image_path)
            input_frame_model2 = cv2.resize(input_frame_model2, (224, 224))
            input_frame_model2 = cv2.cvtColor(input_frame_model2, cv2.COLOR_BGR2RGB)
            input_frame_model2 = input_frame_model2 / 255.0
            input_frame_model2 = np.expand_dims(input_frame_model2, axis=0)
            predictions2 = model2.predict(input_frame_model2)
            predicted_class2 = np.argmax(predictions2)
            class_name2 = class_names_model2[predicted_class2]

            input_frame_model3 = cv2.imread(image_path)
            input_frame_model3 = cv2.resize(input_frame_model3, (224, 224))
            input_frame_model3 = cv2.cvtColor(input_frame_model3, cv2.COLOR_BGR2RGB)
            input_frame_model3 = input_frame_model3 / 255.0
            input_frame_model3 = np.expand_dims(input_frame_model3, axis=0)
            predictions3 = model3.predict(input_frame_model3)
            predicted_class3 = np.argmax(predictions3)
            class_name3 = class_names_model3[predicted_class3]

            return render_template('image.html', class_name1=class_name1, class_name2=class_name2, class_name3=class_name3)
        else:
            return 'No image found in request'
    else:
        return render_template('detection.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        image_path = os.path.join('static', 'captured_image.jpg')
        image.save(image_path)

        # 이미지 전처리 및 모델 예측
        input_frame = cv2.imread(image_path)
        input_frame = cv2.resize(input_frame, (28, 28))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        input_frame = input_frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)
        predictions1 = model1.predict(input_frame)
        predicted_class1 = np.argmax(predictions1)
        class_name1 = class_names_model1[predicted_class1]

        input_frame_model2 = cv2.imread(image_path)
        input_frame_model2 = cv2.resize(input_frame_model2, (224, 224))
        input_frame_model2 = cv2.cvtColor(input_frame_model2, cv2.COLOR_BGR2RGB)
        input_frame_model2 = input_frame_model2 / 255.0
        input_frame_model2 = np.expand_dims(input_frame_model2, axis=0)
        predictions2 = model2.predict(input_frame_model2)
        predicted_class2 = np.argmax(predictions2)
        class_name2 = class_names_model2[predicted_class2]

        input_frame_model3 = cv2.imread(image_path)
        input_frame_model3 = cv2.resize(input_frame_model3, (224, 224))
        input_frame_model3 = cv2.cvtColor(input_frame_model3, cv2.COLOR_BGR2RGB)
        input_frame_model3 = input_frame_model3 / 255.0
        input_frame_model3 = np.expand_dims(input_frame_model3, axis=0)
        predictions3 = model3.predict(input_frame_model3)
        predicted_class3 = np.argmax(predictions3)
        class_name3 = class_names_model3[predicted_class3]

        return render_template('detection.html', image_path=image_path, class_name1=class_name1, class_name2=class_names_model2[predicted_class2], class_name3=class_name3)
    else:
        return 'No image found in request'
    
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_file(filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
