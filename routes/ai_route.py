from flask import render_template, request, jsonify, Blueprint
import pymysql
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from sklearn.preprocessing import LabelEncoder
from db import get_db_connection

ai_route = Blueprint('ai',__name__)


# 모델 로드
lin_reg = joblib.load('house_price_model-lin.pkl')
rf = joblib.load('house_price_model-rf.pkl')
cat_dog_model = tf.keras.models.load_model('cats_and_dogs_classifier.h5')


car_damage_model = tf.keras.models.load_model('car_damage_classifier.h5')
damage_classes = ["Breakage", "Crushed", "Scratch", "Separated"]
label_encoder = LabelEncoder()
label_encoder.fit(damage_classes)

# 비용 데이터 정의
costs = {
    "Breakage": 200000,
    "Crushed": 100000,
    "Scratch": 50000,
    "Separated": 150000,
}

# 이미지 전처리 함수
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # 모델에 맞는 크기로 조정
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 추가
    img_array = img_array / 255.0  # 스케일 조정
    return img_array

# 차량 손상 예측
@ai_route.route("/predict-car-damage", methods=['post'])
def predict_damage():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # 이미지 파일 저장
    image_file = request.files['image']
    file_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(file_path)

    # 이미지 전처리 및 예측
    try:
        preprocessed_img = load_and_preprocess_image(file_path)
        prediction = car_damage_model.predict(preprocessed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]  # 레이블 변환

        # 비용 산정
        cost = costs.get(predicted_label, "비용 정보 없음")

        # 결과 반환
        result = {
            "damage": predicted_label,
            "cost": cost
        }

        # 처리 후 임시 파일 삭제
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"예외 발생: {str(e)}"}), 500





# 고양이, 강아지
@ai_route.route("/predict-cat-dog", methods=['post']) # 사진을 저장할 수 없기 때문에 get 사용 X / 데이터베이스 말고 서버에 저장
def predictCatDog():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # 이미지 파일 저장
    image_file = request.files['image']
    file_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(file_path)

    # 이미지 전처리 및 예측
    try:
        preprocessed_img = load_and_preprocess_image(file_path)
        prediction = cat_dog_model.predict(preprocessed_img)

        # 결과 반환
        if prediction[0] > 0.5:
            result = {
                "label": "dog",
                "confidence": float(prediction[0][0])
            }
        else:
            result = {
                "label": "cat",
                "confidence": float(1 - prediction[0][0])
            }

        # 처리 후 임시 파일 삭제
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500






@ai_route.route("/add-house", methods=['post'])
def addHouse():

    data = request.json

    area=data.get('area')
    rooms=data.get('rooms')
    year=data.get('year')
    income=data.get('income')
    school_rating=data.get('school_rating')
    transit_score=data.get('transit_score')
    pred_lin=data.get('pred_lin')
    pred_rf=data.get('pred_rf')

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    query = """
    INSERT INTO house
    (area,rooms,year,income,school_rating,transit_score,pred_lin,pred_rf,created_date)
    VALUES
    (%s,%s,%s,%s,%s,%s,%s,%s,sysdate())
    """
    cursor.execute(query, (area, rooms, year, income, school_rating, transit_score, pred_lin, pred_rf))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"message": "ok"})



@ai_route.route("/predict-house-price", methods=['get'])
def predicHousePrice():

    area = request.args.get('area')
    rooms = request.args.get('rooms')
    year = request.args.get('year')
    income = request.args.get('income')
    school_rating = request.args.get('school_rating')
    transit_score= request.args.get('transit_score')

    features = np.array([[
        int(area),
        int(rooms),
        int(year),
        int(income),
        int(school_rating),
        int(transit_score)
    ]])
    lin_reg_pred = lin_reg.predict(features)[0]
    rf_reg_pred = rf.predict(features)[0]
    return jsonify({
        "message":"ok",
        "price_linear_prediction": float(lin_reg_pred),
        "price_random_forest_prediction": float(rf_reg_pred)
    })


