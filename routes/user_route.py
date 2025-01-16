from flask import Blueprint, request, jsonify
import pymysql
import pymysql.cursors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np
from db import get_db_connection


user_route = Blueprint('user_route', __name__)


@user_route.route("/add-user", methods=['post'])
def addUser():

    data = request.json

    id=data.get('id')
    pw=data.get('pw')
    nick=data.get('nick')
    type=data.get('type')
    address=data.get('address')

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    query = """
    INSERT INTO user
    (id,pw,nick,type,address,created_date)
    VALUES
    (%s,md5(%s),%s,%s,%s,sysdate())
    """
    cursor.execute(query, (id, pw, nick, type, address))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"message": "ok"})

@user_route.route("/detail-user")
def detailUser():
    user_idx = request.args.get('user_idx')
    try:
        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM user where user_idx=%s", (user_idx,))
        users = cursor.fetchall()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        cursor.close()
        connection.close()



@user_route.route("/study")
def study():
    # 데이터 로드
    diabetes = load_diabetes()
    print(diabetes.DESCR)
    X = diabetes.data
    y = diabetes.target

    # 데이터셋을 학습 세트와 테스트 세트로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # 모델 초기화 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    print(y_pred)

    # 결과 평가 학습한 데이터 와 실제 결과치 비교하기
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # # 예측 결과 시각화 (실제 값 vs 예측 값)
    # plt.scatter(y_test, y_pred)
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.title('Actual vs Predicted')
    # plt.show()
    
    return jsonify({'mse':mse,'r2':r2})




@user_route.route("/saveUser", methods=['POST'])
def saveUser():

    id = request.args.get('id')
    pw = request.args.get('pw')

    

    return 'ok'

if __name__ == "__main__":
    user_route.run(debug=True)