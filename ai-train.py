# 필수 라이브러리 가져오기
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/zzhining/python_ml_dl/main/dataset/concrete.csv')


# 데이터 확인
print(df.shape)
print(df.head())

# 특성과 레이블 분리
X = df.drop(columns=['CompressiveStrength'])  # 입력 데이터 (특성)
y = df['CompressiveStrength']  # 출력 데이터 (레이블)

# 2. 훈련과 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 표준화 (StandardScaler 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# 4. 인공신경망(ANN) 모델 생성
# 층을 순차적으로, 은닉1 -> 은닉2 -> 출력
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # 첫 번째 은닉층
    tf.keras.layers.Dense(32, activation='relu'),  # 두 번째 은닉층 relu : 보편적으로 relu를 많이 쓴다.
    # 출력층 (회귀 문제이므로 출력 뉴런은 1개, 분류문제면 타켓 개수 만큼 설정.)
    tf.keras.layers.Dense(1)  
    
])

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='mse') 
# adam : 경사하강법중 하나로 학습을 개선하는 방식으로 가장 많이 쓴다.
# mse : 실제 데이터와 평균과의 차이를 확연히 보기 위해 제곱한 값으로 오차를 줄여나가는 방식 (평균 제곱 오차)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
# early stop : 반복을 100번해도 값이 97, 1000번해도 값이 97이면 반복을 여러번 할 필요 없이 조기 종료해도 되는 것

# 6. 모델 학습 - 학습결과를 history변수에 저장
history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=200, # epoch : 반복 횟수 
    validation_split=0.2, 
    callbacks = [early_stopping])



# 7. 학습 결과 시각화 (loss와 val_loss 그래프)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# 8. 테스트 데이터로 모델 평가
y_pred = model.predict(X_test_scaled)

# 9. MSE(평균 제곱 오차) 계산
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# MSE 결과 값이 0에 가까울수록 모델링이 잘된 결과이지만 데이터의 최대, 최소값에 따라 결과 해석이 달라진다
# 데이터 값이 70정도 인데, MSE가 40이면 좋은 결과는 아니다 중간정도 된다

# 10. 실제 값과 예측 값 비교
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(comparison_df.head())

# 11. 모델 저장
# model.save('concrete_strength_model.keras')


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib

# # 데이터 생성 (예시 데이터)
# np.random.seed(42)
# data = pd.DataFrame({
#     'area': np.random.randint(50, 200, 500),
#     'rooms': np.random.randint(1, 6, 500),
#     'year_built': np.random.randint(1980, 2024, 500),
#     'income': np.random.randint(3000, 12000, 500),
#     'school_rating': np.random.randint(1, 10, 500),
#     'transit_score': np.random.randint(1, 10, 500),
#     'price': np.random.randint(20000, 100000, 500)
# })

# # 독립 변수와 종속 변수 설정
# X = data[['area', 'rooms', 'year_built', 'income', 'school_rating', 'transit_score']]
# y = data['price']

# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 모델 1: 선형 회귀
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # 선형 회귀 예측
# y_pred_lin = lin_reg.predict(X_test)

# # 선형 회귀 평가
# mse_lin = mean_squared_error(y_test, y_pred_lin)
# r2_lin = r2_score(y_test, y_pred_lin)
# print("Linear Regression - Mean Squared Error:", mse_lin)
# print("Linear Regression - R2 Score:", r2_lin)

# # 모델 2: 랜덤 포레스트 회귀
# rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)
# rf_reg.fit(X_train, y_train)

# # 랜덤 포레스트 예측
# y_pred_rf = rf_reg.predict(X_test)

# # 랜덤 포레스트 평가
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)
# print("Random Forest Regression - Mean Squared Error:", mse_rf)
# print("Random Forest Regression - R2 Score:", r2_rf)

# # 예측 예시 (테스트 데이터 중 일부를 사용)
# sample_data = X_test.iloc[:5]
# predictions_lin = lin_reg.predict(sample_data)
# predictions_rf = rf_reg.predict(sample_data)

# print("Sample Data Predictions (Linear Regression):", predictions_lin)
# print("Sample Data Predictions (Random Forest Regression):", predictions_rf)


# joblib.dump(lin_reg, 'house_price_model-lin.pkl')
# joblib.dump(rf_reg, 'house_price_model-rf.pkl')