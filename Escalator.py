#%%
##1. import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from collections import Counter
import tensorflow as tf
from tensorflow.keras import Model ,models, layers, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#%%
##2. Load & Split Train / Valid / Test Data => data_v2
# 이후 훈련, 검증, 테스트 용 데이터로 분리
x_train = np.load('./nomal-06_nomalized_sequence_dev1000002_Jan_EscalatorDB_2.npy', allow_pickle=True)      #normal 중 60%
x_train = x_train.astype(np.float64)
x_valid = np.load('./nomal-01-1_nomalized_sequence_dev1000002_Jan_EscalatorDB_2.npy', allow_pickle=True)    #normal 중 10%
x_valid = x_valid.astype(np.float64)
x_test = np.load('./nomal-01-2_nomalized_sequence_dev1000002_Jan_EscalatorDB_2.npy', allow_pickle=True)     #normal 중 10%
x_test = x_test.astype(np.float64)

print(x_train.shape)    #(240000, 16, 63)   
print(x_valid.shape)    #(40000, 16, 63)
print(x_test.shape)     #(40000, 16, 63)

# LSTM Autoencoder 학습 시에는 Normal(0) 데이터만으로 학습해야하기 때문에
# 데이터로 부터 Normal(0)과 Break(1) 데이터를 분리 => 이미 분리되어 있음
# For training the autoencoder, split 0 / 1
y_valid = np.load('/users/jsh/escalator/data_v4/abnomal-01-0_nomalized_sequence_dev1000002_Jan_EscalatorDB.npy', allow_pickle=True)     #normal 중 10%를 0~62 센서 중 [1,4,7] 센서의 값을 10% 증가시킨 이상치 파일 
y_valid = y_valid.astype(np.float64)
y_test = np.load('/users/jsh/escalator/data_v4/abnomal-01-1_nomalized_sequence_dev1000002_Jan_EscalatorDB.npy', allow_pickle=True)     #normal 중 10%를 0~62 센서 중 [1,4,7] 센서의 값을 10% 증가시킨 이상치 파일 
y_test = y_test.astype(np.float64)

print(y_valid.shape)    #(40000, 16, 63)
print(y_test.shape)     #(40000, 16, 63)



#%%
##3. Training LSTM Autoencoder
# 대칭 구조의 Staked Autoencoder 형태로 LSTM Autoencoder를 구성하여
# 정상 데이터로만 구성 된 데이터를 통해 총 200 epoch 학습
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)
epochs = 1
batch = 2
lr = 0.001
lstm_ae = models.Sequential()
timesteps = 16
n_features = 63
# Encoder
lstm_ae.add(layers.LSTM(256, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_ae.add(layers.LSTM(128, activation='relu', return_sequences=True))
lstm_ae.add(layers.LSTM(64, activation='relu', return_sequences=False))
lstm_ae.add(layers.Dense(20, activation= 'relu'))
lstm_ae.add(layers.RepeatVector(timesteps))
# Decoder
lstm_ae.add(layers.LSTM(64, activation='relu', return_sequences=True))
lstm_ae.add(layers.LSTM(128, activation='relu', return_sequences=True))
lstm_ae.add(layers.LSTM(256, activation='relu', return_sequences=True))
lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))   

lstm_ae.summary()

# compile
lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(lr))


# fit
history = lstm_ae.fit(x_train, x_train,
                     epochs=epochs, batch_size=batch,
                     validation_data=(x_test, x_test))

# train loss와 valid loss 모두 0.1 근처로 수렴
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('loss')
plt.show()


lstm_ae.save('lstm_ae_epoch1.h5')



#%%
##4. Threshold 
#정상의 최대값과 비정상의 최솟값의 2분의 1을 임계값으로 함
model = tf.keras.models.load_model('./lstm_ae_epoch1.h5')   #모델 로드
valid_x_predictions = model.predict(x_valid)
# valid_x_predictions = lstm_ae.predict(x_valid)
mse_0 = []
for i in range(x_valid.shape[0]):
    mse_0.append(np.square(x_valid[i]-valid_x_predictions[i]).mean())
mse_0 = np.array(mse_0)
print(mse_0.shape)  #(40000,)

valid_y_predictions = model.predict(y_valid)
mse_1 = []
for i in range(y_valid.shape[0]):
    mse_1.append(np.square(y_valid[i]-valid_y_predictions[i]).mean())
mse_1 = np.array(mse_1)
print(mse_1.shape)  #(40000,)


print(valid_x_predictions.shape)    #(40000, 16, 63)
print(valid_y_predictions.shape)    #(40000, 16, 63)
print(mse_0.shape)                  #(40000,)
print(mse_1.shape)                  #(40000,)

max_0 = np.max(mse_0) #정상데이터 최대
min_1 = np.min(mse_1) #비정상 데이터 최소
print(max_0)    #0.02568725202085197
print(min_1)    #3.0145217271751704e-05

#2분의 1
threshold_fixed = (max_0+min_1)/2
print('threshold: ',threshold_fixed)    #threshold: 0.01285869861906186

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold_fixed:
        counter += 1
print(counter)        #37008

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold_fixed:
        counter += 1
print(counter)        #3072


#4분의 1
threshold_fixed_fixed = (threshold_fixed+min_1)/2
print('threshold: ',threshold_fixed_fixed)    #threshold: 0.006444421918166806

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold_fixed_fixed:
        counter += 1
print(counter)        #9493

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold_fixed_fixed:
        counter += 1
print(counter)        #30639



#8분의 1
threshold_fixed_fixed_fixed = (threshold_fixed_fixed+min_1)/2
print('threshold: ',threshold_fixed_fixed_fixed)    #threshold:  0.003237283567719279

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold_fixed_fixed_fixed:
        counter += 1
print(counter)        #7336

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold_fixed_fixed_fixed:
        counter += 1
print(counter)        #32725



#16분의 1
threshold_fixed_fixed_fixed_fixed = (0.003237283567719279+min_1)/2
print('threshold: ',threshold_fixed_fixed_fixed_fixed)    #threshold: 0.0016337143924955153

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold_fixed_fixed_fixed_fixed:
        counter += 1
print(counter)        #7335

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold_fixed_fixed_fixed_fixed:
        counter += 1
print(counter)        #32727




mean_0 = np.mean(mse_0) #정상데이터 평균
mean_1 = np.mean(mse_1) #비정상 데이터 평균
print(mean_0)    #0.007877538780372915
print(mean_1)    #0.007939296289986313

#각각의 평균의 2분의 1
threshold_fixed_average = (mean_0+mean_1)/2
print('threshold: ',threshold_fixed_average)    #threshold: 0.007908417535179613

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold_fixed_average:
        counter += 1
print(counter)        #15147

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold_fixed_average:
        counter += 1
print(counter)        #25154



#Precision Recall Curve
mse = np.concatenate((mse_0,mse_1),axis=0)
print(mse)      #[0.0124643  0.01427961 0.01488178 ... 0.01073279 0.01385324 0.00909304]
print(mse.shape)       #(80000,)


true_class = []
for i in range(0, 40000):
    true_class.append(0)
for i in range(40000, 80000):
    true_class.append(1)

error_df = pd.DataFrame({'Reconstruction_error':mse, 'True_class':true_class})
error_df = error_df.sample(frac=1).reset_index(drop=True)

precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])

plt.figure(figsize=(8,5))
plt.plot(threshold_rt, precision_rt[1:], label='Precision')
plt.plot(threshold_rt, recall_rt[1:], label='Recall')
plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# best position of threshold
index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r][0]
print('precision: ',precision_rt[index_cnt],', recall: ',recall_rt[index_cnt])

# fixed Threshold
threshold = threshold_rt[index_cnt]
print('threshold: ',threshold)      
# precision:  0.505225 , recall:  0.505225
# threshold:  0.008813259503079639

counter=0
for i in range(mse_0.shape[0]):
    if mse_0[i] < threshold:
        counter += 1
print(counter)        #20209

counter=0
for i in range(mse_1.shape[0]):
    if mse_1[i] > threshold:
        counter += 1
print(counter)        #20208



#%%
##5. Predict Test 
# 테스트 셋 적용
# 학습하였던 LSTM Autoencoder 모델을 통해 테스트 셋을 예측 후 재구성 손실 계산
# 그 후 위에서 찾은 threshold를 적용하여 Normal과 Break 구분
test_x_predictions = model.predict(x_test)
mse_00 = []
for i in range(x_test.shape[0]):
    mse_00.append(np.square(x_test[i]-test_x_predictions[i]).mean())
mse_00 = np.array(mse_00)

test_y_predictions = model.predict(y_test)
mse_01 = []
for i in range(y_test.shape[0]):
    mse_01.append(np.square(y_test[i]-test_y_predictions[i]).mean())
mse_01 = np.array(mse_01)


print(test_x_predictions.shape)    #(40000, 16, 63)
print(test_y_predictions.shape)    #(40000, 16, 63)
print(mse_00.shape)                  #(40000,)
print(mse_01.shape)                  #(40000,)
predict_df = np.concatenate((mse_00,mse_01),axis=0)
print(predict_df.shape)     #(80000,)

true_class = []
for i in range(0, 40000):
    true_class.append(0)
for i in range(40000, 80000):
    true_class.append(1)

# prediction_df = pd.DataFrame({'predict': mse_00,'True_class': true_class})
prediction_df = pd.DataFrame({'predict': predict_df,'True_class': true_class})
print(prediction_df)
prediction_df = prediction_df.sample(frac=1).reset_index(drop=True)
print(prediction_df)
groups = prediction_df.groupby('True_class')
fig, ax = plt.subplots()


# ###normal
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()



# ###2분의 1
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()

# counter=0
# for i in range(mse_00.shape[0]):
#     if mse_00[i] < threshold_fixed:
#         counter += 1
# print(counter)        #37021


# counter=0
# for i in range(mse_01.shape[0]):
#     if mse_01[i] > threshold_fixed:
#         counter += 1
# print(counter)        #3073



# ###4분의 1
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()

# counter=0
# for i in range(mse_00.shape[0]):
#     if mse_00[i] < threshold_fixed_fixed:
#         counter += 1
# print(counter)        #9433

# counter=0
# for i in range(mse_01.shape[0]):
#     if mse_01[i] > threshold_fixed_fixed:
#         counter += 1
# print(counter)        #30770


# ###8분의 1
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed_fixed_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()

# counter=0
# for i in range(mse_00.shape[0]):
#     if mse_00[i] < threshold_fixed_fixed_fixed:
#         counter += 1
# print(counter)        #7263

# counter=0
# for i in range(mse_01.shape[0]):
#     if mse_01[i] > threshold_fixed_fixed_fixed:
#         counter += 1
# print(counter)        #32746



# ###16분의 1
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed_fixed_fixed_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()

# counter=0
# for i in range(mse_00.shape[0]):
#     if mse_00[i] < threshold_fixed_fixed_fixed_fixed:
#         counter += 1
# print(counter)        #7261

# counter=0
# for i in range(mse_01.shape[0]):
#     if mse_01[i] > threshold_fixed_fixed_fixed_fixed:
#         counter += 1
# print(counter)        #32746



# ###평균
# for name, group in groups:
#     ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='',label= "Abnormal" if name == 1 else "Normal")
# ax.hlines(threshold_fixed_average, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Threshold for different classes")
# plt.ylabel("threshold")
# plt.xlabel("Data point index")
# plt.show()

# counter=0
# for i in range(mse_00.shape[0]):
#     if mse_00[i] < threshold_fixed_average:
#         counter += 1
# print(counter)        #15154

# counter=0
# for i in range(mse_01.shape[0]):
#     if mse_01[i] > threshold_fixed_average:
#         counter += 1
# print(counter)        #25144




###Precision Recall Curve
for name, group in groups:
    ax.plot(group.index, group.predict, marker='o', ms=1.5, linestyle='', label= "Break" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

counter=0
for i in range(mse_00.shape[0]):
    if mse_00[i] < threshold:
        counter += 1
print(counter)        #20250

counter=0
for i in range(mse_01.shape[0]):
    if mse_01[i] > threshold:
        counter += 1
print(counter)        #20224




#%%
##6. Evaluation
# confusion matrix
# 테스트 셋에 대한 재구성 손실을 threshold 기준으로 0/1로 나누고 이를 confusion matrix로 표현
# classification by threshold
LABELS = ['Normal', 'Abnormal']
# pred_y = [1 if e > threshold_fixed else 0 for e in prediction_df['predict'].values]   #2분의 1   
# pred_y = [1 if e > threshold_fixed_fixed else 0 for e in prediction_df['predict'].values]   #4분의 1  
# pred_y = [1 if e > threshold_fixed_fixed_fixed else 0 for e in prediction_df['predict'].values]   #8분의 1  
# pred_y = [1 if e > threshold_fixed_fixed_fixed_fixed else 0 for e in prediction_df['predict'].values]   #16분의 1  
# pred_y = [1 if e > threshold_fixed_average else 0 for e in prediction_df['predict'].values]   #각각의 평균의 2분의 1  
pred_y = [1 if e > threshold else 0 for e in prediction_df['predict'].values]   #Precision Recall Curve 

conf_matrix = metrics.confusion_matrix(prediction_df['True_class'], pred_y)
plt.figure(figsize=(7, 7))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class'); plt.ylabel('True -Class')
plt.show()


# ROC Curve and AUC
false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(prediction_df['True_class'], prediction_df['predict'])
roc_auc = metrics.auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
plt.show()


#%%
##7. Result
# 최종적으로 테스트 셋에 대한 재구성 손실을 threshold를 통해 구분한
# pred_y의 마지막 30번째 (timestep만큼)을 출력하여 예측 결과 확인 가능
# print(pred_y[-30:])  # []