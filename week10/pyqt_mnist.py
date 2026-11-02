import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

dmlp = Sequential()
dmlp.add(Dense(units=1024, activation='relu', input_shape=(784,)))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(Dense(units=10, activation='softmax'))

dmlp.compile(loss='categorical_crossentropy',   # 모델 컴파일(모델 학습 방법 결정)
             optimizer=Adam(learning_rate=0.0001), metrics=['accuracy']) # metrics는 학습/평가 중에 화면에 보여주고 기록할 평가 지표를 지정하는 compile()의 파라미터
hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=50,    # batch_size: 한 번에 처리할 샘플의 수
                validation_data=(x_test, y_test), verbose=2)
print('정확률=', dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100) # 모델 성능 평가

dmlp.save('dmlp_trained.h5')    # 모델 저장

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])  # 손실 변화 그래프, 학습 손실, 검증 손실 겹쳐서 그림
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'])
plt.grid()
plt.show()