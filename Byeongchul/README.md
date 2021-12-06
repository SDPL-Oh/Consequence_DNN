# Consequence_DNN
Accident/Failure Prediction

## 실행방법
```
python __main__.py
```
## 데이터 파일 저장 경로
```
Byeongchul/data/acciden_train.csv or Consequence_211104_mod.csv
```
## 알고리즘 옵션은 __main__.py에서 변경 가능
모델 변경의 경우, model.py에서 변경 가능
```
class NetworkModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    10, activation='relu',
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    input_shape=(input_size,)),
                tf.keras.layers.Dense(
                    10, activation='relu',
                    kernel_initializer=tf.keras.initializers.HeNormal()),
                tf.keras.layers.Dense(
                    5, activation='relu',
                    kernel_initializer=tf.keras.initializers.HeNormal()),
                tf.keras.layers.Dense(output_size, activation=None)
            ],
            name="Densenet",
        )

    def call(self, inputs):
        return self.logic(inputs)
```
