import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"],
                                                   cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print(tf.__version__)
print('사용된 장치 수: {}'.format(mirrored_strategy.num_replicas_in_sync))


class Dataset:
    def __init__(self, data_dir, input_column):
        raw_dataset = pd.read_csv(data_dir, names=input_column,
                                  na_values="?", comment='\t',
                                  skipinitialspace=True)

        self.dataset = raw_dataset.copy()

    def norm(self, x):
        data_stats = self.dataset.describe()
        data_stats = data_stats.transpose()
        return (x - data_stats['min']) / ((data_stats['max']) - data_stats['min'])

    def deNorm(self, x, x_names):
        data_stats = self.dataset.describe()
        data_stats = data_stats.transpose()
        for x_name in x_names:
            x["Pred_" + x_name] = \
                x["Pred_" + x_name] * ((data_stats['max'][x_name]) -
                                     data_stats['min'][x_name]) + data_stats['min'][x_name]
        return x

    def batchData(self, random_state, outputs, is_norm=True):
        train_dataset = self.dataset.sample(frac=0.85, random_state=random_state)
        test_dataset = self.dataset.drop(train_dataset.index)
        train_y = train_dataset[outputs]
        test_y = test_dataset[outputs]
        if is_norm:
            train_dataset = self.norm(train_dataset)
            test_dataset = self.norm(test_dataset)
        return train_dataset.reset_index(drop=True), \
               train_y.reset_index(drop=True), \
               test_dataset.reset_index(drop=True), \
               test_y.reset_index(drop=True)


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


class Algorithm:
    def __init__(self, hparams):
        self.inputs = hparams['input']
        self.outputs = hparams['output']
        self.dir_data = hparams['dir_data']
        self.dir_model =hparams['dir_model']
        self.dir_log = hparams['dir_log']
        self.dir_result = hparams['dir_result']
        self.epochs = hparams['epochs']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.lr = hparams['lr']
        self.random_state = hparams['random_state']
        self.batch_size = hparams['batch_size']
        self.data = Dataset(self.dir_data, hparams['input'])

    def trainRun(self, is_trans=False):
        train_dataset, train_y, test_dataset, test_y = self.data.batchData(self.random_state, self.outputs)

        with mirrored_strategy.scope():
            models = NetworkModel(len(self.inputs), len(self.outputs))
            if is_trans:
                models.load_weights(self.dir_model)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        models.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mse', 'mae']
        )

        history = models.fit(
            x=train_dataset[self.inputs],
            y=train_dataset[self.outputs],
            epochs=self.epochs,
            validation_data=(
                test_dataset[self.inputs],
                test_dataset[self.outputs]),
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[self.callBacks()]
        )
        models.save(self.dir_model)

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        self.plotHistory(history)

    def testRun(self):
        train_dataset, train_y, test_dataset, test_y = self.data.batchData(self.random_state, self.outputs)

        with mirrored_strategy.scope():
            models = tf.keras.models.load_model(self.dir_model, compile=True)

        train_mse = models.evaluate(
            x=train_dataset[self.inputs],
            y=train_dataset[self.outputs],
            verbose=2,
            batch_size=1
        )

        test_mse = models.evaluate(
            x=test_dataset[self.inputs],
            y=test_dataset[self.outputs],
            verbose=2,
            batch_size=1
        )

        mse_mean = tf.reduce_mean(train_mse)
        print("Training 세트의 평균 절대 오차: {:5.2f} mm".format(mse_mean))
        mse_mean = tf.reduce_mean(test_mse)
        print("Test 세트의 평균 절대 오차: {:5.2f} mm".format(mse_mean))

        self.predicts = []
        for output_name in self.outputs:
            self.predicts.append("Pred_" + output_name)

        train_predictions = pd.DataFrame(
            models.predict(train_dataset[self.inputs]),
            columns=self.predicts
        )
        test_predictions = pd.DataFrame(
            models.predict(test_dataset[self.inputs]),
            columns=self.predicts
        )

        train_denorm = self.data.deNorm(train_predictions, self.outputs)
        test_denorm = self.data.deNorm(test_predictions, self.outputs)

        final_result = pd.concat([test_y, test_denorm], axis=1)
        self.plotR2(train_y, train_denorm)
        self.plotR2(test_y, test_denorm)
        np.savetxt(self.dir_result, final_result, delimiter=',')

    def makedDir(self, dir_name):
        root_logdir = os.path.join(os.curdir, dir_name)
        sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(root_logdir, sub_dir_name)

    def callBacks(self):
        TB_dir = self.makedDir(self.dir_log)
        TB = tf.keras.callbacks.TensorBoard(
            TB_dir, histogram_freq=100
        )
        CP = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.dir_model,
            save_weights_only=True,
            verbose=1,
            save_freq=3000)
        return TB, CP

    def plotHistory(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure(figsize=(8, 12))
        plt.subplot(1, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('MeanSquareError [$Effect^2$]')
        plt.plot(hist['epoch'], hist['loss'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
        plt.legend()
        plt.show()

    def plotR2(self, labels, predict):
        line_max = max(labels.max())
        plt.scatter(labels, predict)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.text(100, 50, 'R-squared: {:1.4f}'.format(r2_score(labels, predict)))
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        _ = plt.plot([0, line_max], [0, line_max])
        plt.show()
        print('r2_score: {}'.format(r2_score(labels, predict)))
