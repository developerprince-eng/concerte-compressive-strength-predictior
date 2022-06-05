"""This Class is responsible for generating a model for RESC ML """
from __future__ import print_function, absolute_import, division
from operator import mod

import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.patches as mpatches
from keras_visualizer import visualizer
import os

os.getcwd()
os.listdir(os.getcwd())

class generate_model:
    def __init__(self):
        self.is_model = True

    @staticmethod
    def model1_thread(train_labels, train_data, test_labels, test_data):
        tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/relu_sequential_model", histogram_freq=1)

        model = Sequential([
                    keras.layers.Dense(8, activation=tf.nn.elu),
                    keras.layers.Dense(16, activation=tf.nn.elu),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(48, activation=tf.nn.elu),
                    keras.layers.Dense(48, activation=tf.nn.elu),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(16, activation=tf.nn.elu),
                    keras.layers.Dense(1)
                ])

        model.compile(loss='mean_squared_logarithmic_error',
                        optimizer='adam',
                        metrics=['mae'])
        model.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=1000, callbacks=[tb_callbacks])

        model.summary()
        visualizer(model, format='png', view=True)

        score = model.evaluate(x=test_data.values, y=test_labels.values, verbose=2)
        print(f'Loss for Sequential Model ==================================> {score[0]}')
        print(f'Mean Absolute Error for Sequential Model ==============================> {score[1]}' )
        model.predict(x=test_data.values).flatten()
        """ Save Model """
        model_json = model.to_json()
        with open("jsn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("jsn_model.h5")
        print("Saved model to disk")
        model.save('ccst_predictor_sequential_model.h5')
        CCST_model = keras.models.load_model('ccst_predictor_sequential_model.h5')
        predictions = CCST_model.predict(x=test_data.values)

        print(test_data.values)
        plt.plot(test_labels.values, color='red')
        plt.plot(predictions, color='green')
        red_patch = mpatches.Patch(color='red', label='Test Data for Model 1')
        green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 1')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()
        print("")

    @staticmethod
    def model2_thread(train_labels, train_data, test_labels, test_data):
            tb_callbacks2 = tf.keras.callbacks.TensorBoard(log_dir="logs/relu_model", histogram_freq=1)

            inputs = tf.keras.Input(shape=(8))
            x = tf.keras.layers.Dense(16, activation=tf.nn.relu6)(inputs)
            x1 = tf.keras.layers.Dense(16, activation=tf.nn.relu6)(x)
            x2 = tf.keras.layers.Dense(32, activation=tf.nn.relu6)(x1)
            x3 = tf.keras.layers.Dense(16, activation=tf.nn.relu6)(x2)
            x4 = tf.keras.layers.Dense(8, activation=tf.nn.relu6)(x3)
            outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu6)(x4)
            model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

            model2.compile(loss='mean_squared_logarithmic_error',
                optimizer='adam',
                metrics=['mae'])
            model2.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=10, callbacks=[tb_callbacks2])
            model2.summary()
            visualizer(model2, format='png', view=True)
            score2 = model2.evaluate(x=test_data.values, y=test_labels.values, verbose=2)

            print(f'Loss for Model ==================================> {score2[0]}')
            print(f'Mean Absolute Error for Model ==============================> {score2[1]}' )

            model2.predict(x=test_data.values)
            """ Save Model """
            model_json = model2.to_json()
            with open("jsn_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model2.save_weights("jsn_model_2.h5")
            print("Saved model to disk")
            model2.save('ccst_predictor_model.h5')
            CCST_model = keras.models.load_model('ccst_predictor_model.h5')
            predictions = CCST_model.predict(x=test_data.values)

            print(test_data.values)
            plt2.plot(test_labels.values, color='red')
            plt2.plot(predictions, color='green')
            red_patch = mpatches.Patch(color='red', label='Test Data for Model 2')
            green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 2')
            plt2.legend(handles=[red_patch, green_patch])
            plt2.show()
            print("")


    def __generate__(self, train_labels, train_data, test_labels, test_data):
        self.model1_thread(train_labels, train_data, test_labels, test_data)
        print("Done")


    def __normalize__(self, train_data, test_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        data = [train_data, test_data]
        return data

