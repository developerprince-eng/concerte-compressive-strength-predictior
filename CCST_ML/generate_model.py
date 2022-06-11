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
from threading import Thread
import os

os.getcwd()
os.listdir(os.getcwd())

class generate_model:
    def __init__(self):
        self.is_model = True

    @staticmethod
    def model1_thread(train_labels, train_data, test_labels, test_data):
        tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/elu_sequential1_model", histogram_freq=1)

        model = Sequential([
                    keras.layers.Dense(8, activation=tf.nn.elu),
                    keras.layers.Dense(16, activation=tf.nn.elu),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(48, activation=tf.nn.elu),
                    keras.layers.Dense(48, activation=tf.nn.elu),
                    keras.layers.Dense(96, activation=tf.nn.elu),
                    keras.layers.Dense(192, activation=tf.nn.elu),
                    keras.layers.Dense(96, activation=tf.nn.elu),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(16, activation=tf.nn.elu),
                    keras.layers.Dense(1)
                ])

        model.compile(loss='mean_squared_logarithmic_error',
                        optimizer='adam',
                        metrics=['mae', 'mse', 'mape','msle', 'logcosh'])
        model.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=1000, callbacks=[tb_callbacks])

        model.summary()
        visualizer(model,filename='graph_1' ,format='png', view=True)

        score = model.evaluate(x=test_data.values, y=test_labels.values, verbose=2)
        print(f'Loss for Sequential Model 1 ==================================> {score[0]}')
        print(f'Mean Absolute Error for Sequential Model 1 ==============================> {score[1]}' )
        model.predict(x=test_data.values).flatten()
        """ Save Model """
        model_json = model.to_json()
        with open("jsn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("jsn_model.h5")
        print("Saved model to disk")
        model.save('ccst_predictor_sequential_model1.h5')
        CCST_model = keras.models.load_model('ccst_predictor_sequential_model1.h5')
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
        tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/relu_sequential2_model", histogram_freq=1)

        model = Sequential([
                    keras.layers.Dense(8, activation=tf.nn.relu),
                    keras.layers.Dense(16, activation=tf.nn.relu),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(48, activation=tf.nn.relu),
                    keras.layers.Dense(48, activation=tf.nn.relu),
                    keras.layers.Dense(40, activation=tf.nn.relu6),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(16, activation=tf.nn.relu),
                    keras.layers.Dense(1)
                ])

        model.compile(loss='mean_squared_logarithmic_error',
                        optimizer='adam',
                        metrics=['mae', 'mse', 'mape', 'msle', 'logcosh'])
        model.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=1000, callbacks=[tb_callbacks])

        model.summary()
        visualizer(model,filename='graph_2' ,format='png', view=True)

        score = model.evaluate(x=test_data.values, y=test_labels.values, verbose=2)
        print(f'Loss for Sequential Model 2 ==================================> {score[0]}')
        print(f'Mean Absolute Error for Sequential Model 2 ==============================> {score[1]}' )
        model.predict(x=test_data.values).flatten()
        """ Save Model """
        model_json = model.to_json()
        with open("jsn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("jsn_model.h5")
        print("Saved model to disk")
        model.save('ccst_predictor_sequential_model2.h5')
        CCST_model = keras.models.load_model('ccst_predictor_sequential_model2.h5')
        predictions = CCST_model.predict(x=test_data.values)

        print(test_data.values)
        plt.plot(test_labels.values, color='red')
        plt.plot(predictions, color='green')
        red_patch = mpatches.Patch(color='red', label='Test Data for Model 2')
        green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 2')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()
        print("")

    @staticmethod
    def model3_thread(train_labels, train_data, test_labels, test_data):
        tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/leaky_relu_sequential3_model", histogram_freq=1)

        model = Sequential([
                    keras.layers.Dense(8, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(16, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(32, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(40, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(40, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(32, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(16, activation=tf.nn.leaky_relu),
                    keras.layers.Dense(1)
                ])

        model.compile(loss='mean_squared_logarithmic_error',
                        optimizer='adam',
                        metrics=['mae', 'mse', 'mape', 'msle', 'logcosh'])
        model.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=1000, callbacks=[tb_callbacks])

        model.summary()
        visualizer(model,filename='graph_3' ,format='png', view=True)

        score = model.evaluate(x=test_data.values, y=test_labels.values, verbose=2)
        print(f'Loss for Sequential Model 3 ==================================> {score[0]}')
        print(f'Mean Absolute Error for Sequential Model 3 ==============================> {score[1]}' )
        model.predict(x=test_data.values).flatten()
        """ Save Model """
        model_json = model.to_json()
        with open("jsn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("jsn_model.h5")
        print("Saved model to disk")
        model.save('ccst_predictor_sequential_model3.h5')
        CCST_model = keras.models.load_model('ccst_predictor_sequential_model3.h5')
        predictions = CCST_model.predict(x=test_data.values)

        print(test_data.values)
        plt.plot(test_labels.values, color='red')
        plt.plot(predictions, color='green')
        red_patch = mpatches.Patch(color='red', label='Test Data for Model 3')
        green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 3')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()
        print("")

    @staticmethod
    def model4_thread(train_labels, train_data, test_labels, test_data):
        tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/relu6_sequential4_model", histogram_freq=1)

        model = Sequential([
                    keras.layers.Dense(8, activation=tf.nn.relu6),
                    keras.layers.Dense(16, activation=tf.nn.relu6),
                    keras.layers.Dense(32, activation=tf.nn.relu6),
                    keras.layers.Dense(48, activation=tf.nn.relu6),
                    keras.layers.Dense(52, activation=tf.nn.relu6),
                    keras.layers.Dense(1)
                ])

        model.compile(loss='mean_squared_logarithmic_error',
                        optimizer='adam',
                        metrics=['mae', 'mse', 'mape', 'msle', 'logcosh'])
        model.fit(x=train_data.values, y=train_labels.values,batch_size=10, epochs=1000, callbacks=[tb_callbacks])

        model.summary()
        visualizer(model,filename='graph_4' ,format='png', view=True)

        score = model.evaluate(x=test_data.values, y=test_labels.values, verbose=2)
        print(f'Loss for Sequential Model 4 ==================================> {score[0]}')
        print(f'Mean Absolute Error for Sequential Model 4 ==============================> {score[1]}' )
        model.predict(x=test_data.values).flatten()
        """ Save Model """
        model_json = model.to_json()
        with open("jsn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("jsn_model.h5")
        print("Saved model to disk")
        model.save('ccst_predictor_sequential_model4.h5')
        CCST_model = keras.models.load_model('ccst_predictor_sequential_model4.h5')
        predictions = CCST_model.predict(x=test_data.values)

        print(test_data.values)
        plt.plot(test_labels.values, color='red')
        plt.plot(predictions, color='green')
        red_patch = mpatches.Patch(color='red', label='Test Data for Model 4')
        green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 4')
        plt.legend(handles=[red_patch, green_patch])
        plt.show()
        print("")
    @staticmethod

    def model5_thread(train_labels, train_data, test_labels, test_data):
            tb_callbacks2 = tf.keras.callbacks.TensorBoard(log_dir="logs/relu6_model5", histogram_freq=1)

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
                metrics=['mae', 'mse', 'mape', 'msle', 'logcosh'])
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
            red_patch = mpatches.Patch(color='red', label='Test Data for Model 5')
            green_patch = mpatches.Patch(color='green', label='Predictated Data for Model 5')
            plt2.legend(handles=[red_patch, green_patch])
            plt2.show()
            print("")


    def __generate__(self, train_labels, train_data, test_labels, test_data):

        p1 = Thread(target=self.model1_thread, args=(train_labels, train_data, test_labels, test_data))
        p1.start()
        p2 = Thread(target=self.model2_thread, args=(train_labels, train_data, test_labels, test_data))
        p2.start()
        p3 = Thread(target=self.model3_thread, args=(train_labels, train_data, test_labels, test_data))
        p3.start()
        p4 = Thread(target=self.model4_thread, args=(train_labels, train_data, test_labels, test_data))
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        print("Done")


    def __normalize__(self, train_data, test_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        data = [train_data, test_data]
        return data

