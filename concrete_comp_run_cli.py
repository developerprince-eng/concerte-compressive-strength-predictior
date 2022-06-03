# Developer Prince - Concrete Compressive Strength Regression Machine Learning Demo
""" This is Regression problem which make use of Data set with 8 Features inorder to Determine Concrete Compressve Strength(MPa, Mega Pascals) """

from __future__ import print_function, absolute_import, division

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    if len(sys.argv) == 10:

        if sys.argv[1] == 'compile':
            while True:

                    test_data = np.array([[float(sys.argv[2]), float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9])]])

                    CCST_model = tf.keras.models.load_model('CCST_predictor.h5')

                    predictions = CCST_model.predict(x=test_data)
                    results = predictions.tolist()
                    json_obj = json.dumps({'ccst':results[0][0]})
                    print(json_obj)
                    break

        elif sys.argv[1] != 'compile':
            json_obj = json.dumps({'error':5})
            print(json_obj)

    elif len(sys.argv) > 10:
        json_obj = json.dumps({'error':1})
        print(json_obj)
    elif len(sys.argv) < 10:
        json_obj = json.dumps({'error':0})
        print(json_obj)
if __name__ == '__main__':
	main()