import multiprocessing
import os
import os.path as osp
import uuid

import numpy as np
import random

from math import sqrt
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.video import write_segment_to_video, upload_to_gcs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def synthetic_pref(_reward_list, left_reward, right_reward):
    min_reward = np.percentile(_reward_list,10)
    max_reward = np.percentile(_reward_list,90)
    normalise_left = ( ( left_reward - min_reward ) / ( max_reward - min_reward ) )
    normalise_right = ( ( right_reward - min_reward ) / ( max_reward - min_reward ) )

    if normalise_left > 1.0: 
        normalise_left = 1.0
    if normalise_right > 1.0:
        normalise_right = 1.0
    if left_reward > right_reward :
        final_label = 0.5 + normalise_left * 0.5
    elif left_reward < right_reward:
        final_label = 0.5 - normalise_right * 0.5
    else:
        final_label = 0.5
    
    return final_label

def process_reward_file(path):
    f = open(path ,"r")
    line = f.readline()
    f.close()
    return eval(line)

def output_file(output_path, data_list,append=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "w+"
    if append:
        mode = "a+"
    f = open(output_path,mode)
    f.write("{}\n".format(data_list))
    f.close()

def predict_pref(_predictor, left_reward, right_reward):
    pref = None
    val = _predictor.predict( [ [left_reward, right_reward] ])
    pref = max(0,val[0])
    pref = min(1,pref)
    return pref

def process_predict_file(input_path,output_path,predictor_choice=0):
    f = open(input_path, "r")
    lines = f.readlines()
    f.close()
    _X = []
    _y = []
    result = []
    if predictor_choice == 1:
        predictor = SVR(kernel='rbf')
    else:
        predictor = LinearRegression()
    for line in lines:
        record = eval(line)
        left_reward = record[0]
        right_reward = record[1]
        ground_truth = record[2]
        mode = record[3]
        if mode == 'H':
            _X.append([left_reward,right_reward])
            _y.append(ground_truth)
            result.append([left_reward,right_reward,ground_truth,'H'])
        else:
            left_reward = record[0]
            right_reward = record[1]
            predict_preference = record[2]
            ground_truth=record[3]
            mse = record[5]
            result.append([left_reward, right_reward,predict_preference,ground_truth, mse,'A'])

    predictor.fit(_X, _y)
    for rec in result:
        if rec[3] == 'H':
            left_reward=rec[0]
            right_reward=rec[1]
            ground_truth=rec[2]
            predict_preference = predict_pref(predictor,rec[0],rec[1])
            mse = (predict_preference - ground_truth) ** 2
            output_file(output_path, [left_reward, right_reward,predict_preference,ground_truth, mse],True)
        else:
            left_reward=rec[0]
            right_reward=rec[1]
            predict_preference = rec[2]
            ground_truth = rec[3]
            mse = rec[4]
            output_file(output_path, [left_reward, right_reward,predict_preference,ground_truth, mse],True)

    
root_folder = "/home/kcwong3/rl-teacher/log/"
scenario = input("Which scenario? ")
run_id = input("Run ID: ")
predictor=input("Which Predictor? (0-Linear Regression, 1-SVR-rbf)")
predict_file_path = "{}/{}/{}-{}.txt".format(root_folder, scenario, run_id, "pref")
output_file_path = "{}/{}/{}-{}.txt".format(root_folder, scenario, run_id, "predict")

process_predict_file(predict_file_path, output_file_path,predictor)