import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import onnxruntime


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


model = 'models/supercombo.onnx'
session = onnxruntime.InferenceSession(model, None)

plan_start_idx = 0
plan_end_idx = 4955

lanes_start_idx = plan_end_idx
lanes_end_idx = lanes_start_idx + 528

lane_lines_prob_start_idx = lanes_end_idx
lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

data = np.load('data/numpy3.npz')
# data2 = np.load('data/numpy2.npz')

inputImgs_data = data['inputImgs'][:10, :]
bigInputImgs_data = data['bigInputImgs'][:10, :]
desire_data = data['desire'][:10, :]
trafficConvention_data = data['trafficConvention'][:10, :]
initialState_data = data['initialState'][:10, :]
output_data = data['output'][:10, :]

# inputImgs_data2 = data2['inputImgs']
# bigInputImgs_data2 = data2['bigInputImgs']
# desire_data2 = data2['desire']
# trafficConvention_data2 = data2['trafficConvention']
# initialState_data2 = data2['initialState']
# output_dat2a = data2['output']

# datasize = data['inputImgs'].shape[0]
datasize = 10

input_imgs = session.get_inputs()[0].name
big_input_imgs = session.get_inputs()[1].name
desire = session.get_inputs()[2].name
traffic_convention = session.get_inputs()[3].name
initial_state = session.get_inputs()[4].name
output_name = session.get_outputs()[0].name

initialState_data = np.array([0]).astype('float32')
initialState_data.resize((1, 512), refcheck=False)

results = None
results2 = None
for x in range(datasize):
    result = session.run([output_name], {input_imgs: np.vsplit(inputImgs_data, datasize)[x],
                                         big_input_imgs: np.vsplit(bigInputImgs_data, datasize)[x],
                                         desire: np.vsplit(desire_data, datasize)[x],
                                         traffic_convention: np.vsplit(trafficConvention_data, datasize)[x],
                                         initial_state: initialState_data
                                         })
    if np.any(results):
        results = np.concatenate((results, result[0]), axis=0)
    else:
        results = result[0]

# for x in range(datasize):
#     result2 = session.run([output_name], {input_imgs: np.vsplit(inputImgs_data2, datasize)[x],
#                                          big_input_imgs: np.vsplit(bigInputImgs_data2, datasize)[
#                                              x],
#                                          desire: np.vsplit(desire_data2, datasize)[x],
#                                          traffic_convention:
#                                              np.vsplit(trafficConvention_data2, datasize)[x],
#                                          initial_state: initialState_data,
#                                          })
#     res2 = np.array(result2)
#     lane_lines_prob2 = res2[:, :, lane_lines_prob_start_idx:lane_lines_prob_end_idx]
#     lane_lines_prob2 = sigmoid(lane_lines_prob2)
#     lane_lines_prob2.resize([1, 8], refcheck=False)
#
#     if not np.any(results2):
#         results2 = lane_lines_prob2
#     else:
#         results2 = np.concatenate((results2, lane_lines_prob2), axis=0)

lane_lines_prob = results[:, lane_lines_prob_start_idx:lane_lines_prob_end_idx]
lane_lines_prob = sigmoid(lane_lines_prob)
lane_lines_prob = lane_lines_prob[:, ::2]

print('test')
