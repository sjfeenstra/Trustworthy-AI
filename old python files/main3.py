import os

import cv2
import onnx
from onnx2keras import onnx_to_keras
from onnx_pytorch import code_gen

from pytorchGenerator.model import Model
import numpy as np
import onnxruntime
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

import alibi_detect as ad

from art.attacks.evasion import FastGradientMethod, FeatureAdversariesNumpy
from art.estimators.classification import PyTorchClassifier, KerasClassifier
from art.utils import load_mnist

from matplotlib import pyplot as plt

onnx_model = "models/supercombo.onnx"


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def main():
    # Run first time
    # onnx2Pytorch(onnx_model)

    print("start")

    datasize = 198

    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    data = np.load('../data/numpy.npz')

    inputImgs_data = data['inputImgs']
    bigInputImgs_data = data['bigInputImgs']
    desire_data = data['desire']
    trafficConvention_data = data['trafficConvention']
    initialState_data = data['initialState']
    output_data = data['output']

    input_dataset = tf.data.Dataset.from_tensors({"input_imgs": inputImgs_data,
                                                  "big_input_imgs": bigInputImgs_data,
                                                  "desire": desire_data,
                                                  "traffic_convention": trafficConvention_data,
                                                  "initial_state": initialState_data})
    output_dataset = tf.data.Dataset.from_tensors({"outputs": output_data})

    # Keras runner
    tf.compat.v1.disable_eager_execution()
    k_model = tf.keras.models.load_model("models/keras")

    result = k_model.predict(input_dataset)


    # pytorch model
    # p_model = torch.load('models/supercombo.pt')
    # p_model.eval()
    #
    # result = p_model(torch.from_numpy(inputImgs_data), torch.from_numpy(bigInputImgs_data),
    #                  torch.from_numpy(desire_data), torch.from_numpy(trafficConvention_data),
    #                  torch.from_numpy(initialState_data))
    # result = result.detach().numpy()
    lane_lines_prob = sigmoid(result[:, lane_lines_prob_start_idx + 1:lane_lines_prob_end_idx:2])

    probability = (lane_lines_prob >= 0.7).astype(int)
    accuracy = np.sum(probability == output_data) / (len(output_data) * len(output_data[0]))
    print("Accuracy on examples: {}%".format(accuracy * 100))

    print("mid")
    x = np.arange(0, datasize)
    stacked = np.column_stack(lane_lines_prob)
    number = 1
    for prob in stacked:
        y = np.array(prob)
        plt.plot(x, y)
        if number == 1:
            plt.title("left far")
        if number == 2:
            plt.title("left near")
        if number == 3:
            plt.title("right near")
        if number == 4:
            plt.title("right far")
        plt.show()
        number = number + 1
    print("end")


if __name__ == "__main__":
    main()
