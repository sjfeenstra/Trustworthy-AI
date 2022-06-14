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

from matplotlib import pyplot as plt

onnx_model = "models/supercombo.onnx"

X_IDXS = np.array([0., 0.1875, 0.75, 1.6875, 3., 4.6875,
                   6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
                   27., 31.6875, 36.75, 42.1875, 48., 54.1875,
                   60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
                   108., 117.1875, 126.75, 136.6875, 147., 157.6875,
                   168.75, 180.1875, 192.])


def onnx2Keras(path):
    onnx_model = onnx.load(path)
    model = onnx_to_keras(onnx_model, ['input_imgs', 'big_input_imgs', 'desire', 'traffic_convention', 'initial_state'],
                          name_policy='renumerate', change_ordering=False)
    model.save("models/keras")


def onnx2Pytorch(path):
    code_gen.gen(path, "pytorchGenerator")
    model = Model()
    torch.save(model, 'models/supercombo.pt')


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def parse_image(frame):
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))
    return parsed


def seperate_points_and_std_values(df):
    points = df.iloc[lambda x: x.index % 2 == 0]
    std = df.iloc[lambda x: x.index % 2 != 0]
    points = pd.concat([points], ignore_index=True)
    std = pd.concat([std], ignore_index=True)

    return points, std


def main():
    # Run first time
    # onnx2Pytorch(onnx_model)

    print("start")
    k_model = tf.keras.models.load_model("models/keras")
    p_model = torch.load('models/supercombo.pt')

    results = []
    counter = 0
    end_counter = 3
    width = 512
    height = 256
    dim = (width, height)
    isRHD = True

    initial_state_data = np.array([0]).astype('float32')
    initial_state_data.resize((1, 512), refcheck=False)

    cap = cv2.VideoCapture('data/roadcamera.mp4')
    parsed_images = []

    cap2 = cv2.VideoCapture('data/widecamera.mp4')
    parsed_images2 = []

    session = onnxruntime.InferenceSession(onnx_model, None)

    input_imgs = session.get_inputs()[0].name
    big_input_imgs = session.get_inputs()[1].name
    desire = session.get_inputs()[2].name
    traffic_convention = session.get_inputs()[3].name
    initial_state = session.get_inputs()[4].name
    output_name = session.get_outputs()[0].name

    while cap.isOpened() and cap2.isOpened():
        counter = counter + 1

        if counter >= end_counter:
            break

        ret, frame = cap.read()
        if ret == False:
            break

        ret2, frame2 = cap2.read()
        if ret2 == False:
            break

        if frame is not None:
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            parsed = parse_image(img_yuv)

        if frame2 is not None:
            img2 = cv2.resize(frame2, dim)
            img_yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV_I420)
            parsed2 = parse_image(img_yuv2)

        if len(parsed_images) >= 2:
            del parsed_images[0]

        if len(parsed_images2) >= 2:
            del parsed_images2[0]

        parsed_images.append(parsed)
        parsed_images2.append(parsed2)

        if len(parsed_images) >= 2:
            # combined_images = np.append(parsed_images[0],parsed_images[1], axis=0)
            # test = np.array([combined_images]).astype('float32')
            parsed_arr = np.array(parsed_images).astype('float32')
            parsed_arr.resize((1, 12, 128, 256))

            parsed_arr2 = np.array(parsed_images2).astype('float32')
            parsed_arr2.resize((1, 12, 128, 256))

            desire_data = np.array([0]).astype('float32')
            desire_data.resize((1, 8), refcheck=False)

            if isRHD:
                traffic_convention_data = np.array([[0, 1]]).astype('float32')
            else:
                traffic_convention_data = np.array([[1, 0]]).astype('float32')

            # onnx runner
            result = session.run([output_name], {input_imgs: parsed_arr,
                                                 big_input_imgs: parsed_arr2,
                                                 desire: desire_data,
                                                 traffic_convention: traffic_convention_data,
                                                 initial_state: initial_state_data
                                                 })

            results.append(np.array(result[0]))

            # Keras runner

            result = k_model.predict(
                [parsed_arr, parsed_arr2, desire_data, traffic_convention_data, initial_state_data])
            results.append(np.array(result))

            # pytorch runner
            with torch.no_grad():
                result = p_model(torch.from_numpy(parsed_arr), torch.from_numpy(parsed_arr2),
                                 torch.from_numpy(desire_data), torch.from_numpy(traffic_convention_data),
                                 torch.from_numpy(initial_state_data))
                results.append(np.array(result))
            combined = np.concatenate(results, axis=0)

    print('end')

    # data = np.load('numpy.npz')
    #
    # inputImgs_data = data['inputImgs']
    # bigInputImgs_data = data['bigInputImgs']
    # desire_data = data['desire']
    # trafficConvention_data = data['trafficConvention']
    # initialState_data = data['initialState']
    # output_data = data['output']

    # input_dataset = tf.data.Dataset.from_tensor_slices(
    #     (inputImgs_data, bigInputImgs_data, desire_data, trafficConvention_data, initialState_data))
    # input_dataset = tf.data.Dataset.from_tensors({"input_imgs": inputImgs_data,
    #                                               "big_input_imgs": bigInputImgs_data,
    #                                               "desire": desire_data,
    #                                               "traffic_convention": trafficConvention_data,
    #                                               "initial_state": initialState_data})
    # output_dataset = tf.data.Dataset.from_tensors({"outputs": output_data})

    # print(input_dataset.element_spec)
    # print(output_dataset.element_spec)

    # result = tf_rep.run({input_imgs: inputImgs_data,
    #                      big_input_imgs: bigInputImgs_data,
    #                      desire: desire_data,
    #                      traffic_convention: trafficConvention_data,
    #                      initial_state: initialState_data
    #                      })
    #
    # res = np.array(result)
    #
    # plan = res[:, :, plan_start_idx:plan_end_idx]
    # lanes = res[:, :, lanes_start_idx:lanes_end_idx]
    # lane_lines_prob = res[:, :, lane_lines_prob_start_idx:lane_lines_prob_end_idx]
    # lane_road = res[:, :, road_start_idx:road_end_idx]
    # # lead = res[:,:,lead_start_idx:lead_end_idx]
    # # lead_prob = res[:,:,lead_prob_start_idx:lead_prob_end_idx]
    # # desire_state = res[:,:,desire_start_idx:desire_end_idx]
    # # meta = res[:,:,meta_start_idx:meta_end_idx]
    # # desire_pred = res[:,:,desire_pred_start_idx:desire_pred_end_idx]
    # # pose = res[:,:,pose_start_idx:pose_end_idx]
    # recurrent_layer = res[:, :, recurent_start_idx:recurent_end_idx]
    # initial_state_data = recurrent_layer[0]
    #
    # lanes_flat = lanes.flatten()
    # df_lanes = pd.DataFrame(lanes_flat)
    #
    # ll_t = df_lanes[0:66]
    # ll_t2 = df_lanes[66:132]
    # points_ll_t, std_ll_t = seperate_points_and_std_values(ll_t)
    # points_ll_t2, std_ll_t2 = seperate_points_and_std_values(ll_t2)
    #
    # l_t = df_lanes[132:198]
    # l_t2 = df_lanes[198:264]
    # points_l_t, std_l_t = seperate_points_and_std_values(l_t)
    # points_l_t2, std_l_t2 = seperate_points_and_std_values(l_t2)
    #
    # r_t = df_lanes[264:330]
    # r_t2 = df_lanes[330:396]
    # points_r_t, std_r_t = seperate_points_and_std_values(r_t)
    # points_r_t2, std_r_t2 = seperate_points_and_std_values(r_t2)
    #
    # rr_t = df_lanes[396:462]
    # rr_t2 = df_lanes[462:528]
    # points_rr_t, std_rr_t = seperate_points_and_std_values(rr_t)
    # points_rr_t2, std_rr_t2 = seperate_points_and_std_values(rr_t2)
    #
    # road_flat = lane_road.flatten()
    # df_road = pd.DataFrame(road_flat)
    #
    # roadr_t = df_road[0:66]
    # roadr_t2 = df_road[66:132]
    # points_road_t, std_ll_t = seperate_points_and_std_values(roadr_t)
    # points_road_t2, std_ll_t2 = seperate_points_and_std_values(roadr_t2)
    #
    # roadl_t = df_road[132:198]
    # roadl_t2 = df_road[198:264]
    # points_roadl_t, std_rl_t = seperate_points_and_std_values(roadl_t)
    # points_roadl_t2, std_rl_t2 = seperate_points_and_std_values(roadl_t2)
    #
    # print("mid")
    # x = np.arange(2, end_counter)
    # combined = np.concatenate(results, axis=0)
    # combined = np.column_stack(combined)
    # number = 1
    # for prob in combined:
    #     if (number % 2) == 0:
    #         y = np.array(sigmoid(prob))
    #         plt.plot(x, y)
    #         if number == 2:
    #             plt.title("left far")
    #         if number == 4:
    #             plt.title("left near")
    #         if number == 6:
    #             plt.title("right near")
    #         if number == 8:
    #             plt.title("right far")
    #         plt.show()
    #     number = number + 1
    # print("end")


if __name__ == "__main__":
    main()
