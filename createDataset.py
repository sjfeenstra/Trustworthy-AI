import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd


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


def main():
    model = "models/supercombo.onnx"

    cap = cv2.VideoCapture('data/roadcamera.mp4')
    parsed_images = []

    cap2 = cv2.VideoCapture('data/widecamera.mp4')
    parsed_images2 = []

    results = []
    counter = 0
    start_counter = 0
    end_counter = 500
    width = 512
    height = 256
    dim = (width, height)
    isRHD = True

    inputImgsNPZ = None
    bigInputImgsNPZ = None
    desireNPZ = None
    trafficConventionNPZ = None
    initialStateNPZ = None
    outputNPZ = None

    initial_state_data = np.array([0], dtype=np.float32)
    initial_state_data.resize((1, 512), refcheck=False)

    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264

    lead_start_idx = road_end_idx
    lead_end_idx = lead_start_idx + 102

    lead_prob_start_idx = lead_end_idx
    lead_prob_end_idx = lead_prob_start_idx + 3

    stop_start_idx = lead_prob_end_idx
    stop_end_idx = stop_start_idx + 52

    meta_start_idx = stop_end_idx
    meta_end_idx = meta_start_idx + 88

    pose_start_idx = meta_end_idx
    pose_end_idx = pose_start_idx + 12

    recurent_start_idx = pose_end_idx
    recurent_end_idx = recurent_start_idx + 512

    session = onnxruntime.InferenceSession(model, None)
    while cap.isOpened() and cap2.isOpened():
        counter = counter + 1
        print(counter)

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
            parsed_arr = np.array(parsed_images, dtype=np.float32)
            parsed_arr.resize((1, 12, 128, 256))

            parsed_arr2 = np.array(parsed_images2, dtype=np.float32)
            parsed_arr2.resize((1, 12, 128, 256))

            input_imgs = session.get_inputs()[0].name
            big_input_imgs = session.get_inputs()[1].name
            desire = session.get_inputs()[2].name
            traffic_convention = session.get_inputs()[3].name
            initial_state = session.get_inputs()[4].name
            output_name = session.get_outputs()[0].name

            desire_data = np.array([0], dtype=np.float32)
            desire_data.resize((1, 8), refcheck=False)

            if isRHD:
                traffic_convention_data = np.array([[0, 1]], dtype=np.float32)
            else:
                traffic_convention_data = np.array([[1, 0]], dtype=np.float32)

            result = session.run([output_name], {input_imgs: parsed_arr,
                                                 big_input_imgs: parsed_arr2,
                                                 desire: desire_data,
                                                 traffic_convention: traffic_convention_data,
                                                 initial_state: initial_state_data
                                                 })

            res = np.array(result)

            if not np.any(inputImgsNPZ):
                inputImgsNPZ = parsed_arr
                bigInputImgsNPZ = parsed_arr2
                desireNPZ = desire_data
                trafficConventionNPZ = traffic_convention_data
                initialStateNPZ = initial_state_data
                outputNPZ = np.array([[1, 1, 1, 0]], dtype=np.float32)
            else:
                inputImgsNPZ = np.concatenate((inputImgsNPZ, parsed_arr), axis=0)
                bigInputImgsNPZ = np.concatenate((bigInputImgsNPZ, parsed_arr2), axis=0)
                desireNPZ = np.concatenate((desireNPZ, desire_data), axis=0)
                trafficConventionNPZ = np.concatenate((trafficConventionNPZ, traffic_convention_data), axis=0)
                initialStateNPZ = np.concatenate((initialStateNPZ, initial_state_data), axis=0)
                outputNPZ = np.concatenate((outputNPZ, [[1, 1, 1, 0]]), axis=0)

            # recurrent_layer = res[:, :, recurent_start_idx:recurent_end_idx]
            # initial_state_data = recurrent_layer[0]
    np.savez_compressed('data/np500', inputImgs=inputImgsNPZ, bigInputImgs=bigInputImgsNPZ,
                        desire=desireNPZ,
                        trafficConvention=trafficConventionNPZ, initialState=initialStateNPZ, output=outputNPZ)


if __name__ == "__main__":
    main()
