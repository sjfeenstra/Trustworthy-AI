import onnxruntime
import numpy as np

model = "models/supercombo.onnx"

session = onnxruntime.InferenceSession(model, None)

input_imgs = session.get_inputs()[0].name
big_input_imgs = session.get_inputs()[1].name
desire = session.get_inputs()[2].name
traffic_convention = session.get_inputs()[3].name
initial_state = session.get_inputs()[4].name
output_name = session.get_outputs()[0].name

data = np.load('data/numpy.npz')

inputImgs_data = data['inputImgs']
bigInputImgs_data = data['bigInputImgs']
desire_data = data['desire']
trafficConvention_data = data['trafficConvention']
initialState_data = data['initialState']

result = session.run([output_name], {input_imgs: np.vsplit(inputImgs_data, 198)[0],
                                     big_input_imgs: np.vsplit(bigInputImgs_data, 198)[0],
                                     desire: np.vsplit(desire_data, 198)[0],
                                     traffic_convention: np.vsplit(trafficConvention_data, 198)[0],
                                     initial_state: np.vsplit(initialState_data, 198)[0]
                                     })

res = np.array(result)
