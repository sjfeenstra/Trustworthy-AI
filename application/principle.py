from abc import ABC, abstractmethod
import onnxruntime
import numpy as np
import os


# Strategy interface
class Principle(ABC):
    model = os.getenv('MODEL')
    session = onnxruntime.InferenceSession(model, None)

    data = np.load(os.getenv('DATA'))

    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    datasize = data['inputImgs'].shape[0]

    inputImgs_data = data['inputImgs']
    bigInputImgs_data = data['bigInputImgs']
    desire_data = data['desire']
    trafficConvention_data = data['trafficConvention']
    # initialState_data = data['initialState']
    output_data = data['output']

    initialState_data = np.array([0]).astype('float32')
    initialState_data.resize((1, 512), refcheck=False)

    input_imgs = session.get_inputs()[0].name
    big_input_imgs = session.get_inputs()[1].name
    desire = session.get_inputs()[2].name
    traffic_convention = session.get_inputs()[3].name
    initial_state = session.get_inputs()[4].name
    output_name = session.get_outputs()[0].name

    @abstractmethod
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    @property
    @abstractmethod
    def name(self) -> None:
        pass

    @property
    @abstractmethod
    def toolResults(self) -> None:
        pass

    @property
    @abstractmethod
    def results(self) -> None:
        pass

    @abstractmethod
    def doTests(self) -> None:
        pass

    @abstractmethod
    def addResults(self, data: np.array, type: str) -> None:
        pass
    @abstractmethod
    def addToolResults(self) -> None:
        pass

    @abstractmethod
    def useTools(self) -> None:
        pass

    @abstractmethod
    def runModel(self, inputImgs=inputImgs_data, bigInputImgs=bigInputImgs_data, desire=desire_data,
                 trafficConvention=trafficConvention_data, initialState=initialState_data):
        results = None
        for x in range(self.datasize):
            result = self.session.run([self.output_name], {self.input_imgs: np.vsplit(inputImgs, self.datasize)[x],
                                                           self.big_input_imgs: np.vsplit(bigInputImgs, self.datasize)[
                                                               x],
                                                           self.desire: np.vsplit(desire, self.datasize)[x],
                                                           self.traffic_convention:
                                                               np.vsplit(trafficConvention, self.datasize)[x],
                                                           self.initial_state: initialState
                                                           })
            res = np.array(result)
            lane_lines_prob = res[:, :, self.lane_lines_prob_start_idx:self.lane_lines_prob_end_idx]
            lane_lines_prob = self.sigmoid(lane_lines_prob)
            lane_lines_prob.resize([1, 8], refcheck=False)

            if not np.any(results):
                results = lane_lines_prob
            else:
                results = np.concatenate((results, lane_lines_prob), axis=0)
        return results

    @abstractmethod
    def calculateAccuracy(self, result):
        probabilityPoint = 0.7
        lane_lines_prob = result[:, ::2]
        probability = (lane_lines_prob >= probabilityPoint).astype(int)
        accuracy = np.sum(probability == self.output_data) / (
                len(self.output_data) * len(self.output_data[0]))
        return lane_lines_prob, accuracy
