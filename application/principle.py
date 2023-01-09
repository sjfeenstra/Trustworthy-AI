from abc import ABC, abstractmethod
import onnxruntime
import numpy as np
import os


# Strategy interface
class Principle(ABC):
    data = np.load(os.getenv('DATA'))
    data2 = np.load(os.getenv('DATA2'))

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
    initialState_data = data['initialState']
    output_data = data['output']

    inputImgs_data2 = data2['inputImgs']
    bigInputImgs_data2 = data2['bigInputImgs']
    desire_data2 = data2['desire']
    trafficConvention_data2 = data2['trafficConvention']
    initialState_data2 = data2['initialState']
    output_data2 = data2['output']

    # initialState_data = np.array([0]).astype('float32')
    # initialState_data.resize((1, 512), refcheck=False)

    @abstractmethod
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    @property
    @abstractmethod
    def name(self) -> None:
        pass

    @property
    @abstractmethod
    def results(self) -> None:
        pass

    @abstractmethod
    def doTests(self) -> None:
        pass

    @abstractmethod
    def addResults(self, data: np.array, typ2: str, type: str) -> None:
        pass

    @abstractmethod
    def runModel(self, inputImgs=inputImgs_data, bigInputImgs=bigInputImgs_data, desire=desire_data,
                 trafficConvention=trafficConvention_data, initialState=initialState_data):

        model = os.getenv('MODEL')
        session = onnxruntime.InferenceSession(model, None)
        results = None

        for x in range(self.datasize):
            result = session.run([session.get_outputs()[0].name], {
                session.get_inputs()[0].name: np.vsplit(inputImgs, self.datasize)[x],
                session.get_inputs()[1].name: np.vsplit(bigInputImgs, self.datasize)[x],
                session.get_inputs()[2].name: np.vsplit(desire, self.datasize)[x],
                session.get_inputs()[3].name: np.vsplit(trafficConvention, self.datasize)[x],
                session.get_inputs()[4].name: np.vsplit(initialState, self.datasize)[x],
            })

            if np.any(results):
                results = np.concatenate((results, result[0]), axis=0)
            else:
                results = result[0]
        return results

    @abstractmethod
    def getLaneLineProb(self, result):
        lane_lines_prob = result[:, self.lane_lines_prob_start_idx:self.lane_lines_prob_end_idx:2]
        lane_lines_prob = self.sigmoid(lane_lines_prob)
        return lane_lines_prob

    @abstractmethod
    def calculateAccuracy(self, lane_lines_prob):
        probabilityPoint = 0.7
        probability = (lane_lines_prob >= probabilityPoint).astype(int)
        accuracy = np.sum(probability == self.output_data) / \
                   (len(self.output_data) * len(self.output_data[0]))
        return accuracy
