from principle import Principle
import numpy as np
import os


class TechnicalRobustnessAndSafety(Principle):
    data = np.load(os.getenv('DATA2'))

    inputImgs_data = data['inputImgs']
    bigInputImgs_data = data['bigInputImgs']
    desire_data = data['desire']
    trafficConvention_data = data['trafficConvention']
    initialState_data = data['initialState']
    output_data = data['output']

    def __init__(self):
        self._results = []
        self._name = "Technical Robustness and Safety"

    @property
    def name(self) -> str:
        return self._name

    @property
    def results(self) -> list[dict]:
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def getLaneLineProb(self, result):
        return super().getLaneLineProb(result)

    def sigmoid(self, input) -> np.array:
        return super().sigmoid(input)

    def runModel(self, inputImgs=None, bigInputImgs=None, desire=None,
                 trafficConvention=None, initialState=None):
        return super().runModel(inputImgs, bigInputImgs, desire, trafficConvention, initialState)

    def calculateAccuracy(self, result, outputData):
        return super().calculateAccuracy(result, outputData)

    def addResults(self, data: np.array, type2: str, type: str) -> None:
        lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
        lane_line_prob = self.getLaneLineProb(data)
        if data[0].shape != self.output_data[0].shape:
            acc = self.calculateAccuracy(lane_line_prob, super().output_data)
        else:
            acc = self.calculateAccuracy(lane_line_prob, self.output_data)
        for b in range(4):
            self._results.append({
                "type": type,
                "type2": type2,
                "line": lineList[b],
                "data": np.column_stack(lane_line_prob)[b].tolist(),
                "accuracy": acc
            })

    def doTests(self):
        trafficConventionShape = super().trafficConvention_data.shape
        trafficConvention_data_inverted = super().trafficConvention_data[::, ::-1]
        trafficConvention_data_ones = np.ones(trafficConventionShape).astype(np.float32)
        trafficConvention_data_zeros = np.zeros(trafficConventionShape).astype(np.float32)

        desireShape = super().desire_data.shape
        desire_data_ones = np.ones(desireShape).astype(np.float32)
        desire_data_zeros = np.zeros(desireShape).astype(np.float32)
        desire_data_0 = desire_data_zeros.copy()
        desire_data_0[:, 0] = 1
        desire_data_1 = desire_data_zeros.copy()
        desire_data_1[:, 1] = 1.0
        desire_data_2 = desire_data_zeros.copy()
        desire_data_2[:, 2] = 1.0
        desire_data_3 = desire_data_zeros.copy()
        desire_data_3[:, 3] = 1.0
        desire_data_4 = desire_data_zeros.copy()
        desire_data_4[:, 4] = 1.0
        desire_data_5 = desire_data_zeros.copy()
        desire_data_5[:, 5] = 1.0
        desire_data_6 = desire_data_zeros.copy()
        desire_data_6[:, 6] = 1.0
        desire_data_7 = desire_data_zeros.copy()
        desire_data_7[:, 7] = 1.0

        self.addResults(super().runModel(), "", "Default", )
        self.addResults(super().runModel(inputImgs=self.inputImgs_data, bigInputImgs=self.bigInputImgs_data,
                                         desire=self.desire_data, trafficConvention=self.trafficConvention_data),
                        "Geen Close Left", "CameraFrames")
        self.addResults(super().runModel(trafficConvention=trafficConvention_data_inverted), "Left hand drive",
                        "Traffic Convention")
        self.addResults(super().runModel(trafficConvention=trafficConvention_data_ones), "Ones", "Traffic Convention")
        self.addResults(super().runModel(trafficConvention=trafficConvention_data_zeros), "Zeros",
                        "Traffic Convention")

        self.addResults(super().runModel(desire=desire_data_zeros), "Zeros", "Desire")
        self.addResults(super().runModel(desire=desire_data_ones), "Ones", "Desire")
        self.addResults(super().runModel(desire=desire_data_0),
                        "None", "Desire")
        self.addResults(super().runModel(desire=desire_data_1),
                        "Turn left", "Desire")
        self.addResults(super().runModel(desire=desire_data_2),
                        "Turn right", "Desire")
        self.addResults(super().runModel(desire=desire_data_3),
                        "Lane change left", "Desire")
        self.addResults(super().runModel(desire=desire_data_4),
                        "Lane change right", "Desire")
        self.addResults(super().runModel(desire=desire_data_5),
                        "Keep left", "Desire")
        self.addResults(super().runModel(desire=desire_data_6),
                        "Keep right", "Desire")
        self.addResults(super().runModel(desire=desire_data_7),
                        "Null", "Desire")
