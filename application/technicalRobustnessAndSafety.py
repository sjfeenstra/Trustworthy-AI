from principle import Principle
import numpy as np


class TechnicalRobustnessAndSafety(Principle):
    @property
    def toolResults(self) -> list[dict]:
        return self._toolResults

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def runModel(self, inputImgs=None, bigInputImgs=None, desire=None,
                 trafficConvention=None, initialState=None):
        pass

    def calculateAccuracy(self, result):
        pass

    def useTools(self):
        self.addToolResults()

    def __init__(self):
        self._results = []
        self._toolResults = []

    @property
    def name(self) -> str:
        return "Technical Robustness and Safety"

    @property
    def results(self) -> list[dict]:
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @toolResults.setter
    def toolResults(self, value):
        self._results = value

    def addResults(self, data: np.array, type: str) -> None:
        lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
        res, acc = super().calculateAccuracy(data)
        for b in range(4):
            self._results.append({
                "type": type,
                "line": lineList[b],
                "data": np.column_stack(res)[b].tolist(),
                "accuracy": acc
            })

    def addToolResults(self) -> None:
        tools = ["AdvBox", "Adversial Robustness 360 Toolbox", "Cleverhans", "Foolbox"]
        for b in range(len(tools)):
            self._toolResults.append({
                "tool": tools[b],
                "result": "Expected 1 input variable but got 5",
            })

    def doTests(self) -> None:
        # trafficConventionShape = super().trafficConvention_data.shape
        # trafficConvention_data_inverted = super().trafficConvention_data[::, ::-1]
        # trafficConvention_data_ones = np.ones(trafficConventionShape).astype(np.float32)
        # trafficConvention_data_zeros = np.zeros(trafficConventionShape).astype(np.float32)
        #
        # desireShape = super().desire_data.shape
        # desire_data_ones = np.ones(desireShape).astype(np.float32)
        # desire_data_zeros = np.zeros(desireShape).astype(np.float32)
        # desire_data_0 = desire_data_zeros.copy()
        # desire_data_0[:, 0] = 1
        # desire_data_1 = desire_data_zeros.copy()
        # desire_data_1[:, 1] = 1.0
        # desire_data_2 = desire_data_zeros.copy()
        # desire_data_2[:, 2] = 1.0
        # desire_data_3 = desire_data_zeros.copy()
        # desire_data_3[:, 3] = 1.0
        # desire_data_4 = desire_data_zeros.copy()
        # desire_data_4[:, 4] = 1.0
        # desire_data_5 = desire_data_zeros.copy()
        # desire_data_5[:, 5] = 1.0
        # desire_data_6 = desire_data_zeros.copy()
        # desire_data_6[:, 6] = 1.0
        # desire_data_7 = desire_data_zeros.copy()
        # desire_data_7[:, 7] = 1.0

        self.addResults(super().runModel(),
                        "Default")
        # self.addResults(super().runModel(trafficConvention=trafficConvention_data_inverted),
        #                 "Inverted Traffic Convention")
        # self.addResults(super().runModel(trafficConvention=trafficConvention_data_ones),
        #                 "Ones Traffic Convention")
        # self.addResults(super().runModel(trafficConvention=trafficConvention_data_zeros),
        #                 "Zeros Traffic Convention")
        #
        # self.addResults(super().runModel(desire=desire_data_zeros),
        #                 "Zeros Desire")
        # self.addResults(super().runModel(desire=desire_data_ones),
        #                 "Ones Desire")
        # self.addResults(super().runModel(desire=desire_data_0),
        #                 "index 0 Desire")
        # self.addResults(super().runModel(desire=desire_data_1),
        #                 "index 1 Desire")
        # self.addResults(super().runModel(desire=desire_data_2),
        #                 "index 2 Desire")
        # self.addResults(super().runModel(desire=desire_data_3),
        #                 "index 3 Desire")
        # self.addResults(super().runModel(desire=desire_data_4),
        #                 "index 4 Desire")
        # self.addResults(super().runModel(desire=desire_data_5),
        #                 "index 5 Desire")
        # self.addResults(super().runModel(desire=desire_data_6),
        #                 "index 6 Desire")
        # self.addResults(super().runModel(desire=desire_data_7),
        #                 "index 7 Desire")
