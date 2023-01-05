from trustworthyAI import TrustworthyAI
from principle import Principle
import numpy as np


# Concrete strategies
class Transparency(Principle):

    def __init__(self):
        self._results = []

    def addResults(self, data: np.array, type2: str, type: str) -> None:
        pass

    def sigmoid(self, input):
        pass

    def runModel(self, inputImgs=None, bigInputImgs=None, desire=None,
                 trafficConvention=None, initialState=None):
        pass

    def calculateAccuracy(self, result):
        pass

    @property
    def results(self) -> None:
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def doTests(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "Transparency"
