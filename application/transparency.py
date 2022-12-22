from trustworthyAI import TrustworthyAI
from principle import Principle
import numpy as np


# Concrete strategies
class Transparency(Principle):
    @property
    def toolResults(self) -> None:
        return self._toolResults

    def addToolResults(self) -> None:
        pass

    def useTools(self) -> None:
        pass

    def sigmoid(self, input):
        pass

    def runModel(self, inputImgs=None, bigInputImgs=None, desire=None,
                 trafficConvention=None, initialState=None):
        pass

    def calculateAccuracy(self, result):
        pass

    def __init__(self):
        self._results = None
        self._toolResults = None

    def addResults(self, data: np.array, type: str) -> None:
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
