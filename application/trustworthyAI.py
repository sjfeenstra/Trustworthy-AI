import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from principle import Principle

lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
typeList = ["Traffic Convention", "Desire"]


# Context class
class TrustworthyAI:
    def __init__(self, p: Principle) -> None:
        self._principle = p

    @property
    def principle(self) -> Principle:
        return self._principle

    @principle.setter
    def principle(self, p: Principle) -> None:
        self._principle = p

    def do_tests(self):
        self._principle.doTests()

    def update_data(self, principle):
        a = np.arange(0, self.principle.datasize)
        X = [float(x) for x in a]
        for type in typeList:
            filtered = list(filter(lambda x: type in x['type'] or x['type'] == "Default", principle.results))
            filtered3 = []
            for i in filtered:
                if not any((d['type'] == i["type"] and d['type2'] == i["type2"]) for d in filtered3):
                    filtered3.append(i)
            for resul in filtered3:
                dpg.add_text(str.strip(
                    f'{resul["type2"]} {resul["type"]} :'),
                    parent=f'{principle.name}{type} type group')
                dpg.add_text(str.strip(
                    f'{resul["accuracy"]:.5f}%'),
                    parent=f'{principle.name}{type} accuracy group')
            for line in lineList:
                filtered2 = list(filter(lambda x: x['line'] == line, filtered))
                for item in filtered2:
                    dpg.add_line_series(X, item['data'],
                                        label=item["type2"] + " " + item[
                                            "type"],
                                        use_internal_label=False, parent=f'{principle.name}{type}{line} yaxis')
                    dpg.fit_axis_data(f'{principle.name}{type}{line} xaxis')
