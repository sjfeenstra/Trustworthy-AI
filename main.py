from abc import ABC, abstractmethod
from typing import Dict, Any

import dearpygui.dearpygui as dpg
import numpy as np
import tensorflow as tf


# Strategy interface
class Principle(ABC):
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
    def addResults(self, data: np.array, type: str) -> None:
        pass


# Concrete strategies
class Transparency(Principle):
    def __init__(self):
        self._results = None

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


class TechnicalRobustnessAndSafety(Principle):
    def __init__(self):
        self._results = []

    @property
    def name(self) -> str:
        return "Technical Robustness and Safety"

    @property
    def results(self) -> list[dict]:
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def addResults(self, data: np.array, type: str) -> None:
        lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
        res, acc = trustworthyAI.calculateAccuracy(data)
        for b in range(4):
            self._results.append({
                "type": type,
                "line": lineList[b],
                "data": np.column_stack(res)[b].tolist(),
                "accuracy": acc
            })

    def doTests(self) -> None:
        trafficConventionShape = trustworthyAI.trafficConvention_data.shape
        trafficConvention_data_inverted = trustworthyAI.trafficConvention_data[::, ::-1]
        trafficConvention_data_ones = np.ones(trafficConventionShape)
        trafficConvention_data_zeros = np.zeros(trafficConventionShape)

        desireShape = trustworthyAI.desire_data.shape
        desire_data_ones = np.ones(desireShape)
        desire_data_zeros = np.zeros(desireShape)
        desire_data_0 = desire_data_zeros.copy()
        desire_data_0[:, 0] = 1
        desire_data_1 = desire_data_zeros.copy()
        desire_data_1[:, 1] = 1
        desire_data_2 = desire_data_zeros.copy()
        desire_data_2[:, 2] = 1
        desire_data_3 = desire_data_zeros.copy()
        desire_data_3[:, 3] = 1
        desire_data_4 = desire_data_zeros.copy()
        desire_data_4[:, 4] = 1
        desire_data_5 = desire_data_zeros.copy()
        desire_data_5[:, 5] = 1
        desire_data_6 = desire_data_zeros.copy()
        desire_data_6[:, 6] = 1
        desire_data_7 = desire_data_zeros.copy()
        desire_data_7[:, 7] = 1

        self.addResults(trustworthyAI.runModel(),
                        "Default")
        self.addResults(trustworthyAI.runModel(trafficConvention=trafficConvention_data_inverted),
                        "Inverted Traffic Convention")
        self.addResults(trustworthyAI.runModel(trafficConvention=trafficConvention_data_ones),
                        "Ones Traffic Convention")
        self.addResults(trustworthyAI.runModel(trafficConvention=trafficConvention_data_zeros),
                        "Zeros Traffic Convention")

        self.addResults(trustworthyAI.runModel(desire=desire_data_zeros),
                        "Zeros Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_ones),
                        "Ones Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_0),
                        "index 0 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_1),
                        "index 1 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_2),
                        "index 2 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_3),
                        "index 3 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_4),
                        "index 4 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_5),
                        "index 5 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_6),
                        "index 6 Desire")
        self.addResults(trustworthyAI.runModel(desire=desire_data_7),
                        "index 7 Desire")


# Context class
class TrustworthyAI:
    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    data = np.load('data/numpy.npz')

    datasize = data['inputImgs'].shape[0]

    inputImgs_data = data['inputImgs']
    bigInputImgs_data = data['bigInputImgs']
    desire_data = data['desire']
    trafficConvention_data = data['trafficConvention']
    initialState_data = data['initialState']
    output_data = data['output']

    def __init__(self, p: Principle) -> None:
        self._principle = p

    @property
    def principle(self) -> Principle:
        return self._principle

    @principle.setter
    def principle(self, p: Principle) -> None:
        self._principle = p

    def setPrinciple(self, p: Principle):
        self._principle = p

    def runModel(self, inputImgs=inputImgs_data, bigInputImgs=bigInputImgs_data, desire=desire_data,
                 trafficConvention=trafficConvention_data, initialState=initialState_data):
        m = tf.keras.models.load_model("models/keras", compile=False)
        r = m.predict(tf.data.Dataset.from_tensors({"input_imgs": inputImgs,
                                                    "big_input_imgs": bigInputImgs,
                                                    "desire": desire,
                                                    "traffic_convention": trafficConvention,
                                                    "initial_state": initialState}))
        return r

    def calculateAccuracy(self, result):
        probabilityPoint = 0.7
        lane_lines_prob = sigmoid(
            result[:, trustworthyAI.lane_lines_prob_start_idx + 1:trustworthyAI.lane_lines_prob_end_idx:2])
        probability = (lane_lines_prob >= probabilityPoint).astype(int)
        accuracy = np.sum(probability == trustworthyAI.output_data) / (
                len(trustworthyAI.output_data) * len(trustworthyAI.output_data[0]))
        return lane_lines_prob, accuracy


if __name__ == "__main__":
    dpg.create_context()

    # Create all available principles
    transparencyPrinciple = Transparency()
    TRaSPrinciple = TechnicalRobustnessAndSafety()

    principles = [transparencyPrinciple, TRaSPrinciple]

    # Create context class
    trustworthyAI = TrustworthyAI(transparencyPrinciple)


    def sigmoid(input):
        return 1 / (1 + np.exp(-input))


    def tab_callback(sender, app_data, user_data):
        trustworthyAI.setPrinciple(dpg.get_item_user_data(app_data))


    a = np.arange(0, trustworthyAI.datasize)
    X = [float(x) for x in a]

    lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
    typeList = ["Traffic Convention", "Desire"]

    width, height, channels, data = dpg.load_image("data/openpilot.png")
    width2, height2, channels2, data2 = dpg.load_image("data/roadlines.png")

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=500, height=200, default_value=data, tag="texture_tag")
        dpg.add_static_texture(width=500, height=200, default_value=data2, tag="texture_tag2")

    with dpg.window(tag="Primary Window", height=1200, width=1200) as window:
        with dpg.tab_bar(tag="test_tab_bar", callback=tab_callback) as tb:
            with dpg.tab(label="Overview"):
                dpg.add_text("Overview")
                dpg.add_image("texture_tag")
                dpg.add_image("texture_tag2")
            for principle in principles:
                with dpg.tab(label=principle.name, tag=principle.name):
                    dpg.set_item_user_data(principle.name, principle)
                    principle.doTests()
                    resu = principle.results
                    if resu != None:
                        with dpg.tab_bar(tag="test_tab_bar2") as tb2:
                            with dpg.tab(label="Tools"):
                                dpg.add_text("Tools")
                                dpg.add_text(f'AdvBox : resultaat')
                                dpg.add_text(f'Adversial Robustness 360 Toolbox : resultaat')
                                dpg.add_text(f'Cleverhans : resultaat')
                                dpg.add_text(f'Foolbox : resultaat')
                            for type in typeList:
                                filtered = list(
                                    filter(lambda x: type in x['type'] or x['type'] == "Default", resu))
                                with dpg.tab(label=type):
                                    if len(filtered) != 0:
                                        with dpg.group(horizontal=True):
                                            with dpg.subplots(2, 2, label=f'{principle.name}', height=-1, width=-200,
                                                              row_ratios=[5.0, 5.0],
                                                              column_ratios=[5.0, 5.0], column_major=True) as subplot_id:
                                                dpg.add_plot_legend()
                                                for line in lineList:
                                                    filtered2 = list(filter(lambda x: x['line'] == line, filtered))
                                                    if len(filtered2) != 0:
                                                        with dpg.plot(tag=f'{type}{line}', anti_aliased=True, ):
                                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frames")
                                                            dpg.set_axis_limits(dpg.last_item(), -5, 200)
                                                            dpg.set_item_label(f'{type}{line}', line)
                                                            with dpg.plot_axis(dpg.mvYAxis, label="Probability"):
                                                                dpg.set_axis_limits(dpg.last_item(), -0.05, 1.1)
                                                                for item in filtered2:
                                                                    dpg.add_line_series(X, item["data"], label=item["type"].replace(type,""),
                                                                                        use_internal_label=False)
                                            with dpg.group() as group:
                                                dpg.add_text(f"Accuracy {type}")
                                                filtered3 = []
                                                for i in filtered:
                                                    if not any(d['type'] == i["type"] for d in filtered3):
                                                        filtered3.append(i)
                                                for resul in filtered3:
                                                    dpg.add_text(f'{resul["type"].replace(type,"")} : {resul["accuracy"]:.5f}%')
                    else:
                        dpg.add_text("No results")

    dpg.create_viewport(title='Trustworthy AI')
    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
