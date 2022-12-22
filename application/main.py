import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm

from transparency import Transparency
from technicalRobustnessAndSafety import TechnicalRobustnessAndSafety
from trustworthyAI import TrustworthyAI

if __name__ == "__main__":
    dpg.create_context()

    # Create all available principles
    TRaSPrinciple = TechnicalRobustnessAndSafety()
    transparencyPrinciple = Transparency()

    principles = [TRaSPrinciple, transparencyPrinciple]

    # Create context class
    trustworthyAI = TrustworthyAI(TRaSPrinciple)

    cap = cv2.VideoCapture('data/roadcamera2.mp4')

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []
    while cap.isOpened():

        ret, frame = cap.read()
        if ret == False:
            break
        if len(frames) == 1000:
            break
        frames.append(frame)


    def getTextureData(frames, framenumber):
        data = np.flip(frames[framenumber], 2)
        data = data.ravel()
        data = np.asfarray(data, dtype='f')
        texture_data = np.true_divide(data, 255.0)
        return texture_data


    def tab_callback(sender, app_data, user_data):
        trustworthyAI.setPrinciple(dpg.get_item_user_data(app_data))


    def plot_line(sender, app_data, user_data):
        aliases = dpg.get_aliases()
        filteredAliases = filter(lambda x: "yaxis" in x, aliases)
        for alias in filteredAliases:
            dpg.set_value(alias, [[app_data, app_data], [0, 1.1]])
        for type in typeList:
            dpg.set_value(f'{type} slider', app_data)
        dpg.set_value("texture_tag", getTextureData(frames, app_data))


    a = np.arange(0, trustworthyAI.principle.datasize)
    X = [float(x) for x in a]

    lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
    typeList = ["Traffic Convention", "Desire"]

    width, height, channels, data = dpg.load_image('data/roadlines.png')

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(frame_width, frame_height, getTextureData(frames, 0), format=dpg.mvFormat_Float_rgb,
                            tag="texture_tag")
        dpg.add_static_texture(width, height, data, tag="image_id")

    with dpg.window(tag="Primary Window", height=1200, width=1200) as window:
        with dpg.tab_bar(tag="test_tab_bar", callback=tab_callback) as tb:
            with dpg.tab(label="Overview"):
                with dpg.tab_bar(tag="test_tab_bar2") as tb2:
                    with dpg.tab(label="Overview page"):
                        with dpg.drawlist(width=600, height=400):
                            dpg.draw_image("image_id", (0, 0), (600, 400), uv_min=(0, 0), uv_max=(1, 1))
                    with dpg.tab(label="Betrouwbare KI"):
                        trustworthyAI.showTrustworthyAI()
                    with dpg.tab(label="Openpilot model", tag="model_tab"):
                        trustworthyAI.showModel()
            for principle in principles:
                with dpg.tab(label=principle.name, tag=principle.name):
                    dpg.set_item_user_data(principle.name, principle)
                    principle.doTests()
                    resu = principle.results
                    if resu != None:
                        with dpg.tab_bar(tag="test_tab_bar3") as tb3:
                            for type in typeList:
                                filtered = list(
                                    filter(lambda x: type in x['type'] or x['type'] == "Default", resu))
                                with dpg.tab(label=type):
                                    if len(filtered) != 0:
                                        with dpg.group(horizontal=True):
                                            with dpg.subplots(2, 2, label=f'{principle.name}', height=-1, width=-400,
                                                              row_ratios=[5.0, 5.0],
                                                              column_ratios=[5.0, 5.0],
                                                              column_major=True, callback=plot_line) as subplot_id:
                                                dpg.add_plot_legend()
                                                for line in lineList:
                                                    filtered2 = list(filter(lambda x: x['line'] == line, filtered))
                                                    if len(filtered2) != 0:
                                                        with dpg.plot(tag=f'{type}{line}', anti_aliased=True, ):
                                                            dpg.add_plot_axis(dpg.mvXAxis, label="Frames")
                                                            dpg.set_item_label(f'{type}{line}', line)
                                                            with dpg.plot_axis(dpg.mvYAxis, label="Probability"):
                                                                dpg.add_line_series([0, 0], [0, 1.1],
                                                                                    tag=f'{type}{line} yaxis')
                                                                for item in filtered2:
                                                                    dpg.add_line_series(X, item["data"],
                                                                                        label=item["type"].replace(type,
                                                                                                                   ""),
                                                                                        use_internal_label=False)
                                            with dpg.group() as group:
                                                dpg.add_text(f"Accuracy {type}")
                                                filtered3 = []
                                                for i in filtered:
                                                    if not any(d['type'] == i["type"] for d in filtered3):
                                                        filtered3.append(i)
                                                for resul in filtered3:
                                                    dpg.add_text(
                                                        f'{resul["type"].replace(type, "")} : {resul["accuracy"]:.5f}%')
                                                dpg.add_slider_int(width=-1, max_value=trustworthyAI.principle.datasize,
                                                                   callback=plot_line, tag=f'{type} slider')
                                                with dpg.drawlist(width=400, height=350):
                                                    dpg.draw_image("texture_tag", (0, 0), (400, 300), uv_min=(0, 0),
                                                                   uv_max=(1, 1))
                            with dpg.tab(label="Tools"):
                                principle.useTools()
                                tools = principle.toolResults
                                for t in tools:
                                    dpg.add_text(f'{t["tool"]} : {t["result"]}')
                    else:
                        dpg.add_text("No results")
    dpg.create_viewport(title='Trustworthy AI')
    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
