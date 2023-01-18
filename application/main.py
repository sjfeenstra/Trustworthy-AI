import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv(dotenv_path=Path('config.env'))

from transparency import Transparency
from technicalRobustnessAndSafety import TechnicalRobustnessAndSafety
from trustworthyAI import TrustworthyAI

if __name__ == "__main__":
    dpg.create_context()

    # Create all available principles
    TRaSPrinciple = TechnicalRobustnessAndSafety()
    transparencyPrinciple = Transparency()

    # add principles to list
    principles = [TRaSPrinciple, transparencyPrinciple]

    # Create context class
    trustworthyAI = TrustworthyAI(TRaSPrinciple)

    cap = cv2.VideoCapture(os.getenv('FCAM'))

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break
        if len(frames) == trustworthyAI.principle.datasize:
            cap.release()
            break
        frames.append(frame)

    a = np.arange(0, trustworthyAI.principle.datasize)
    X = [float(x) for x in a]

    lineList = ["Far Left", "Close Left", "Close Right", "Far Right"]
    typeList = ["Traffic Convention", "Desire", "CameraFrames"]

    width, height, channels, data = dpg.load_image('../data/roadlines.png')
    width2, height2, channels2, data2 = dpg.load_image('../data/openpilot.png')
    width3, height3, channels3, data3 = dpg.load_image('../data/cameraexample.png')
    width4, height4, channels4, data4 = dpg.load_image('../data/widecameraexample.png')


    def getTextureData(frames, framenumber):
        data = np.flip(frames[framenumber], 2)
        data = data.ravel()
        data = np.asfarray(data, dtype='f')
        texture_data = np.true_divide(data, 255.0)
        return texture_data


    def button_callback(sender, app_data, user_data):
        dpg.configure_item("button", show=False)
        for principle in principles:
            user_data.principle = principle
            user_data.do_tests()
            user_data.update_data()


    def tab_callback(sender, app_data, user_data):
        trustworthyAI.principle = dpg.get_item_user_data(app_data)


    def plot_line(sender, app_data, user_data):
        aliases = dpg.get_aliases()
        filteredAliases = filter(lambda x: "line" in x, aliases)
        for alias in filteredAliases:
            dpg.set_value(alias, [[app_data, app_data], [0, 1.1]])
        filteredAliases2 = filter(lambda x: "slider" in x, aliases)
        for alias in filteredAliases2:
            dpg.set_value(alias, app_data)
        dpg.set_value("texture_tag", getTextureData(frames, app_data))


    def modelStructureRecursion(data):
        if "objects" in data:
            for ob in data["objects"]:
                if "objects" in ob:
                    with dpg.tree_node(label=ob['objectType'] + " " + ob['name'] + " (" + str(ob['size']) + ")"):
                        modelStructureRecursion(ob)
                else:
                    dpg.add_text(ob['objectType'] + " " + ob['name'] + " (" + str(ob['size']) + ")")
        else:
            dpg.add_text(data['objectType'] + " " + data['name'] + " (" + str(data['size']) + ")")


    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(frame_width, frame_height, getTextureData(frames, 0), format=dpg.mvFormat_Float_rgb,
                            tag="texture_tag")
        dpg.add_static_texture(width, height, data, tag="image_id")
        dpg.add_static_texture(width2, height2, data2, tag="image_id2")
        dpg.add_static_texture(width3, height3, data3, tag="image_id3")
        dpg.add_static_texture(width4, height4, data4, tag="image_id4")

    # application contextmanager
    with dpg.window(tag="Primary Window", height=1200, width=1200) as window:
        with dpg.tab_bar(tag="tab_bar", callback=tab_callback):
            with dpg.tab(label="Overview"):
                with dpg.tab_bar(tag="tab_bar2"):
                    with dpg.tab(label="Overview page") as tb1:
                        dpg.add_text("Weglijnen zijn enorm belangrijk voor deze applicatie. In het figuur hieronder "
                                     "worden 4 verschillende weglijnen aangeduidt met afkortingen, deze afkortingen "
                                     "staan voor:", wrap=1200)
                        dpg.add_text("FL: Far Left (Ver Links)")
                        dpg.add_text("CL: Close Left (Dichtbij Links)")
                        dpg.add_text("CR: Close Right (Dichtbij Rechts)")
                        dpg.add_text("FR: Far Right (Ver Rechts)")
                        dpg.add_separator()
                        with dpg.drawlist(width=600, height=400):
                            dpg.draw_image("image_id", (0, 0), (600, 400), uv_min=(0, 0), uv_max=(1, 1))
                        dpg.add_separator()
                        dpg.add_button(label="Uitvoer Testen", callback=button_callback, user_data=trustworthyAI,
                                       tag="button")

                    with dpg.tab(label="Betrouwbare KI"):
                        f = open(os.getenv('TRUSTWORTHYAI'))
                        data = json.load(f)
                        dpg.add_text("Principes van Betrouwbare KI van AI HLEG:")
                        dpg.add_separator()
                        for principle in data['principles']:
                            dpg.add_text(principle['name'])
                            dpg.add_text(principle['description'], wrap=1200)
                            dpg.add_separator()
                        f.close()

                    with dpg.tab(label="Openpilot model", tag="model_tab"):
                        f = open(os.getenv('MODELINFORMATION'))
                        data = json.load(f)
                        with dpg.collapsing_header(label="Model visualisatie"):
                            dpg.add_separator()
                            with dpg.drawlist(width=500, height=300):
                                dpg.draw_image("image_id2", (0, 0), (500, 300), uv_min=(0, 0), uv_max=(1, 1))
                            dpg.add_text(data['information'], wrap=1200)
                            dpg.add_separator()

                        with dpg.collapsing_header(label="Beschrijving inputs model"):
                            dpg.add_separator()
                            for input in data['inputs']:
                                dpg.add_text(input['name'] + " : " + input['size'])
                                dpg.add_text(input['description'], wrap=1200)
                                if "cameraexample" not in input["example"]:
                                    dpg.add_text(input['example'], wrap=1200)
                                if input['name'] == "Image Stream":
                                    with dpg.drawlist(width=600, height=150):
                                        dpg.draw_image("image_id3", (0, 0), (600, 150), uv_min=(0, 0), uv_max=(1, 1))
                                if input['name'] == "Wide Image Stream":
                                    with dpg.drawlist(width=600, height=150):
                                        dpg.draw_image("image_id4", (0, 0), (600, 150), uv_min=(0, 0), uv_max=(1, 1))
                                dpg.add_separator()

                        with dpg.collapsing_header(label="Beschrijving resultaten model"):
                            dpg.add_separator()
                            for output in data['outputs']:
                                dpg.add_text(output['name'] + " : " + output['size'])
                                dpg.add_text(output['description'], wrap=1200)
                                dpg.add_separator()
                        f.close()

                        with dpg.collapsing_header(label="Technische opbouw resultaten"):
                            dpg.add_separator()
                            f = open(os.getenv('RESULTSINFORMATION'))
                            data = json.load(f)
                            modelStructureRecursion(data)
                            f.close()

            # add results pages for the principles
            for principle in principles:
                with dpg.tab(label=principle.name, tag=principle.name):
                    with dpg.tab_bar(parent=principle.name):
                        for type in typeList:
                            with dpg.tab(label=type):
                                with dpg.group(horizontal=True):
                                    with dpg.subplots(2, 2, height=-1, width=-400, row_ratios=[5.0, 5.0],
                                                      column_ratios=[5.0, 5.0],
                                                      column_major=True, link_all_x=True):
                                        dpg.add_plot_legend()
                                        for line in lineList:
                                            with dpg.plot(tag=f'{principle.name}{type}{line}plot', anti_aliased=True):
                                                dpg.set_item_label(f'{principle.name}{type}{line}plot', line)
                                                dpg.add_plot_axis(dpg.mvXAxis, label="Frames",
                                                                  tag=f'{principle.name}{type}{line} xaxis')
                                                with dpg.plot_axis(dpg.mvYAxis, label="Probability",
                                                                   tag=f'{principle.name}{type}{line} yaxis'):
                                                    dpg.set_axis_limits(dpg.last_item(), -0.05, 1.05)
                                                    dpg.add_line_series([0, 0], [-0.05, 1.05],
                                                                        tag=f'{principle.name}{type}{line} line')
                                    with dpg.group(tag=f'{principle.name}{type} group'):
                                        dpg.add_slider_int(width=-1, max_value=trustworthyAI.principle.datasize - 2,
                                                           callback=plot_line, tag=f'{principle.name}{type} slider')
                                        dpg.add_separator()
                                        with dpg.drawlist(width=400, height=300):
                                            dpg.draw_image("texture_tag", (0, 0), (400, 300), uv_min=(0, 0),
                                                           uv_max=(1, 1))
                                        dpg.add_separator()
                                        with dpg.group(horizontal=True):
                                            dpg.add_group(tag=f'{principle.name}{type} type group')
                                            dpg.add_group(tag=f'{principle.name}{type} accuracy group')
    dpg.create_viewport(title='Trustworthy AI')
    dpg.set_primary_window("Primary Window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
