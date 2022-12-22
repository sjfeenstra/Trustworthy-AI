import numpy as np
import onnxruntime
from principle import Principle
import dearpygui.dearpygui as dpg

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

    def setPrinciple(self, p: Principle):
        self._principle = p

    def showTrustworthyAI(self):
        dpg.add_text("Principes van Betrouwbare KI van AI HLEG:")
        dpg.add_text("-------------------------------------------")
        dpg.add_text("Menselijke controle en menselijk toezicht")
        dpg.add_text("Technische robuustheid en veiligheid")
        dpg.add_text("Privacy en data-governance")
        dpg.add_text("Transparantie")
        dpg.add_text("Diversiteit, non-discriminatie en rechtvaardigheid")
        dpg.add_text("Maatschappelijk en milieuwelzijn")
        dpg.add_text("Verantwoording")

    def showModel(self):
        with dpg.tree_node(label="ModelOutputPlans plans; (4955)"):
            with dpg.tree_node(label="ModelOutputPlanPrediction prediction *5"):
                with dpg.tree_node(label="ModelOutputPlanElement mean *33 (495)"):
                    with dpg.tree_node(label="ModelOutputXYZ position"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ velocity"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ acceleration"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation_rate"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputPlanElement mean *33 (495)"):
                    with dpg.tree_node(label="ModelOutputXYZ position"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ velocity"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ acceleration"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation_rate"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                dpg.add_text("float prob")
        with dpg.tree_node(label="ModelOutputLaneLines lane_lines; (536)"):
            with dpg.tree_node(label="ModelOutputLinesXY mean; (264)"):
                with dpg.tree_node(label="ModelOutputYZ  left_far *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  left_near *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  right_near *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  right_far *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputLinesXY std; (264)"):
                with dpg.tree_node(label="ModelOutputYZ  left_far *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  left_near *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  right_near *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ  right_far *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputLinesProb prob; (8)"):
                with dpg.tree_node(label="ModelOutputLinesProb prob; (8)"):
                    with dpg.tree_node(label="ModelOutputLineProbVal left_far"):
                        dpg.add_text("val_deprecated")
                        dpg.add_text("val")
                    with dpg.tree_node(label="ModelOutputLineProbVal left_near"):
                        dpg.add_text("val_deprecated")
                        dpg.add_text("val")
                    with dpg.tree_node(label="ModelOutputLineProbVal right_near"):
                        dpg.add_text("val_deprecated")
                        dpg.add_text("val")
                    with dpg.tree_node(label="ModelOutputLineProbVal right_far"):
                        dpg.add_text("val_deprecated;")
                        dpg.add_text("val")
        with dpg.tree_node(label="ModelOutputRoadEdges road_edges; (264)"):
            with dpg.tree_node(label="ModelOutputEdgessXY mean; (132)"):
                with dpg.tree_node(label="ModelOutputYZ left; *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ right; *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputEdgessXY std; (132)"):
                with dpg.tree_node(label="ModelOutputYZ left; *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
                with dpg.tree_node(label="ModelOutputYZ right; *33 (66)"):
                    dpg.add_text("float y")
                    dpg.add_text("float z")
        with dpg.tree_node(label="ModelOutputLeads leads; (105)"):
            with dpg.tree_node(label="ModelOutputLeadPrediction prediction *2 (102)"):
                with dpg.tree_node(label="ModelOutputLeadElement mean; *6 (24)"):
                    dpg.add_text("float x")
                    dpg.add_text("float y")
                    dpg.add_text("float velocity")
                    dpg.add_text("float acceleration")
                with dpg.tree_node(label="ModelOutputLeadElement std; *6 (24)"):
                    dpg.add_text("float x")
                    dpg.add_text("float y")
                    dpg.add_text("float velocity")
                    dpg.add_text("float acceleration")
                dpg.add_text("float prob *3  (3)")
            dpg.add_text("float prob *3  (3)")
        with dpg.tree_node(label="ModelOutputStopLines stop_lines; (52)"):
            with dpg.tree_node(label="ModelOutputStopLinePrediction prediction; *3 (51)"):
                with dpg.tree_node(label="ModelOutputStopLineElement mean; (8)"):
                    with dpg.tree_node(label="ModelOutputXYZ position;"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation;"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    dpg.add_text("float speed")
                    dpg.add_text("float time")
                with dpg.tree_node(label="ModelOutputStopLineElement std; (8)"):
                    with dpg.tree_node(label="ModelOutputXYZ position;"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    with dpg.tree_node(label="ModelOutputXYZ rotation;"):
                        dpg.add_text("float x")
                        dpg.add_text("float y")
                        dpg.add_text("float z")
                    dpg.add_text("float speed")
                    dpg.add_text("float time")
                dpg.add_text("float prob")
            dpg.add_text("float prob")
        with dpg.tree_node(label="ModelOutputMeta meta; (88)"):
            with dpg.tree_node(label="ModelOutputDesireProb desire_state_prob; (8)"):
                dpg.add_text("float none")
                dpg.add_text("float turn_left")
                dpg.add_text("float turn_right")
                dpg.add_text("float lane_change_left")
                dpg.add_text("float lane_change_right")
                dpg.add_text("float keep_left")
                dpg.add_text("float keep_right")
                dpg.add_text("float null")
            dpg.add_text("float engaged_prob")
            with dpg.tree_node(label="ModelOutputDisengageProb disengage_prob; *5 (35)"):
                dpg.add_text("float gas_disengage;")
                dpg.add_text("float brake_disengage;")
                dpg.add_text("float steer_override;")
                dpg.add_text("float brake_3ms2;")
                dpg.add_text("float brake_4ms2;")
                dpg.add_text("float brake_5ms2;")
                dpg.add_text("float gas_pressed;")
            with dpg.tree_node(label="ModelOutputDisengageProb disengage_prob; *5 (35)"):
                dpg.add_text("float left;")
                dpg.add_text("float right;")
            with dpg.tree_node(label="ModelOutputDesireProb desire_pred_prob; *4 (32)"):
                dpg.add_text("float none")
                dpg.add_text("float turn_left")
                dpg.add_text("float turn_right")
                dpg.add_text("float lane_change_left")
                dpg.add_text("float lane_change_right")
                dpg.add_text("float keep_left")
                dpg.add_text("float keep_right")
                dpg.add_text("float null")
        with dpg.tree_node(label="ModelOutputPose pose; (12)"):
            with dpg.tree_node(label="ModelOutputXYZ velocity_mean;"):
                dpg.add_text("float x")
                dpg.add_text("float y")
                dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputXYZ rotation_mean;;"):
                dpg.add_text("float x")
                dpg.add_text("float y")
                dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputXYZ velocity_std;;"):
                dpg.add_text("float x")
                dpg.add_text("float y")
                dpg.add_text("float z")
            with dpg.tree_node(label="ModelOutputXYZ rotation_std;;"):
                dpg.add_text("float x")
                dpg.add_text("float y")
                dpg.add_text("float z")
        dpg.add_text("ModelState current_state; (512)")



