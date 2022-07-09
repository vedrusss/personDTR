from structs import RoI
from typing  import List
from calculus import iou
from copy    import deepcopy

class SelectedObjects:
    def __init__(self):
        self.__selections : List[RoI] = []

    def addSelected(self, objects : List[RoI]):
        self.__selections += deepcopy(objects)
        return self

    def initSelected(self, objects : List[RoI]):
        self.__selections = deepcopy(objects)
        return self

    def checkNewObjects(self, objects : List[RoI]):
        selectedIds = [roi.getInd() for roi in self.__selections]
        newObjects = [roi for roi in objects if not roi.getInd() in selectedIds]
        return newObjects

    @property
    def isSelected(self):
        return len(self.__selections) > 0

    def __intersectsAny(self, box, threshold):
        for obj in self.__selections:
            if iou(obj.getBox(), box) > threshold: return True
        return False

    def _leaveSelected(self, objects : List[RoI]):
        intersectionThreshold = 0.84
        while len(objects) > len(self.__selections):
            objects = [obj for obj in objects if self.__intersectsAny(obj.getBox(), intersectionThreshold)]
            intersectionThreshold += 0.03
        return objects
