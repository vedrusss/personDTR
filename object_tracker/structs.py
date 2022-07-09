import calculus
from copy  import deepcopy

class RoI:
    def __init__(self, box, ind=0, features={}):
        self.__box        = list(box)[:] # [left, top, right, bottom]
        self.__ind        = ind
        self.__features   = deepcopy(features)

    def assignFeatures(self, assignee):
        self.__features = deepcopy(assignee.__features)
        return self

    def getBox(self):
        return self.__box
  
    def getCentroid(self):
        return calculus.getCentroid(calculus.box2Corners(self.__box))

    def getFeature(self, name, default=None):
        if not name in self.__features: return default # self.__features[name] = default
        return self.__features[name]

    def getInd(self):
        return self.__ind

    def getSize(self):
        return self.__box[2]-self.__box[0], self.__box[3]-self.__box[1]

    def setBox(self, box):
        self.__box      = list(box)[:] # [left, top, right, bottom]
        return self

    def setFeature(self, name, value):
        self.__features[name] = deepcopy(value)
        return self

    def setInd(self, ind):
        self.__ind = ind
        return self

    def toStr(self):
        wh = int(self.__box[2] - self.__box[0]), int(self.__box[3] - self.__box[1])
        c  = int(self.__box[0] + wh[0]/2),       int(self.__box[1] + wh[1]/2)
        return "{}, {}".format(c, wh)  


class Trajectory:     # keeps RoIs (deep copies of them)
    def __init__(self, roi):
        self.__rois = []
        self.appendRoI(roi)
        self.__redetected = True

    def appendRoI(self, roi):
        self.__rois.append(deepcopy(roi))
        return self

    def getLength(self):
        return len(self.__rois)

    def getLastRoI(self):
        assert(self.getLength() > 0),"Requested last RoI from empty trajectory"
        return self.__rois[-1]

    def getLastNRoIs(self, amount=None):
        if amount is None: return self.__rois
        if amount > len(self.__rois): amount = len(self.__rois)
        return self.__rois[-amount:]

    def getRedetected(self):
        return self.__redetected

    def getRoI(self, ind):
        assert(ind < self.getLength()),"Requested RoI #{} from trajectory of length {}".format(ind, self.getLength())
        return self.__rois[ind]

    def setRedetected(self, redetected):
        self.__redetected = redetected
        return self