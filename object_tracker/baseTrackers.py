import cv2
import dlib
import numpy as np
from structs import RoI, Trajectory
from copy    import deepcopy
from skimage.metrics import structural_similarity as compare_ssim

createBaseTracker  = lambda frame, roi, ind, trackerType, verbose: \
                     OpenCVTracker(frame, roi, ind, trackerType, verbose) if trackerType in OpenCVTrackers else \
                     DlibCNTracker(frame, roi, ind, trackerType, verbose)

OpenCVTrackers = {
    'BOOSTING'  : lambda: cv2.legacy.TrackerBoosting_create(),
    'MIL'       : lambda: cv2.TrackerMIL_create(),
    'KCF'       : lambda: cv2.TrackerKCF_create(),
    'TLD'       : lambda: cv2.legacy.TrackerTLD_create(),
    'MEDIANFLOW': lambda: cv2.legacy.TrackerMedianFlow_create(),
    'GOTURN'    : lambda: cv2.TrackerGOTURN_create(),
    'MOSSE'     : lambda: cv2.legacy.TrackerMOSSE_create(),
    'CSRT'      : lambda: cv2.TrackerCSRT_create()
}

BaseTrackers = list(OpenCVTrackers.keys()) + ['DLIB']

def getImageGrayChip(image, box):
    box = [int(el) for el in box]
    if len(image.shape) > 2: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[box[1]:box[3],box[0]:box[2]]

def getSimilarity(image, box, template):
    if template is None: return -2
    chip = getImageGrayChip(image, box)
    h, w = template.shape[:2]
    #chip = cv2.resize(chip, (w, h))
    (score, diff) = compare_ssim(chip, template, full=True)
    return score

class BaseTracker:
    def __init__(self, frame, roi, ind, trackerType, verbose):
        assert(trackerType in BaseTrackers),"{} is not implemented. Try one of {}".format(trackerType, BaseTrackers)
        self.__trackerType = trackerType
        self.__verbose = verbose
        self.__ind = ind
        self.__fvs = []
        self.__fvsLimit = 20
        self.__activeTime = 0
        self.__initialize(frame, roi)

    def __initialize(self, frame, roi):
        self._initialize(frame, roi.getBox(), self.__trackerType)
        self.__trajectory = Trajectory(deepcopy(roi).setInd(self.__ind))
        self.__trajectory.getLastRoI().setFeature('chip', getImageGrayChip(frame, roi.getBox()))
        self.__active = True
        self.__redetected  = True
        self.__notApproved = 0
        if self.__verbose: print("Tracker {} initialized with position {} {}".format(self.getInd(), self.getBox(), self.getCentroid()))

    def _initialize(self, frame, box, trackerType):
        raise NotImplementedError

    def addFV(self, fv):
        if len(self.__fvs) > self.__fvsLimit: self.__fvs = self.__fvs[1:]
        self.__fvs.append(fv)

    def checkRedetection(self):
        if self.__redetected: self.__notApproved  = 0
        else:                 self.__notApproved += 1
        if self.__verbose: print("Tracker {} was re-detected {} (notApproved count {})".format(self.getInd(), self.__redetected, self.__notApproved))
        if self.__notApproved >= 10: # TODO: this parameter must be picked up or calculated
           self.setActive(False)
        if self.getActive(): self.__activeTime += 1
        return self

    def getActive(self):
        return self.__active

    def getActiveTime(self):
        return self.__activeTime

    def getFVs(self):
        return self.__fvs

    def getFVsLimit(self):
        return self.__fvsLimit

    def getInd(self):
        return self.__ind

    def getCurRoI(self):
        return self.__trajectory.getLastRoI()

    def getLastNRoIs(self, numRoIs):
        end = self.__trajectory.getLength()
        beg = end - numRoIs
        if beg < 0: beg = 0
        return [self.__trajectory.getRoI(ind) for ind in range(beg, end)]

    def getPrevRoI(self):
        trajectoryLength = self.__trajectory.getLength()
        if trajectoryLength < 2: 
            return self.getCurRoI()
        return self.__trajectory.getRoI(trajectoryLength - 2)

    def getRedetected(self):
        return self.__redetected
    
    @staticmethod
    def fit2Frame(ltrb, hw):
        h, w       = hw
        l, t, r, b = ltrb
        if l  < 1: l = 1
        if t  < 1: t = 1
        if r >= w: r = w - 1
        if b >= h: b = h - 1
        return (l, t, r, b)

    @staticmethod
    def ltrb2ltwh(ltrb):
        return (ltrb[0], ltrb[1], ltrb[2]-ltrb[0], ltrb[3]-ltrb[1])

    @staticmethod
    def ltwh2ltrb(ltwh):
        return (ltwh[0], ltwh[1], ltwh[0]+ltwh[2], ltwh[1]+ltwh[3])

    def setActive(self, active):
        self.__active = active
        if self.__verbose: print("Tracker {} set active {}".format(self.getInd(), active))
        return self

    def setBox(self, frame, box):
        box = [int(el) for el in box]
        self.__trajectory.getLastRoI().setBox(box)
        self.__trajectory.getLastRoI().setFeature('chip', getImageGrayChip(frame, box))
        self._initialize(frame, box, self.__trackerType)
        if self.__verbose: print("Tracker {} box position corrected to {} {}".format(self.getInd(), self.getBox(), self.getCentroid()))
        return self

    def setRedetected(self, redetected):
        self.__redetected = redetected
        return self

    def update(self, frame):
        box = self._update(frame)
        if box is None:
            if self.__verbose: print("Tracker {} couldn't update box position. Keep position {} {}".format(self.getInd(), self.getBox(), self.getCentroid()))
            return False
        self.__trajectory.appendRoI(RoI(box, self.getInd()).setFeature('chip', getImageGrayChip(frame, box)))
        if self.__verbose: print("Tracker {} successfully updated position to {} {}".format(self.getInd(), self.getBox(), self.getCentroid()))
        return True

    def _update(self, frame):
        raise NotImplementedError


class OpenCVTracker(BaseTracker):
    def _initialize(self, frame, box, trackerType):
        self.tracker = OpenCVTrackers[trackerType]()
        try:
            self.tracker.init(frame, BaseTracker.ltrb2ltwh(box))
        except Exception as e:
            print("Cannot initialize tracker {} with box {} onto frame with shape {}".format(trackerType, box, frame.shape))
            print("Error: {}".format(e))

    def _update(self, frame):
        success, box = self.tracker.update(frame)
        box = tuple([int(el) for el in box])
        if success: return BaseTracker.fit2Frame(BaseTracker.ltwh2ltrb(box), frame.shape[:2])
        return None


class DlibCNTracker(BaseTracker):
    def _initialize(self, frame, box, trackerType):
        self.tracker = dlib.correlation_tracker()
        box  = [int(el) for el in box]
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        self.tracker.start_track(frame, rect)

    def _update(self, frame):
        try:
            self.tracker.update(frame)
            pos = self.tracker.get_position()
            # unpack the position object
            l, t = int(pos.left()),  int(pos.top())
            r, b = int(pos.right()), int(pos.bottom())
            return BaseTracker.fit2Frame((l, t, r, b), frame.shape[:2])
        except Exception as e:
            print("Dlib tracker error while update: {}".format(e))
            return None