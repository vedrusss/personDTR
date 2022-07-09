from copy    import deepcopy
from typing  import List, Dict
from structs import RoI, Trajectory
from objectTracker.selectedObjects import SelectedObjects

THRESHOLD  = 40
TRACE_SIZE = 15
DIST_THR_SQ = (THRESHOLD * (1+2/10)) ** 2

class NikitaDistanceTracker(SelectedObjects):
    def __init__(self, useConstThreshold=True, verbose=False):
        SelectedObjects.__init__(self)
        self.__trajectories : Dict[int:Trajectory] = {}
        self.__nextTraceId = 0
        self.__frameNum = 0
        self.__useConstThreshold = useConstThreshold
        self.__verbose : bool = verbose

    def __call__(self, frame, detections: List[RoI], newSelections: List[RoI] = []):
        rois = self.__processFrame(self.__frameNum, detections, newSelections)

        rois = [trajectory.getLastRoI().setFeature('tail', [roi.getCentroid() 
            for roi in trajectory.getLastNRoIs(TRACE_SIZE)]) for _,trajectory in self.__trajectories.items()
                                                                               if trajectory.getRedetected()]

        debugInfo = {'notCorrectedTrackedRoIs':rois}
        self.__frameNum += 1
        return rois, debugInfo

    def __getDistances(self, prevDetections : List[RoI], newDetections : List[RoI],
                       trackedIndexes, distanceThresholdSQ):
        if self.__useConstThreshold: distanceThresholdSQ = DIST_THR_SQ
        pDistances = {}
        for i, d1 in enumerate(newDetections):
            cx1, cy1 = d1.getCentroid()
            dist = {}
            for d2 in prevDetections:
                ind2 = d2.getInd()
                if ind2 in trackedIndexes: continue # if detection has been already paired skip it
                cx2, cy2 = d2.getCentroid()
                d = (cx1 - cx2)**2 + (cy1 - cy2)**2 # save all cross distances
                if d <= distanceThresholdSQ: dist[ind2] = d # if closer than threshold
            #   there might be no close pair
            if len(dist): #   { 0: [(key, dist),..], .. }
                pDistances[i] = sorted(dist.items(), key=lambda kv: kv[1])
        return pDistances

    def __getTrackedDetections(self, frameNum):
        detections = []
        for _,trajectory in self.__trajectories.items():
            for roi in reversed(trajectory.getLastNRoIs()):
                if roi.getFeature('fnum') == frameNum:
                    detections.append(roi)
                    break
        return detections

    def __matchDetections(self, frameNum, detections : List[RoI]):
        i = 1 
        maxTrajectoryLength = max([t.getLength() for _,t in self.__trajectories.items()]) \
                                                 if len(self.__trajectories) else 0
        traceTailLength   = TRACE_SIZE if maxTrajectoryLength > TRACE_SIZE else maxTrajectoryLength
        leftDetections    = deepcopy(detections)
        matchedDetections = []

        while i <= traceTailLength and len(leftDetections):
            prevDetections = self.__getTrackedDetections(frameNum - i)
            trackedIndexes = [d.getInd() for d in matchedDetections]
            distanceThresholdSQ = (THRESHOLD * (1+(i-1)*2/10)) ** 2 # look i frames back
            p_distances = self.__getDistances(prevDetections, leftDetections, trackedIndexes, distanceThresholdSQ)
            while len(p_distances):
                #   among all trackedDetections pick closest distance to prev frame
                #   [ (0, [(key, dist),..]), .. }
                p_distance_sorted = sorted(p_distances.items(), key=lambda kv: kv[1][0][1])
                #   p_distance_sorted[0][0] - "t"
                detection = leftDetections[p_distance_sorted[0][0]]
                leftDetections[p_distance_sorted[0][0]] = None
                ind = p_distance_sorted[0][1][0][0]
                detection.setInd(ind)
                matchedDetections.append(detection)
                #   ind detection pair was found, remove from the list and repeat
                del p_distances[p_distance_sorted[0][0]]
                p_distances_new = {}
                for t, dist_sorted in p_distances.items():
                    new_dist_sorted = [k for k in dist_sorted if k[0] != ind]
                    if len(new_dist_sorted): p_distances_new[t] = new_dist_sorted
                p_distances = p_distances_new

            leftDetections = [detection for detection in leftDetections if detection is not None]
            if len(matchedDetections) == len(detections):
                break
            i += 1
        return matchedDetections, leftDetections

    def __createTracker(self, frameNum, detection : RoI):
        self.__trajectories[self.__nextTraceId] = \
        Trajectory( detection.setInd(self.__nextTraceId).setFeature('fnum',frameNum) )
        self.__nextTraceId += 1

    def __processFrame(self, frameNum, detections: List[RoI], newSelections: List[RoI] = []):
        # match new detections to existing trajectories
        matchedDetections, leftDetections = self.__matchDetections(frameNum, detections)           

        # set re-detected to False, then set True for matched ones
        for _,trajectory in self.__trajectories.items():
            trajectory.setRedetected(False)

        # proceed existing trajectories with matched detections
        for detection in matchedDetections:
            ind = detection.getInd()
            trajectory = self.__trajectories.get(ind)
            assert(trajectory is not None),"Detection was matched to unexisting trajectory ({})".format(ind)
            assert(trajectory.getLastRoI().getInd() == ind),"Trajectory ind {} is not equal to RoI ind {}".\
                format(trajectory.getLastRoI().getInd(), ind)
            trajectory.setRedetected(True).appendRoI(detection.setFeature('fnum',frameNum))

        # append new detections (create new trajectories)

        for selection in newSelections:
            self.__createTracker(frameNum, selection)

        if not self.isSelected: # create trackers from new detections in not selection mode only
            for detection in leftDetections:
                self.__createTracker(frameNum, detection)

#        if self.isSelected: 
#            leftDetections = self._leaveSelected(leftDetections)
        
#        for detection in leftDetections: 
#            self.__createTracker(frameNum, detection)
#            self.__trajectories[self.__nextTraceId] = \
#                Trajectory( detection.setInd(self.__nextTraceId).setFeature('fnum',frameNum) )
#            self.__nextTraceId += 1