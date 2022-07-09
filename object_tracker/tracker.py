import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from   typing    import List
from   structs   import RoI
import calculus
from   baseTrackers import createBaseTracker, BaseTracker
from   selectedObjects import SelectedObjects

class MultiTracker(SelectedObjects):
    def __init__(self, trackerType='MOSSE', verbose=False):
        SelectedObjects.__init__(self)
        self.__trackers    : List[BaseTracker] = []
        self.__trackerType : str               = trackerType
        self.__verbose     : bool              = verbose
        self.__frameNum = 0

    def __call__(self, frame, detections: List[RoI], newSelections: List[RoI] = []):
        # 1) update all created previous base trackers
        notCorrectedTrackedRoIs = self.__updateBaseTrackers(frame)  # notCorrectedTrackedRoIs are used for debug purposes

        if len(newSelections) > 0:
            self.__createTrackers(frame, newSelections)

        # 2) match detections with existing trackers; use not matching (new ones) to run new trackers
        newDetections = self.__matchTrackersDetections(frame, detections)
        
        # 3) re-tun inactive trackers if their FVs suit to newDetections
        used_inds = []
        for tracker in self.__trackers:
            if tracker.getActive(): continue  # check inactive trackers only
            for detection in newDetections:
                print(tracker.getInd())
                if self.__matchByFVs(tracker.getFVs(), detection.getFeature('fv'), 0.18):
                    tracker.setBox(frame, detection.getBox()).setRedetected(True).setActive(True)
                    tracker.getCurRoI().assignFeatures(detection)
                    used_inds.append(detection.getInd())
                    print(f"Matched new detection with inactive tracker {tracker.getInd()} with detection {detection.getInd()}")
        used_inds = set(used_inds)
        newDetections = [det for det in newDetections if det.getInd() not in used_inds]

        if not self.isSelected:  # create trackers from new detections in not selection mode only
            self.__createTrackers(frame, newDetections)

        # 4) run through active only trackers checking redetection flag, flag notActive if not redetected for N times consiquently
        for tracker in self.__trackers:
            if tracker.getActive():
                #if not tracker.getRedetected() and self.__verbose: print("Tracker {} was not redetected".format(tracker.getInd()))
                tracker.checkRedetection()
        
        # 5) find overlappings
        for     tracker1 in [t for t in self.__trackers if t.getActive()]:
            for tracker2 in [t for t in self.__trackers if t.getActive() and t.getInd() != tracker1.getInd()]:
                iou = calculus.iou(tracker1.getCurRoI().getBox(), tracker2.getCurRoI().getBox())
                if iou > 0.1:
                    tracker1.getCurRoI().getFeature('overlappings', []).append(tracker2.getInd())

        # 6) add new FV to gallery if it's too far from existing FVs and there is no overlappings with other RoIs
        for tracker in self.__trackers:
            if tracker.getRedetected() and len(tracker.getCurRoI().getFeature('overlappings', [])) == 0:
                new_fv = tracker.getCurRoI().getFeature('fv')
                #print('new_fv', new_fv)
                if not new_fv is None:
                    for fv in tracker.getFVs():
                        if cosine(fv, new_fv) > 0.2:  #  tune threshold
                            tracker.addFV(new_fv)
                            break
                    if len(tracker.getFVs()) == 0: tracker.addFV(new_fv)
            #print(f"tracker {tracker.getInd()} includes {len(tracker.getFVs())}")

        #  output data
        rois = [tracker.getCurRoI().setFeature('tail', [roi.getCentroid() for roi in tracker.getLastNRoIs(10)]) 
            for tracker in self.__trackers if tracker.getActive()]
        
        if self.__verbose: print("Active trackers: {}".format([t.getInd() for t in self.__trackers if t.getActive()]))
        if self.__verbose: print("Not corrected trackers: {}".format([roi.getInd() for roi in notCorrectedTrackedRoIs]))
        
        self.__frameNum += 1
        debugInfo = {'notCorrectedTrackedRoIs':notCorrectedTrackedRoIs,
                     'active':[t.getInd() for t in self.__trackers if t.getActive()],
                     'inactive':[t.getInd() for t in self.__trackers if not t.getActive() and len(t.getFVs()) > t.getFVsLimit()]}
        return rois, debugInfo

    def __closeEnough(self, detection: RoI, tracker: BaseTracker):
        return self.__closeEnoughByDistance(detection, tracker)

    def __closeEnoughByDistance(self, detection: RoI, tracker: BaseTracker):
        thresholdFactor = 0.3  # TODO: must be tuned
        prevRoI, curRoI = tracker.getPrevRoI(), tracker.getCurRoI()
        mx = abs(prevRoI.getCentroid()[0] - curRoI.getCentroid()[0])
        my = abs(prevRoI.getCentroid()[1] - curRoI.getCentroid()[1])
        w, h = curRoI.getSize() # (r-l), (b-t)
        tx, ty  = w*thresholdFactor + mx, 2.*h*thresholdFactor + my
        #t = sqrt(w*w + h*h)*thresholdFactor
        dx = abs(curRoI.getCentroid()[0] - detection.getCentroid()[0])
        dy = abs(curRoI.getCentroid()[1] - detection.getCentroid()[1])

        if dx > tx or dy > ty: return False 
        return True

    def __correctBox(self, tracker: BaseTracker, detection: RoI):
        preRoI = tracker.getPrevRoI()
        curRoI = tracker.getCurRoI()
        pc, pwh = np.array(preRoI.getCentroid()), np.array(preRoI.getSize())
        tc, twh = np.array(curRoI.getCentroid()), np.array(curRoI.getSize())
        dc, dwh = np.array(detection.getCentroid()),  np.array(detection.getSize())
        nc  = (tc + dc) / 2           # new box position must not be pulled to previous position
        #nwh = (pwh + twh + dwh) / 3  # but new box shape should be dependant of previous shape
        nwh = 0.25*pwh + 0.35*twh + 0.40*dwh  # weighted variant is more smart but needs fine tuning
        lt  = nc - nwh/2
        rb  = lt + nwh
        return [lt[0], lt[1], rb[0], rb[1]]

    def __createTracker(self, frame, detection: RoI):
        newInd = 0 if len(self.__trackers) == 0 else max([tracker.getInd() for tracker in self.__trackers]) + 1
        self.__trackers.append(createBaseTracker(frame, detection, newInd, self.__trackerType, False))
        if self.__verbose: print("Created tracker {} - {} from detection {}".format(newInd, detection.toStr(), detection.getInd()))
        return self.__trackers[newInd].getCurRoI()

    def __createTrackers(self, frame, detections: List[RoI]):
        rois = [self.__createTracker(frame, detection) for detection in detections]
        return rois, {'notCorrectedTrackedRoIs':rois}

    def __getDetectionsToTracksDistanceMatrix(self, detections: List[RoI]):
        # 1) check matching of detections to trackers
        dtMatrix = defaultdict(dict)
        usedTrackersInds = []
        for dInd, detection in enumerate(detections):
            for tInd, tracker in enumerate(self.__trackers):
                if not self.__closeEnough(detection, tracker):
                    continue
                trackerPoint   = tracker.getCurRoI().getCentroid()
                detectionPoint = detection.getCentroid()
                dtMatrix[dInd][tInd] = calculus.getDistance(trackerPoint, detectionPoint)
                usedTrackersInds.append(tInd)        
        usedTrackersInds = set(usedTrackersInds)
        # 2) check matching of trackers to detections
        tdMatrix = defaultdict(dict)
        usedDetectionsInds = []
        for tInd in usedTrackersInds:
            dInd_dist = {dInd:tInd_dist[tInd] for dInd, tInd_dist in dtMatrix.items() if tInd in tInd_dist}
            if len(dInd_dist) > 0:
                bestDetectionInd = sorted(dInd_dist, key=dInd_dist.get)[0]
                tdMatrix[tInd][bestDetectionInd] = dInd_dist[bestDetectionInd]
                usedDetectionsInds.append(bestDetectionInd)
        usedDetectionsInds = set(usedDetectionsInds)        
        # 3) make up resulting matrix of suitable detections to tracks distances
        detections_to_trackers_DistanceMatrix = defaultdict(dict)
        for dInd in usedDetectionsInds:
            for tInd in usedTrackersInds:
                if tInd not in tdMatrix: continue
                if dInd not in tdMatrix[tInd]: continue
                detections_to_trackers_DistanceMatrix[dInd][tInd] = tdMatrix[tInd][dInd]
        return detections_to_trackers_DistanceMatrix

    def __getClosestTracker(self, tInd, detection, closestTrackersInds, distanceThreshold):
        tracker = self.__trackers[tInd]
        return tracker   # TURN OFF COLOR USAGE LOGIG FOR NOW
        # THE METHOD MAY BE USED FURTHER FOR CORRELATION LOGIG TO TRY

        overlappingTrackerInds = [ind for ind in tracker.getCurRoI().getFeature('overlappings', []) 
                                       if ind in closestTrackersInds]
        candidates = {ind:dist for ind,dist in closestTrackersInds if ind != tInd and dist < distanceThreshold}
        candidates = sorted(candidates.items(), key=lambda x: x[1]) # sort ascending order by value
        if len(candidates) == 0:
            return tracker
        altTracker = self.__trackers[candidates[0][0]]
        detColor = detection.getFeature('color')
        t1Color  = tracker.getCurRoI().getFeature('color')
        t2Color  = altTracker.getCurRoI().getFeature('color')
        if abs(t1Color - detColor) < abs(t2Color - detColor):
            return tracker
        return altTracker        

    @staticmethod
    def __matchByFVs(tracked_fvs, det_fv, threshold):
        for fv in tracked_fvs:
            print(f"\t{cosine(fv, det_fv)} comparing with {threshold}")
            if cosine(fv, det_fv) < threshold: return True
        return False

    def __matchTrackersDetections(self, frame, detections: List[RoI]):
        # get distances between detections and trackers
        detections_to_trackers_DistanceMatrix = self.__getDetectionsToTracksDistanceMatrix(detections)
        # get new detections to create new trackers
        newDetections = [det for dInd, det in enumerate(detections) if dInd not in detections_to_trackers_DistanceMatrix]
        
        # match detections to closest active trackers and correct trackers' boxes using the detection's one
        for dInd, tInd_dist in detections_to_trackers_DistanceMatrix.items():
            closestTrackersInds = sorted(tInd_dist.items(), key=lambda x: x[1]) # sort ascending order by value
            closestTrackersInds = [(tInd, distance) for tInd, distance in closestTrackersInds 
                                    if self.__trackers[tInd].getActive()] # choose from active trackers only
            # TODO: shouldn't we resume inactive tracker???
            if len(closestTrackersInds) == 0:
                continue
            tInd, distance = closestTrackersInds[0] # we need just first (closest) active tracker
            detection = detections[dInd]             # TODO: Fine tune distance factor below
            tracker = self.__getClosestTracker(tInd, detection, closestTrackersInds, distance * 2.0)
            
            #  this is try of FV usage for matching
#            if not self.__matchByFVs(tracker.getFVs(), detection.getFeature('fv'), 0.18): continue
#            chip = tracker.getPrevRoI().getFeature('chip')
#            simi = getSimilarity(frame, detection.getBox(), chip)
#            if tracker.getInd() == 7: print(simi)
#            simi = simi[0][0]
#            if tracker.getInd() == 7: print(simi)
            
            correctedBox = self.__correctBox(tracker, detection)  # TODO: try other weights for correction
            tracker.setBox(frame, correctedBox).setRedetected(True).setActive(True)
            tracker.getCurRoI().assignFeatures(detection)
            #tracker.getCurRoI().setFeature('color', imageProc.getColor(frame, tracker.getCurRoI().getCentroid()))

#            tracker.getCurRoI().setFeature('similarity', simi)
            if self.__verbose: print("Detection {} {} matches to tracker {} - {} (initial {} with distance {:.3f})".
               format(detection.getInd(), detection.toStr(), tracker.getInd(), tracker.getCurRoI().toStr(), tInd, distance))
        
        return newDetections

    def __updateBaseTrackers(self, frame):
        notCorrectedTrackedRoIs = []  # used for debug purposes
        for tracker in self.__trackers:
            #if not tracker.getActive(): continue
            tracker.setRedetected(False)             
            if tracker.update(frame):
                tracker.setActive(True)
                notCorrectedTrackedRoIs.append(tracker.getCurRoI())
                #tracker.getCurRoI().setFeature('color', imageProc.getColor(frame, tracker.getCurRoI().getCentroid()))
            else:
                #tracker.setActive(False)  # DO NOT set inactive - LET it be updated by detection if redetected
                if self.__verbose: print("Tracker {} wasn't updated".format(tracker.getInd()))
        return notCorrectedTrackedRoIs
