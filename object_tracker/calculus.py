from   math import sqrt
from shapely.geometry import Polygon

def area(box):
    return abs((box[2]-box[0])*(box[3]-box[1]))

def box2Corners(ltrb):
    return [(ltrb[0],ltrb[1]), (ltrb[2],ltrb[1]), (ltrb[2],ltrb[3]), (ltrb[0],ltrb[3])]

def getCentroid(points):
    cx = cy = 0.0
    for point in points:
        cx += point[0]
        cy += point[1]
    cx /= float(len(points))
    cy /= float(len(points))
    return (cx, cy)

def getDistance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def intersects(polyPoints1, polyPoints2):
    return Polygon(polyPoints1).intersects(Polygon(polyPoints2))

def iou(b1, b2):
    l, t = max(b1[0], b2[0]), max(b1[1], b2[1])
    r, b = min(b1[2], b2[2]), min(b1[3], b2[3])
    if l >= r or t >= b: 
        return 0.0 # no intersection at all
    interArea = max(0, r-l+1) * max(0, b-t+1)
    b1Area = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
    b2Area = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
    return interArea / float(b1Area + b2Area - interArea)
