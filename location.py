import math
import numpy as np


def clamp_lat_lon(lat_attr, lon_attr):
    lat = float(lat_attr)
    lon = float(lon_attr)

    if lat > 90:
        lat = 90
    elif lat < -90:
        lat = -90

    if lon > 180.0:
        lon = 180.0
    elif lon < -180.0:
        lon = -180.0

    return lat, lon


class Location:
    name = ""
    latitude = 0
    longitude = 0
    point = np.empty
    attr1 = False
    attr2 = False
    classification = None
    class_distance = None


    def __init__(self, row):
        self.name = row["Name"]
        self.latitude, self.longitude = clamp_lat_lon(row["Latitude"], row["Longitude"])
        self.point = np.array([self.latitude, self.longitude])
        self.attr1 = row["Attribute1"]
        self.attr2 = row["Attribute2"]
