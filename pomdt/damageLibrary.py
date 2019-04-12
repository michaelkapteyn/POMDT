from shmm.utils import *

import numpy as np

class damageLibrary():
    def __init__(self, damageLibraryJSONPath):
        self.predictedMeasurements = read_json_file(damageLibraryJSONPath)
        self.states = list(self.predictedMeasurements.keys())
        self.nSensors = len(list(self.predictedMeasurements.values())[0])
