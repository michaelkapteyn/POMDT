from pomdt.utils import *

import numpy as np
from itertools import chain, starmap, product

class damageLibrary():
    def __init__(self, damageLibraryJSONPath):
        dlib = read_json_file(damageLibraryJSONPath)
        self.components = list(dlib["components"].keys())
        self.n_components = len(self.components)
        self.states = dlib["components"]
        self.modes = list(product(*self.states.values()))
        modeIdxs = list(product(*[range(0,len(self.states[self.components[0]])),range(0,len(self.states[self.components[1]]))]))

        self.predictedMeasurements = flatten_dict(dlib["predictedMeasurements"], self.n_components)
        self.loadCases = list(self.predictedMeasurements[(self.modes[0][0],self.modes[0][1])].keys())

        self.transitionProbabilities = dlib["transitionProbabilities"]
        self.TPM = {}
        for load in self.loadCases:
            self.TPM[load] = np.zeros((len(self.modes),len(self.modes)))
            for fromIdx, fromMode in enumerate(modeIdxs):
                for toIdx, toMode in enumerate(modeIdxs):
                    self.TPM[load][fromIdx,toIdx] = self.transitionProbabilities[self.components[0]][load][fromMode[0]][toMode[0]] * self.transitionProbabilities[self.components[1]][load][fromMode[1]][toMode[1]]

        self.nSensors = len(list(self.predictedMeasurements[(self.modes[0][0],self.modes[0][1])][self.loadCases[0]]))

def flatten_dict(dictionary,n_components):
    """Flatten a nested dictionary structure"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in a dictionary"""
        try:
            items = parent_value.items()
        except AttributeError:
            # parent_value was not a dict, no need to flatten
            yield (parent_key, parent_value)
        else:
            for key, value in items:
                yield (parent_key + (key,), value)

    # Put each key into a tuple to initiate building a tuple of subkeys
    dictionary = {(key,): value for key, value in dictionary.items()}

    for i in range(n_components-1): #want to leave the final layer as a dict (with loadcases as keys)
        # Keep unpacking the dictionary until all value's are not dictionary's
        dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
        if not any(isinstance(value, dict) for value in dictionary.values()):
            break

    return dictionary
