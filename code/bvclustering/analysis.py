import numpy as np
import matplotlib.pyplot as plt
import json
import os


class Profile:
    def __init__(self, profile_id: str, x: float, y: float, profile: np.ndarray = None, is_in_control: bool = True) -> None:
        self.profile_id = profile_id
        self.is_in_control = is_in_control
        self.x = x
        self.y = y
        self.profile = profile

    @property
    def position(self):
        return (self.x, self.y)

    @classmethod
    def fill_from_array(cls, profile_id: str, x: float, y: float, profile: np.ndarray, is_in_control: bool = True):
        return cls(profile_id, x, y, profile, is_in_control)

    def dump_to_json(self, path: str):
        location = os.path.join(path, "json_output")
        if not os.path.exists(location):
            os.makedirs(location)

        dictionary = self.__dict__
        for k in dictionary.keys():
            if type(dictionary[k]) == np.ndarray:
                print(f"{k} is array")
                dictionary[k] = dictionary[k].tolist()

        with open(os.path.join(location, self.profile_id + ".json"), "w") as f:
            json.dump(dictionary, f, indent=4)


class Structure:
    def __init__(self, profiles: list) -> None:
        self.profiles = profiles
