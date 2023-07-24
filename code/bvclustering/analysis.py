import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

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

    @classmethod
    def fill_from_json(cls, path: str):
        with open(path, "r") as f:
            dictionary = json.load(f)

        return cls(dictionary["profile_id"],
                   dictionary["x"],
                   dictionary["y"],
                   np.ndarray(dictionary["profile"]),
                   dictionary["is_in_control"])


class Structure:
    def __init__(self, profiles: list) -> None:
        self.profiles = profiles
        self.coordinates = dict(id=[],
                                x=[],
                                y=[],
                                t=[],
                                is_in_control=[])

        for i, each_profile in enumerate(profiles):
            self.coordinates["id"].append(each_profile.profile_id)
            self.coordinates["x"].append(each_profile.x)
            self.coordinates["y"].append(each_profile.y)
            self.coordinates["t"].append(each_profile.position)
            self.coordinates["is_in_control"].append(each_profile.is_in_control)
        
        self.df_coordinates = pd.DataFrame.from_dict(self.coordinates)
    
    def print_df(self):
        print(self.df_coordinates)


def main():
    a = Profile(profile_id="a", 
                x=0, 
                y=0, 
                profile=np.arange(0, 60, 1800),
                is_in_control=True)
    
    b = Profile(profile_id="b", 
                x=1, 
                y=0, 
                profile=np.arange(0, 60, 1800),
                is_in_control=False)
    
    c = Profile(profile_id="c", 
                x=0, 
                y=1, 
                profile=np.arange(0, 60, 1800),
                is_in_control=True)

    strc = Structure([a, b, c])
    strc.print_df()
    


if __name__ =="__main__":
    main()
