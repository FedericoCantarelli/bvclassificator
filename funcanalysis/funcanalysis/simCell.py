#########################################################################################
# Lattice simulation package - V 1.0                                                    #
# Package to simulate a lattice structure printed in metal using PBF process with the   #
# possibility of simulating different printing errors.                                  #
#                                                                                       #
# This package was made to simulate data for a new approach of QC of functional data    #
#                                                                                       #
# This is my MSc thesis project                                                         #
# A.Y. 2022/2023                                                                        #
#                                                                                       #
# Copyright by Federico Cantarelli                                                      #
# Maintained by Federico Cantarelli                                                     #
# Reach me if you have suggestion of if you feel lonely: fede.cantarelli98@gmail.com    #
#                                                                                       #
# Dreamed, designed, implemented and ran in 2023 in ["Casalmaggiore", "Milano"]         #
#                                                                                       #
# Project end date: ??/??/??                                                            #
#########################################################################################

import numpy as np
import pandas as pd
from PIL import Image
import os
import random
import json
import matplotlib.pyplot as plt
import re
import shutil
import statistics
import support_functions as utils


class Cell:
    # Class for the single cell with the method to run the simulation and add printing errors
    def __init__(self, cell_id: str, frames: np.ndarray, nominal: bool = True, position: tuple = (0, 0, 0)):

        # True if the cell is in control
        self.in_control = True

        # True if the cell is nominal. Set to false once add_in_control_noise is called
        self.nominal = nominal

        # ID cell
        self.cell_id = cell_id

        # Cell frames
        self.frames = frames

        # Tuple, position of the cell in the space in format (x,y,z)
        self.position = position

        # Dictionary with all the transformations made on the cell
        self.transformations = dict(add_in_control_noise=None, add_missing_material=None, add_extra_material=None,
                                    add_systematic_error=None)

        # Deviation maps
        self.tm_e = None
        self.tm_l = None
        self.d1 = None
        self.d2 = None

    @property
    def cell_name(self):
        if self.nominal:
            return self.cell_id + "_nominal"

        if self.in_control:
            return self.cell_id + "_in_control"

        return self.cell_id + "_out_of_control"

    @classmethod
    def build_from_pngs(cls, path_real: str):
        """Function to build a cell from pngs slices. Png slices must be in the format "1.png".

        Args:
            path_real (str): Path to folder of real png slices

        Returns:
            Cell: return the new built cell 
        """

        real_pngs_list = [j for j in os.listdir(path_real) if ".png" in j]

        real_pngs_list.sort(key=lambda x: int(x.split(".")[0]))

        image = Image.open(os.path.join(path_real, real_pngs_list[0]))
        image = np.array(image)

        real = np.zeros(shape=(len(real_pngs_list),
                        image.shape[0], image.shape[1]))

        for i, f in enumerate(real_pngs_list):
            png_location = os.path.join(path_real, f)
            image = Image.open(png_location)
            image = image.convert("L")
            image = np.array(image) / 255
            real[i] = image.astype(np.uint8)

        return cls("1", real)

    def save_in_gif(self, path: str, milliseconds: int = 50) -> None:

        # Create empty list for GIF's frames
        frames_gif = []

        # Build path
        location = os.path.join(path, self.cell_name)

        # Check if directory exists of the cell exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        # Build path of gif output folder
        output_location = os.path.join(location, "gif")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Convert each numpy.ndarray into a PIL object
        for f in self.layers:
            img = Image.fromarray((f * 255).astype(np.uint8), mode='L')
            frames_gif.append(img)

        # Create file name
        name = self.cell_id + ".gif"

        # Save as GIF
        frames_gif[0].save(os.path.join(output_location, name), save_all=True, append_images=frames_gif[1:], loop=0,
                           duration=milliseconds)

        # Free memory
        del frames_gif
        return None

    def get_deviation_maps(self, nom, which: str = "both") -> None:

        assert which in ['hausdorff', 'truth',
                         'both'], "Error: invalid which parameter."

        if which == 'hausdorff':
            d = self.find_hausdorff_maps(nom)
            self.d1 = d[0]
            self.d2 = d[1]

        elif which == 'truth':
            tm = self.find_tm_maps(nom)
            self.tm_e = tm[0]
            self.tm_l = tm[1]

        elif which == 'both':
            d = self.find_hausdorff_maps(nom)
            tm = self.find_tm_maps(nom)

            self.d1 = d[0]
            self.d2 = d[1]
            self.tm_e = tm[0]
            self.tm_l = tm[1]

        return None

    def add_in_control_noise(self, entity_missing: float = 0.0, entity_extra: float = 0.0, kind: str = "miss") -> None:
        """
        :param entity_missing: Number between 0 and 1. It is the probability a pixel will be selected as random noise.
        It is a repeated and indipendent toss of a coin.
        :param entity_extra: Number between 0 and 1. It is the probability a pixel on the countour will be selected
        as origin for extra material. It is a repeated and indipendent toss of a coin.
        :param kind: If "miss" there will be just random missing pixels. If "both", on the border of sintered metal
        there will be some random extra material.
        :return: None

        Function to simulate the in-control noise. The entity can be seen also as 3d printer reliability.
        The "both" option is a call to function add_extra_material(entity=entity_extra, max_length=1, severity=0, options="all").
        """
        assert kind == "miss" or kind == "both", "Kind must be one of <'miss', 'both'>"

        if kind == "miss":
            assert 0 < entity_missing < 1, "Entity must be a value between ]0; 1[."
            assert entity_extra == 0.0, "If kind is 'miss', entity_extra must be zero."

        if kind == "both":
            assert 0 < entity_extra < 1 and 0 < entity_missing < 1, "Entities must be values between ]0; 1[."

        for i in range(self.n_frames * 2):
            # Generate a probability matrix
            probs = np.random.uniform(
                0, 1, size=(self.dimension, self.dimension))
            mask = probs >= 1 - entity_missing
            self.frames[i][mask] = 1

        # Add extra single pixels on the border of the struct
        if kind == "both":
            self.add_extra_material(
                entity=entity_extra, max_length=1, severity=0, kind="all")

        self.nominal = False
        self.in_control = True
        self.transformations["add_in_control_noise"] = dict(kind=kind,
                                                            entity_missing=entity_missing,
                                                            entity_extra=entity_extra)

        self.transformations["add_extra_material"] = None

        return None

    def add_missing_material(self, entity, severity_x, severity_y, kind: str = "some", happening: int = 0.1) -> None:
        """
        :param entity: How many centroids for defect are choosen.
        :param severity_x: Semi severity of the defect along axis 0.
        :param severity_y: Semi sverity of the defect along axis 1.
        :param options: "all" if you want all the frames to be with material missing, "some" if only some frames.
        :param happening: Probability of a frame to be defective.

        Function to simulate a missing material defect. If severity_x = 2 and severity_y=6, the defect size will be
        a 12x4 defect.
        """

        assert kind == "some" or kind == "all", 'Options must be one of < "some", "all">'

        # If options is "all" all frame will be defective
        if kind == "all":
            happening = 1

        for i in range(self.n_frames * 2):
            # Check if the defect happened
            happened = np.random.uniform(0, 1) >= 1 - happening

            # If happened, remove material from the frame
            if happened:
                # Select possible indexes as centroids of the defect
                coordinates = [t for t in zip(np.where(self.frames[i] == 0)[
                    0], np.where(self.frames[i] == 0)[1])]
                selected = random.sample(
                    coordinates, int(entity * len(coordinates)))

                # Remove material
                for points in selected:
                    self.frames[i][points[0] - severity_y:points[0] + severity_y,
                                   points[1] - severity_x:points[1] + severity_x] = 1

        self.in_control = False
        self.transformations["add_missing_material"] = dict(kind=kind,
                                                            happening=happening,
                                                            entity=entity,
                                                            severity_x=severity_x,
                                                            severity_y=severity_y
                                                            )

        self.nominal = False

        return None

    def add_systematic_error(self, mask: np.ndarray, kind: str, indexes: list = None, option: str = "less", happening=0.1) -> None:
        """
        :param mask: Pixel mask to be selected. Pixel selected must have value 1 and not selected value 0.
        :param kind: "some" o "all". If "some" just some frames will be selected to be defective according to happening.
        :param option: "less" o "more" if the error is extra material or missing material.
        :param happening: probability of happening when "some" is selected.
        Method to apply a systematic error to the cell. Systematic in the sense that defect position is given a priori and it is not randomly selected.
        This method is crucial to simulate 3d printer mis-calibration.
        """

        assert option == "less" or option == "more", "Error: option must be one of <'less', 'more'>."
        assert kind == "all" or kind == "some" or kind == "precise", "Error kind must be one of <'all', 'some', 'precise'>."
        if kind == "precise":
            assert indexes, "If kind is 'precise' you must specify an int indexes list for the systematic error."
        assert mask.shape == (self.dimension, self.dimension), \
            f"Error, mask.shape must be equal to {(self.dimension, self.dimension)}"

        # If kind is "more" then add material, if not then remove
        pixel = 0 if option == "more" else 1

        if kind != "precise":
            # If options is "all" all frame will be defective
            if kind == "all":
                happening = 1

            # For each frame check if the defect happened and
            for i in range(self.n_frames * 2):
                happened = np.random.uniform(
                    low=0.0, high=1.0) >= 1 - happening
                if happened:
                    self.frames[i][(mask == 1)] = pixel

        else:
            happening = None
            for i in range(self.n_frames * 2):
                if i in indexes:
                    self.frames[i][(mask == 1)] = pixel

        self.in_control = False

        self.transformations["add_systematic_error"] = dict(kind=kind,
                                                            index=indexes,
                                                            happening=happening,
                                                            option=option,
                                                            mask_hash=str(
                                                                utils.get_array_hash(mask))
                                                            )

        self.nominal = False

        return None

    def add_extra_material(self, entity, severity, max_length, kind: str = "some", happening: int = 0.1) -> None:
        """
        :param entity: entità del difetto (quanti pixel sul totale vengono selezionati)
        :param severity: spessore dell'asticella
        :param max_length: massima lunghezza dell'asticella
        :param options: "some" o "all"
        :param happening: probabilità di riscontrare un difetto
        :return: None
        """

        final = []
        final_reverse = []

        if kind == "all":
            happening = 1

        for i in range(self.n_frames):
            temp = []
            temp_2 = []

            # Retrieve coordinates divided into upper, lower, right and left (to make it easier to apply the defect)
            # All upper coordinate
            # Upper part of left and right struct
            x_up = []
            y_up = []

            x_temp = [self.dimension // 2 -
                      self.semis for _ in range(self.thick * 2)]
            y_temp = [max(self.dimension // 2 - i - self.thick + j, self.border) for j in range(self.thick)] + [
                min(self.dimension // 2 + i + j + 1, self.dimension - self.border - 1) for j in range(self.thick)]

            x_up = x_up + x_temp
            y_up = y_up + y_temp

            # Upper part of upper and lower struct
            y_temp = [self.dimension // 2 - self.semis + j for j in range(self.semis * 2 + 1)] + [
                self.dimension // 2 - self.semis + j for j in range(self.semis * 2 + 1)]
            x_temp = [max(self.dimension // 2 - i - self.thick, self.border) for _ in range(
                self.semis * 2 + 1)] + [self.dimension // 2 + i for _ in range(self.semis * 2 + 1)]

            y_up = y_up + y_temp
            x_up = x_up + x_temp

            coordinates_up = [(w, z) for w, z in zip(x_up, y_up)]
            coordinates_up_single = []

            # Drop duplicated coordinates
            for element in coordinates_up:
                if element not in coordinates_up_single:
                    coordinates_up_single.append(element)

            # All right coordinates
            x_r = []
            y_r = []

            # Right coordinates of right and left structs
            x_temp = [self.dimension // 2 - self.semis +
                      j for j in range(self.semis * 2 + 1)] * 2
            y_temp = [min(self.dimension // 2 + i + self.thick, self.dimension - self.border)
                      for _ in range(self.semis * 2 + 1)] + [self.dimension // 2 - i for _ in range(self.semis * 2 + 1)]
            y_r = y_r + y_temp
            x_r = x_r + x_temp

            # Right coordinates of upper and lower structs
            x_temp = [max(self.dimension // 2 - i - self.thick + j, self.border) for j in range(self.thick)] + [
                min(self.dimension // 2 + i + j + 1, self.dimension - self.border - 1) for j in range(self.thick)]
            y_temp = [self.dimension // 2 +
                      self.semis for _ in range(self.thick * 2)]

            y_r = y_r + y_temp
            x_r = x_r + x_temp

            coordinates_r = set([(w, z) for w, z in zip(x_r, y_r)])
            coordinates_r_single = []

            # Drop duplicated coordinates
            for element in coordinates_r:
                if element not in coordinates_r_single:
                    coordinates_r_single.append(element)

            # All left coordinates
            y_l = []
            x_l = []

            # Left coordinates of upper and lower structure
            x_temp = [max(self.dimension // 2 - i - self.thick + j, self.border) for j in range(self.thick)] + [
                min(self.dimension // 2 + i + j + 1, self.dimension - self.border - 1) for j in range(self.thick)]
            y_temp = [self.dimension // 2 -
                      self.semis for _ in range(self.thick)] * 2
            x_l = x_l + x_temp
            y_l = y_l + y_temp

            # Left coordinates of right and left structs
            x_temp = [self.dimension // 2 - self.semis +
                      j for j in range(self.semis * 2 + 1)] * 2
            y_temp = [max(self.border, self.dimension // 2 - i - self.thick) for _ in range(
                self.semis * 2 + 1)] + [self.dimension // 2 + i for _ in range(self.semis * 2 + 1)]
            y_l = y_l + y_temp
            x_l = x_l + x_temp

            coordinates_l = [(w, z) for w, z in zip(x_l, y_l)]
            coordinates_l_single = []

            # Drop duplicated coordinates
            for element in coordinates_l:
                if element not in coordinates_l_single:
                    coordinates_l_single.append(element)

            # All lower coordinates

            # Lower of left and right structs
            x_down = []
            y_down = []

            y_temp = [max(self.border, self.dimension // 2 - i - self.thick + j) for j in range(self.thick)] + [
                min(self.dimension - self.border, self.dimension // 2 + i + j + 1) for j in range(self.thick)]
            x_temp = [self.dimension // 2 +
                      self.semis for _ in range(self.thick)] * 2

            x_down = x_down + x_temp
            y_down = y_down + y_temp

            # Lower of upper and lower structs
            y_temp = [self.dimension // 2 - self.semis + j for j in range(self.semis * 2 + 1)] + [
                self.dimension // 2 - self.semis + j for j in range(self.semis * 2 + 1)]
            x_temp = [self.dimension // 2 - i for _ in range(self.semis * 2 + 1)] + [min(
                self.dimension - self.border, self.dimension // 2 + i + self.thick) for _ in range(self.semis * 2 + 1)]

            x_down = x_down + x_temp
            y_down = y_down + y_temp

            coordinates_down = [(w, z) for w, z in zip(x_down, y_down)]
            coordinates_down_single = []

            # Drop duplicates
            for element in coordinates_down:
                if element not in coordinates_down_single:
                    coordinates_down_single.append(element)

            # Create coordinates dictionary to perform random sample
            coordinates = dict(zip(
                coordinates_up_single +
                coordinates_down_single +
                coordinates_l_single +
                coordinates_r_single,
                ["up" for _ in range(len(coordinates_up_single))] +
                ["do" for _ in range(len(coordinates_down_single))] +
                ["le" for _ in range(len(coordinates_r_single))] +
                ["ri" for _ in range(len(coordinates_l_single))]
            ))

            seed = os.urandom(4)
            random.seed(seed)
            happened = np.random.uniform(low=0.0, high=1.0) >= 1 - happening

            if happened:
                keys_list = list(coordinates.keys())
                selected_indexes = random.sample(
                    keys_list, int(len(keys_list) * entity))
                for i in selected_indexes:
                    temp_1 = dict()
                    temp_1[i] = coordinates[i]
                    temp.append(temp_1)

                final.append(temp)

            else:
                final.append(None)

            # seed = os.urandom(4)
            # random.seed(seed)
            happened = np.random.uniform(low=0.0, high=1.0) >= 1 - happening
            if happened:
                keys_list = list(coordinates.keys())
                selected_indexes = random.sample(
                    keys_list, int(len(keys_list) * entity))
                for i in selected_indexes:
                    temp_1 = dict()
                    temp_1[i] = coordinates[i]
                    temp_2.append(temp_1)
                final_reverse.append(temp_2)
            else:
                final_reverse.append(None)

        result = final + final_reverse[::-1]

        # a questo punto avrò una lista di liste di dizionari e in ogni dizionario c'è la chiave e la posizione
        for i, l in enumerate(result):

            if not l:
                pass

            else:
                for d in l:
                    seed = os.urandom(4)
                    random.seed(seed)
                    lun = random.randint(1, max_length)
                    point = list(d.keys())[0]

                    if d[point] == "up":
                        self.frames[i][max(point[0] - lun, 0):point[0] + 1, max(
                            point[1] - severity, 0):min(point[1] + severity + 1, self.dimension)] = 0

                    elif d[point] == "do":
                        self.frames[i][point[0]:min(self.dimension, point[0] + lun + 1), max(
                            point[1] - severity, 0):min(point[1] + severity + 1, self.dimension)] = 0

                    elif d[point] == "ri":
                        self.frames[i][max(point[0] - severity, 0): min(point[0] + severity + 1,
                                                                        self.dimension),
                                       point[1]:min(point[1] + lun + 1, self.dimension)] = 0

                    else:
                        self.frames[i][max(point[0] - severity, 0):min(
                            point[0] + severity + 1, self.dimension), max(point[1] - lun, 0):point[1] + 1] = 0

        # Set in_control attribute to False
        self.in_control = False

        self.transformations["add_extra_material"] = dict(kind=kind,
                                                          happening=happening,
                                                          entity=entity,
                                                          severity=severity,
                                                          max_length=max_length
                                                          )

        self.nominal = False
        return None

    def save_in_gif(self, path: str, milliseconds: int = 50) -> None:
        """
        :param path: Path where to save the cell. Just the path, the name of the output will be IDcell.gif
        :param milliseconds: -> Duration of each frame o the gif. Default 50 ms.
        """

        # If the cell is nominal, change the cell id
        if self.nominal and "nominal" not in self.cell_id:
            self.cell_id = self.cell_id + "_nominal"

        # Create empty list for GIF's frames
        frames_gif = []

        # Build path
        location = os.path.join(path, self.cell_id)

        # Check if directory exists of the cell exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        # Build path of gif output folder
        output_location = os.path.join(location, "gif")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Convert each numpy.ndarray into a PIL object
        for f in self.frames:
            img = Image.fromarray((f * 255).astype(np.uint8), mode='L')
            frames_gif.append(img)

        # Create file name
        name = self.cell_id + ".gif"

        # Save as GIF
        frames_gif[0].save(os.path.join(output_location, name), save_all=True, append_images=frames_gif[1:], loop=0,
                           duration=milliseconds)

        # Free memory
        del frames_gif
        return None

    def save_to_png(self, path: str) -> None:
        """
        :param path: Path where to save the cell. Just the path, the name of the output will be IDcell_t.png inside a directory called IDcell.
        Function to create a sub-directory in path called cell_IDcell to store all the frames of the cell in png format
        """

        # If the cell is nominal, change the cell id
        if self.nominal and "nominal" not in self.cell_id:
            self.cell_id = self.cell_id + "_nominal"

        # Build path
        location = os.path.join(path, self.cell_id)

        # Check if directory exists of the cell exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        # Build path of gif output folder
        output_location = os.path.join(location, "pngs")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Save each frame as a separate .png file
        for i in range(self.frames.shape[0]):
            img = Image.fromarray(
                (self.frames[i] * 255).astype(np.uint8), mode='L')
            name = self.cell_id + "_" + str(i+1) + ".png"
            img.save(os.path.join(output_location, name), "PNG")

        return None

    def save(self, path: str, save_as: list = ["bin", "png", "log", "gif"]) -> None:
        """
        :param path: String, path where to store the file
        :return: None
        File to save a cell in pre defined formats.
        """
        assert type(
            save_as) == list, "Error, save_as must be a list like object."
        assert os.path.exists(
            path), "Error path: file or directory does not exist"
        assert all(j in ["bin", "png", "log", "gif"]
                   for j in save_as), "Error, check save_as_list."

        if "bin" in save_as:
            self.save_to_bin(path)

        if "png" in save_as:
            self.save_to_png(path)

        if "gif" in save_as:
            self.save_in_gif(path)

        if "log" in save_as:
            self.save_log(path)

    def save_to_bin(self, path: str) -> None:
        """
        :param path: Path where to save the struct. Just the path, the name of the output will be IDstruct_t.png inside a directory called struct_IDstruct.
        Function to create a sub-directory in path called cell_IDcell to store the cell in npy object.
        """

        # If the cell is nominal, change the cell id
        if self.nominal and "nominal" not in self.cell_id:
            self.cell_id = self.cell_id + "_nominal"

        # Build path
        location = os.path.join(path, self.cell_id)

        # Check if directory exists of the cell exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        # Build path of gif output folder
        output_location = os.path.join(location, "bin")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Create filename
        name = self.cell_id + ".npy"

        np.save(os.path.join(output_location, name), self.frames)
        return None

    def save_log(self, path) -> None:
        """
        :param path: String to the path where to store the json params file
        :return: None 
        """

        if self.nominal and "nominal" not in self.cell_id:
            self.cell_id = self.cell_id + "_nominal"

        # Build path
        location = os.path.join(path, self.cell_id)

        # Check if directory exists of the cell exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        # Build path of gif output folder
        output_location = os.path.join(location, "log")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Create filename
        name = self.cell_id + "_params.json"

        # Create dictionary with all the params
        dictionary = dict(cell_params=dict(
            cell_id=self.cell_id,
            in_control=self.in_control,
            nominal=self.nominal,
            x=self.position[0],
            y=self.position[1],
            z=self.position[2],
            frames=self.frames.shape[0],
            semis=self.semis,
            border=self.border,
            dimension=self.dimension,
            thick=self.thick
        ), in_control_noise=self.transformations["add_in_control_noise"],
            missing_material=self.transformations["add_missing_material"],
            extra_material=self.transformations["add_extra_material"],
            systematic_error=self.transformations["add_systematic_error"]
        )

        # dump the dictionary to the desired json file
        with open(os.path.join(output_location, name), "w") as file:
            json.dump(dictionary, file, indent=4)

        return None

    def get_deviation_maps(self, nom, which: str = "both") -> None:
        """
        Function to compute deviation map, both Hausdorff distance based and truth matrix based.

        Args:
            nom (Cell): Nominal Cell object.
            which (str, optional): One of <'haudorff', 'truth', 'both'>. Defaults to "both".

        """

        assert which in ['haudorff', 'truth',
                         'both'], "Error: invalid which parameter."

        if which == 'hausdorff':
            d = self.find_hausdorff_maps(nom)
            self.d1 = d[0]
            self.d2 = d[1]

        elif which == 'truth':
            tm = self.find_tm_maps(nom)
            self.tm_e = tm[0]
            self.tm_l = tm[1]

        elif which == 'both':
            d = self.find_hausdorff_maps(nom)
            tm = self.find_tm_maps(nom)

            self.d1 = d[0]
            self.d2 = d[1]
            self.tm_e = tm[0]
            self.tm_l = tm[1]

        return None

    def find_tm_maps(self, nom) -> tuple:
        """
        Function to find the truth matrix deviation map

        Args:
            nom (Cell): Nominal cell object

        Returns:
            tuple: Returns a tuple with tm_e and tm_l
        """

        assert nom.nominal and type(
            nom) == Cell, "Error, argument must be a nominal cell object."

        # Init arrays
        tm_e = np.zeros(shape=nom.frames.shape)
        tm_l = np.zeros(shape=nom.frames.shape)

        # Compute truth matrix
        for i in range(nom.frames.shape[0]):
            mask_1 = nom.frames[i] == 0
            mask_2 = self.frames[i] == 1
            tm_l[i] = mask_1 & mask_2

            mask_1 = nom.frames[i] == 1
            mask_2 = self.frames[i] == 0
            tm_e[i] = mask_1 & mask_2

        return (tm_e, tm_l)

    def find_hausdorff_maps(self, nom) -> tuple:
        """
        Function to find Hausdorff distance based matrix. Find the euclidean distance for each point from the closest point. 

        Args:
            nom (Cell): Nominal Cell object

        Returns:
            tuple: Return d1 and d2
        """

        d1 = np.zeros_like(self.frames)
        d2 = np.zeros_like(self.frames)

        for r in range(self.frames.shape[0]):
            real_0_indexes = np.where(self.frames[r] == 0)
            nominal_0_indexes = np.where(nom.frames[r] == 0)

            d = utils.euclidean_distance_matrix(
                real_0_indexes, nominal_0_indexes)
            min_distances = np.min(d, axis=1)
            for i, j, z in zip(real_0_indexes[0], real_0_indexes[1], min_distances):
                d1[r, i, j] = z

            d = utils.euclidean_distance_matrix(
                nominal_0_indexes, real_0_indexes)
            min_distances = np.min(d, axis=1)
            for i, j, z in zip(nominal_0_indexes[0], nominal_0_indexes[1], min_distances):
                d2[r, i, j] = z

        return (d1, d2)

    def build_from_bin(self, path: str) -> None:
        bin_location = os.path.join(path, "bin")
        bin_name = [j for j in os.listdir(bin_location) if ".npy" in j]

        log_location = os.path.join(path, "log")
        log_name = [j for j in os.listdir(log_location) if ".json" in j]

        assert len(bin_name) == 1, "Error, more than one npy file."
        assert len(log_name) == 1, "Error, more than one json file."

        json_location = os.path.join(path, log_name[0])
        self.load_json(json_location)

        bin_location = os.path.join(path, bin_name[0])
        self.frames = np.load(bin_location)

        return None

    def load_json(self, path: str) -> None:
        """
        :param path: path to json file
        Load json file into cell object.
        """
        with open(path) as f:
            json_file = json.load(f)

        params = json_file["cell_params"]

        self.cell_id = params["cell_id"]
        self.in_control = params["in_control"]
        self.nominal = params["nominal"]
        self.position = (params["x"], params["y"], params["z"])
        self.n_frames = params["frames"]
        self.semis = params["semis"]
        self.border = params["border"]
        self.dimension = params["dimension"]
        self.thick = params["thick"]
        self.frames = np.zeros(
            shape=(self.n_frames, self.dimension, self.dimension))

        if json_file["in_control_noise"]:
            self.transformations["add_in_control_noise"] = json_file["in_control_noise"]

        if json_file["missing_material"]:
            self.transformations["add_missing_material"] = json_file["missing_material"]

        if json_file["extra_material"]:
            self.transformations["add_extra_material"] = json_file["extra_material"]

        if json_file["systematic_error"]:
            self.transformations["add_systematic_error"] = json_file["systematic_error"]

        return None

    def save_to_cmap(self, path: str, which: str, cmap_string: str = "plasma", milliseconds: int = 50):
        """
        Save deviation map to .gif using a cmap.

        Args:
            path (str): Path to the cell folder
            which (str): One of <'d1', 'd2', 'tme', 'tml'>. Array to be saved
            cmap_string (str, optional): Matplotlib cmap. Defaults to 'plasma'.
            milliseconds (int, optional): Duration of each frame of the gif file. Defaults to 50.

        """

        assert which in [
            "d1", "d2", "tme", "tml"], "Error: which must be one of <'d1', 'd2', 'tme', 'tml'>."

        if which == "d1":
            assert self.d1 is not None, "Error: d1 not found."
            to_save = np.copy(self.d1)

        elif which == "d2":
            assert self.d2 is not None, "Error: d2 not found."
            to_save = np.copy(self.d2)

        elif which == "tme":
            assert self.tm_e is not None, "Error: tme not found."
            to_save = np.copy(self.tm_e)

        elif which == "tml":
            assert self.tm_l is not None, "Error: tml not found."
            to_save = np.copy(self.tm_l)

        # Build location
        location = os.path.join(path, self.cell_id)

        # Check if cell location exists
        if not os.path.exists(location):
            # If not, then create it
            os.makedirs(location)

        output_location = os.path.join(location, which + "_cmap")

        # Check if output folder exists
        if not os.path.exists(output_location):
            # If not, then create it
            os.makedirs(output_location)

        # Build temporary location
        temp_location = os.path.join(output_location, "temp")
        if not os.path.exists(temp_location):
            # If not, then create it
            os.makedirs(temp_location)

        for i, f in enumerate(to_save):
            name = str(i) + ".png"
            save_location = os.path.join(temp_location, name)
            plt.imsave(save_location, f, vmax=np.max(
                f), vmin=np.min(f), cmap=cmap_string)

        frames_gif = []
        for i in range(to_save.shape[0]):
            name = str(i) + ".png"
            saved_location = os.path.join(temp_location, name)
            img = Image.open(saved_location)
            frames_gif.append(img)

        name = which + "_cmap.gif"

        # Save as GIF
        frames_gif[0].save(os.path.join(output_location, name), save_all=True, append_images=frames_gif[1:], loop=0,
                           duration=milliseconds)

        shutil.rmtree(temp_location)
        return None

    def save_d1(self, path: str) -> None:
        """
        Save d1 Hausdorff distance based deviation map

        Args:
            path (str): Where to store the file

        """

        assert self.d1 is not None, "Error, d1 is None"

        location = os.path.join(path, "Hausdorff")

        if not os.path.exists(location):
            os.makedirs(location)

        output_location = os.path.join(location, "D1")

        if not os.path.exists(output_location):
            os.makedirs(output_location)

        np.save(output_location + "/d1.npy", self.d1)

        return None

    def save_d2(self, path: str) -> None:
        """
        Save d2 Hausdorff distance based deviation map

        Args:
            path (str): Where to store the file

        """

        assert self.d2 is not None, "Error, d2 is None"

        location = os.path.join(path, "Hausdorff")

        if not os.path.exists(location):
            os.makedirs(location)

        output_location = os.path.join(location, "D2")

        if not os.path.exists(output_location):
            os.makedirs(output_location)

        np.save(output_location + "/d2.npy", self.d2)

        return None

    def save_tme(self, path: str) -> None:
        """
        Save extra material deviation map computed using truth matrix.

        Args:
            path (str): Where to store the file

        """

        assert self.tm_e is not None, "Error, tme is None"

        location = os.path.join(path, "Truth")

        if not os.path.exists(location):
            os.makedirs(location)

        output_location = os.path.join(location, "TM_Extra")

        if not os.path.exists(output_location):
            os.makedirs(output_location)

        np.save(output_location + "/tme.npy", self.tm_e)

        return None

    def save_tml(self, path: str) -> None:
        """
        Save lack material deviation map computed using truth matrix.

        Args:
            path (str): Where to store the file

        """

        assert self.tm_l is not None, "Error, tme is None"

        location = os.path.join(path, "Truth")

        if not os.path.exists(location):
            os.makedirs(location)

        output_location = os.path.join(location, "TM_Lack")

        if not os.path.exists(output_location):
            os.makedirs(output_location)

        np.save(output_location + "/tml.npy", self.tm_l)

        return None

    def simulate_real(self):
        self.frames[10:20, 10:20, 10:20] = 1
        self.nominal = False
        self.in_control = False


class Structure:
    def __init__(self, structure_id: str, dimension: tuple):
        self.dim_x = dimension[0]
        self.dim_y = dimension[1]
        self.dim_z = dimension[2]
        self.structure_id = structure_id
        self.total_defect = 0
        self.defect_indexes = []
        self.defect_locations = []
        self.nominal = False
        self.struct = None

    def run(self, cell_list):
        assert self.dim_x * self.dim_y * self.dim_z == len(cell_list), \
            f"Dimension {(self.dim_x, self.dim_y, self.dim_z)}, does not match cell list length {len(cell_list)}."
        rows = []
        layers = []

        for i in range(self.dim_x * self.dim_y * self.dim_z):
            if not cell_list[i].in_control:
                self.total_defect += 1
                self.defect_indexes.append(i)
                self.defect_locations.append(cell_list[i].position)

        for i in range(0, len(cell_list), self.dim_x):
            batch = cell_list[i:i + self.dim_x]
            temp = []
            for t in batch:
                temp.append(t.frames)

            row = np.concatenate(temp, axis=1)
            rows.append(row)

        for j in range(0, len(rows), self.dim_y):
            batch = rows[j:j + self.dim_y]
            lay = np.concatenate(batch, axis=2)
            layers.append(lay)

        self.struct = np.concatenate(layers, axis=0)

    def run_nominal(self, semis: int = 23, dimension: int = 513, border: int = 0, thick: int = 50):
        single_cell = Cell(cell_id="demo", semis=semis,
                           dimension=dimension, border=border, thick=thick)
        single_cell.run()
        self.struct = np.tile(single_cell.frames,
                              (self.dim_z, self.dim_x, self.dim_y))
        self.nominal = True

    def save_in_gif(self, path: str, milliseconds: int = 50) -> None:
        """
        :param path: Path where to save the structure. Just the path, the name of the output will be IDstructure.gif
        :param milliseconds: -> Duration of each frame o the gif. Default 50 ms.
        """

        # Create empty list for GIF's frames
        location = build_path(path, self)
        if not os.path.exists(location):
            os.mkdir(location)
        os.chdir(location)
        frames_gif = []

        # Convert each numpy.ndarray into a PIL object
        for level in self.struct:
            img = Image.fromarray((level * 255).astype(np.uint8), mode='L')
            frames_gif.append(img)

        # Save as GIF
        frames_gif[0].save(self.structure_id + ".gif", save_all=True, append_images=frames_gif[1:], loop=0,
                           duration=milliseconds)

        os.chdir("..")

        return None

    def save(self, path: str) -> None:
        """
        :param path: Path where to save the struct. Just the path, the name of the output will be IDstruct_t.png inside a directory called struct_IDstruct.
        Function to create a sub-directory in path called struct_IDstruct to store all the frames of the cell in png format.
        """

        # Create directory to save cell frames
        location = build_path(path, self)
        if not os.path.exists(location):
            os.mkdir(location)
        os.chdir(location)

        # Save each frame as a separate .png file
        for k in range(len(self.struct)):
            img = Image.fromarray(
                (self.struct[k] * 255).astype(np.uint8), mode='L')
            img.save(self.structure_id + "_" + str(k + 1) + ".png", "PNG")

        os.chdir("..")
        return None

    def save_log(self, path):
        location = build_path(path, self)
        if not os.path.exists(location):
            os.mkdir(location)
        os.chdir(location)

        dictionary = dict(structure_params=dict(
            structure_id=self.structure_id,
            dimension=[self.dim_x, self.dim_y, self.dim_z],
            nominal=self.nominal,
            in_control=True if self.total_defect == 0 else False,
            frames=self.struct.shape[0]
        ), defects_param=dict(total_defect=self.total_defect,
                              defect_locations=[[t[0], t[1], t[2]] for t in
                                                self.defect_locations] if self.total_defect != 0 else None,
                              defect_indexes=self.defect_indexes if self.total_defect != 0 else None)
        )

        with open(self.structure_id + "_params.json", "w") as file:
            json.dump(dictionary, file, indent=4)

        os.chdir("..")


def run_analytic(path: str, save_to: str) -> None:
    """
    Function to dump all analytics in json formats.

    Args:
        path (str): Parent directory of alla scenarios and severities.
        save_to (str): Directory where to store all json files.
    """

    scenarios_list = [j for j in os.listdir(
        path) if os.path.isdir(os.path.join(path, j))]

    for scenario in scenarios_list:
        print("Dumping scenario " + scenario)
        output_location = os.path.join(save_to, scenario + "_analytic")
        scenario_location = os.path.join(path, scenario)

        if not os.path.exists(output_location):
            os.mkdir(output_location)

        cell_list = [j for j in os.listdir(scenario_location) if os.path.isdir(
            os.path.join(scenario_location, j))]

        for cell in cell_list:
            source_location = os.path.join(scenario_location, cell)

            d1_path = os.path.join(source_location, "Hausdorff/D1/d1.npy")
            d2_path = os.path.join(source_location, "Hausdorff/D2/d2.npy")
            tme_path = os.path.join(source_location, "Truth/TM_Extra/tme.npy")
            tml_path = os.path.join(source_location, "Truth/TM_Lack/tml.npy")

            d1_bool = os.path.exists(d1_path)
            d2_bool = os.path.exists(d2_path)
            tme_bool = os.path.exists(tme_path)
            tml_bool = os.path.exists(tml_path)

            assert any([d1_bool, d2_bool, tme_bool, tml_bool]
                       ), "Error, no deviation map found."

            if d1_bool:
                d1 = np.load(d1_path)
            else:
                print("Warning, d1 not found.")

            if d2_bool:
                d2 = np.load(d2_path)
            else:
                print("Warning, d2 not found.")

            if tme_bool:
                tme = np.load(tme_path)
            else:
                print("Warning, tme not found.")

            if tml_bool:
                tml = np.load(tml_path)
            else:
                print("Warning, tml not found.")

            data = dict(
                cell_id="demo",
                x=0,
                y=0,
                z=0,
                #######
                # Da modificare structure_order#
                #####
                structure_order=6,
                in_control=False,
                scenario=scenario,
                d1_max=None,
                d2_max=None,
                d1_avg=None,
                d2_avg=None,
                tm_e_count=None,
                tm_l_count=None,
            )

            json_list = [j for j in os.listdir(
                os.path.join(source_location, "log")) if ".json" in j]

            assert len(json_list) == 1, "Error, with the json files"

            json_source_path = os.path.join(
                os.path.join(source_location, "log"), json_list[0])

            with open(json_source_path) as f:
                params = json.load(f)["cell_params"]

            data["cell_id"] = params["cell_id"]
            data["x"] = params["x"]
            data["y"] = params["y"]
            data["z"] = params["z"]
            data["in_control"] = params["in_control"]
            # data["structure_order"] = params["structure_order"]

            data["d1_max"] = list(utils.get_layer_max(d1))
            data["d2_max"] = list(utils.get_layer_max(d2))

            data["tm_e_count"] = list(utils.get_layer_count(tme))
            data["tm_l_count"] = list(utils.get_layer_count(tml))

            json_destination = os.path.join(
                output_location, data["cell_id"] + ".json")

            with open(json_destination, "w") as f:
                json.dump(data, f, indent=4)


def run_scenarios(path: str, save_to: str, save_as: list, n_cells: int, save_what: str) -> None:
    """
    Function to simulate scenarios starting from a json file. 

    Args:
        path (str): Path to json file for scenarios generation.
        save_to (str): Parent directory where to store all the scenarios
        save_as (list): List, choose how to save the simulation ['gif', 'png', 'log', 'bin']
        n_cells (int): Number of cells per scenario.
        save_what (str): String, choose what to save between ['data', 'all', 'hausdorff', 'truth']
    """

    with open(path) as f:
        file = json.load(f)
        all_of_scenarios = file["scenarios"]
        cell_specifications_dict = file["cell_specifications"]
        structure_specifications_dict = file["struct_specifications"]

    assert (structure_specifications_dict["order"] **
            3) == n_cells, "Error: check structure orders and n_cells."

    for i in range(len(save_as)):
        assert save_as[i] in ["gif", "png", "log",
                              "bin"], "Error, check save_as argument."

    assert save_what in ["data", "all", "hausdorff",
                         "truth"], "Error, check save_what argument."

    for k in all_of_scenarios:
        location = os.path.join(save_to, k)
        if not os.path.exists(location):
            os.mkdir(location)

        functions = all_of_scenarios[k].keys()

        if "systematic_error" in functions:
            params_dict = all_of_scenarios[k]["systematic_error"]
            array_temp = np.load(params_dict["path_to_npy_object"])

        # Itero sui scenari
        print("Begin " + k)

        for i in range(structure_specifications_dict["order"]):
            for j in range(structure_specifications_dict["order"]):
                for h in range(structure_specifications_dict["order"]):

                    c = Cell(cell_id=str(i*structure_specifications_dict["order"]**2 +
                                         j*structure_specifications_dict["order"] +
                                         h+1),
                             semis=cell_specifications_dict["semis"],
                             dimension=cell_specifications_dict["dimension"],
                             border=cell_specifications_dict["border"],
                             thick=cell_specifications_dict["thick"],
                             position=(i, h, j))

                    c_nom = Cell(cell_id=str(i*structure_specifications_dict["order"]**2 +
                                             j*structure_specifications_dict["order"] +
                                             h+1),
                                 semis=cell_specifications_dict["semis"],
                                 dimension=cell_specifications_dict["dimension"],
                                 border=cell_specifications_dict["border"],
                                 thick=cell_specifications_dict["thick"],
                                 position=(i, h, j))

                    c.run()
                    c_nom.run()

                    if "in_control_noise" in functions:
                        params_dict = all_of_scenarios[k]["in_control_noise"]
                        c.add_in_control_noise(entity_missing=params_dict["entity_missing"],
                                               entity_extra=params_dict["entity_extra"],
                                               kind=params_dict["kind"])

                    if "extra_material" in functions:
                        params_dict = all_of_scenarios[k]["extra_material"]
                        c.add_extra_material(entity=params_dict["entity"],
                                             severity=params_dict["severity"],
                                             max_length=params_dict["max_length"],
                                             happening=params_dict["happening"],
                                             kind=params_dict["kind"])

                    if "missing_material" in functions:
                        params_dict = all_of_scenarios[k]["missing_material"]
                        c.add_missing_material(entity=params_dict["entity"],
                                               severity_x=params_dict["severity_x"],
                                               severity_y=params_dict["severity_y"],
                                               happening=params_dict["happening"],
                                               kind=params_dict["kind"])

                    if "systematic_error" in functions:
                        params_dict = all_of_scenarios[k]["systematic_error"]
                        array_temp = np.load(params_dict["path_to_npy_object"])
                        c.add_systematic_error(mask=array_temp,
                                               kind=params_dict["kind"],
                                               happening=params_dict["happening"],
                                               indexes=params_dict["indexes"],
                                               option=params_dict["option"])

                    output_location = os.path.join(location, c.cell_id)

                    if save_what == "truth":
                        c.find_tm_maps(c_nom)
                        c.save_tme(output_location)
                        c.save_tml(output_location)

                    elif save_what == "hausdorff":
                        c.find_hausdorff_maps(c_nom)
                        c.save_d2(output_location)
                        c.save_d1(output_location)

                    elif save_what == "all":
                        c.get_deviation_maps(c_nom)
                        c.save_tme(output_location)
                        c.save_tml(output_location)
                        c.save_d2(output_location)
                        c.save_d1(output_location)

                    else:
                        pass

                    if "gif" in save_as:
                        c.save_in_gif(location)

                    if "png" in save_as:
                        c.save_to_png(location)

                    if "bin" in save_as:
                        c.save_to_bin(location)

                    if "log" in save_as:
                        c.save_log(location)

                    utils.progress_bar((i*structure_specifications_dict["order"]**2 +
                                        j*structure_specifications_dict["order"] +
                                        h+1)/structure_specifications_dict["order"]**3)

        print("\n")


def get_defective_cluster_dictionary(m: int, n: int, entity: int, order: int) -> dict:
    """
    Function to retrieve a dictionary with n runs of the "entity" closest centroid to m randomly selected centroids. 

    Args:
        m (int): Number of centroids.
        n (int): Number of runs for each centroid.
        entity (int): Entity of the defect area.
        order (int): Order of the lattice structure.

    Returns:
            dict: Return dictionary in the shape of {"n_run":[id(int), ...]}
    """

    coordinates_list = []
    distances_list = []
    runs_dict = dict()

    # Create a list with all the coordinates
    for x in range(order):
        for y in range(order):
            for z in range(order):
                coordinates_list.append((x, y, z))

    # Create a dataframe with distances
    for x in coordinates_list:
        for y in coordinates_list:
            distances_list.append((x, y, utils.distance_3d(x, y)))

    df = pd.DataFrame(distances_list, columns=['P1', 'P2', 'Distance'])

    # Run n time for n scenarios

    for i in range(n):
        final = []
        centroid_list = []
        count = 0
        for _ in range(m):
            while count < m:
                new_centroid = (random.randint(
                    0, order-1), random.randint(0, order-1), random.randint(0, order-1))
                if new_centroid not in centroid_list:
                    centroid_list.append(new_centroid)
                    count += 1

        for c in centroid_list:
            # Find nearest centroids. It's entity+1 because the first element will be c
            coor = df[df["P1"] == c].sort_values(
                by="Distance").head(entity+1)["P2"].values
            final.append([c, [utils.convert_centroid_to_id(j, order)
                         for j in coor]])

        runs_dict[str(i)] = final

    return runs_dict


def run_defective_structure_scenarios(analytic_path: str, in_control_path: str, m: int, n: int, entity: int, order: int) -> None:
    """_summary_

    Args:
        analytic_path (str): Path to analytics folder.
        in_control_path (str): Path to in control analytics folder. All the json files are inside this folder.
        m (int): Number of centroids.
        n (int): Number of runs for each centroid.
        entity (int): Entity of the defect area.
        order (int): Order of the lattice structure.
    """

    analytics_list = [j for j in os.listdir(analytic_path) if os.path.isdir(
        os.path.join(analytic_path, j)) and "in_control" not in j]
    in_control_list = [j for j in os.listdir(in_control_path) if ".json" in j]

    df_in_control = pd.DataFrame(columns=[
        "cell_id",
        "x",
        "y",
        "z",
        "centroid",
        "max_d1_max",
        "max_d2_max",
        "var_d1_max",
        "var_d2_max",
        "avg_tme",
        "avg_tml",
        "var_tme",
        "var_tml",
        "range_tme",
        "range_tml",
        "max_tme",
        "max_tml",
        "out_of_control"])

    for json_file in in_control_list:
        json_incontrol_path = os.path.join(in_control_path, json_file)
        with open(json_incontrol_path) as f:
            loaded_json_incontrol = json.load(f)

        df_temp = pd.DataFrame({
            "cell_id": loaded_json_incontrol["cell_id"],
            "x": loaded_json_incontrol["x"],
            "y": loaded_json_incontrol["y"],
            "z": loaded_json_incontrol["z"],
            "centroid": None,
            "max_d1_max": max(loaded_json_incontrol["d1_max"]),
            "max_d2_max": max(loaded_json_incontrol["d2_max"]),
            "var_d1_max": statistics.variance(loaded_json_incontrol["d1_max"]),
            "var_d2_max": statistics.variance(loaded_json_incontrol["d2_max"]),
            "avg_tme": statistics.mean(loaded_json_incontrol["tm_e_count"]),
            "avg_tml": statistics.mean(loaded_json_incontrol["tm_l_count"]),
            "var_tme": statistics.variance(loaded_json_incontrol["tm_e_count"]),
            "var_tml": statistics.variance(loaded_json_incontrol["tm_l_count"]),
            "range_tme": max(loaded_json_incontrol["tm_e_count"]) - min(loaded_json_incontrol["tm_e_count"]),
            "range_tml": max(loaded_json_incontrol["tm_l_count"]) - min(loaded_json_incontrol["tm_l_count"]),
            "max_tme": max(loaded_json_incontrol["tm_e_count"]),
            "max_tml": max(loaded_json_incontrol["tm_l_count"]),
            "out_of_control": 0
        })

        df_in_control = pd.concat(df_in_control, df_temp, axis=0)

    for folder in analytics_list:
        print("Begin with " + folder)
        analytics_scenario_path = os.path.join(analytic_path, folder)

        r_dict = get_defective_cluster_dictionary(m, n, entity, order)
        for key in r_dict.keys():
            df_final = df_in_control.copy()
            for centroid_realization in r_dict[key]:
                for id_defective in centroid_realization[1]:
                    defective_json_path = os.path.join(
                        analytics_scenario_path, str(id_defective) + ".json")
                    with open(defective_json_path) as f:
                        loaded_json_out_of_control = json.load(f)

                    df_final = df_final[df_final["cell_id"]
                                        != str(id_defective)]
                    df_temp = pd.DataFrame({
                        "cell_id": loaded_json_out_of_control["cell_id"],
                        "x": loaded_json_out_of_control["x"],
                        "y": loaded_json_out_of_control["y"],
                        "z": loaded_json_out_of_control["z"],
                        "centroid": str(centroid_realization[0]),
                        "max_d1_max": max(loaded_json_out_of_control["d1_max"]),
                        "max_d2_max": max(loaded_json_out_of_control["d2_max"]),
                        "var_d1_max": statistics.variance(loaded_json_out_of_control["d1_max"]),
                        "var_d2_max": statistics.variance(loaded_json_out_of_control["d2_max"]),
                        "avg_tme": statistics.mean(loaded_json_out_of_control["tm_e_count"]),
                        "avg_tml": statistics.mean(loaded_json_out_of_control["tm_l_count"]),
                        "var_tme": statistics.variance(loaded_json_out_of_control["tm_e_count"]),
                        "var_tml": statistics.variance(loaded_json_out_of_control["tm_l_count"]),
                        "range_tme": max(loaded_json_out_of_control["tm_e_count"]) - min(loaded_json_out_of_control["tm_e_count"]),
                        "range_tml": max(loaded_json_out_of_control["tm_l_count"]) - min(loaded_json_out_of_control["tm_l_count"]),
                        "max_tme": max(loaded_json_out_of_control["tm_e_count"]),
                        "max_tml": max(loaded_json_out_of_control["tm_l_count"]),
                        "out_of_control": 1
                    })

                    df_final = pd.concat(df_final, df_temp, axis=0)
            csv_folder_output_path = os.path.join(
                analytics_scenario_path, "structure_scenarios")
            if not os.path.exists(csv_folder_output_path):
                os.mkdir(csv_folder_output_path)

            csv_output_path = os.path.join(
                csv_folder_output_path, str(key) + ".csv")
            df_final.to_csv(csv_output_path, index_label=False)
        print("Done with " + folder)

    return None
