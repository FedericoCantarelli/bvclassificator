#########################################################################################
# Lattice simulation package - V 1.0                                                    #
# Package to analyze a lattice structure printed in metal using PBF process with the   #
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

import os
from PIL import Image
import matplotlib.pyplot as plt
import re
import shutil
import numpy as np
import support_functions as utils


class Cell:
    def __init__(self, cell_id: str, position: tuple, layers: np.ndarray = None, is_in_control: bool = True, is_nominal: bool = True) -> None:
        self.cell_id = cell_id
        self.layers = layers
        self.nominal = is_nominal
        self.in_control = is_in_control

        # Tuple, position of the cell in the space in format (x,y,z)
        self.position = position

        # Dictionary with all the transformations made on the cell
        self.transformations = dict(add_in_control_noise=None, add_missing_material=None,
                                    add_extra_material=None, add_systematic_error=None)

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

        return cls("1", (0, 0, 0), real)


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
        for f in self.real:
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
        tm_e = np.zeros(shape=nom.layers.shape)
        tm_l = np.zeros(shape=nom.layers.shape)

        # Compute truth matrix
        for i in range(nom.layers.shape[0]):
            mask_1 = nom.layers[i] == 0
            mask_2 = self.layers[i] == 1
            tm_l[i] = mask_1 & mask_2

            mask_1 = nom.layers[i] == 1
            mask_2 = self.layers[i] == 0
            tm_e[i] = mask_1 & mask_2

        return (tm_e, tm_l)


    def find_hausdorff_maps(self, nom) -> tuple:

        d1 = np.zeros_like(self.layers)
        d2 = np.zeros_like(self.layers)

        for r in range(self.layers.shape[0]):
            real_0_indexes = np.where(self.layers[r] == 0)
            nominal_0_indexes = np.where(nom.layers[r] == 0)

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
    

    def simulate_real(self):
        self.layers[10:20, 10:20, 10:20] = 1
        self.nominal = False
        self.in_control = False


    def save_to_cmap(self, path: str, which: str, cmap_string: str = "plasma", milliseconds: int = 50):
        """
        Save deviation map to .gif using a cmap.

        Args:
            path (str): Path to the cell folder
            which (str): One of <'d1', 'd2', 'tme', 'tml'>. Array to be saved
            cmap_string (str, optional): Matplotlib cmap. Defaults to 'plasma'.
            milliseconds (int, optional): Duration of each frame of the gif file. Defaults to 50.

        """

        assert which in ["d1", "d2", "tme", "tml"], "Error: which must be one of <'d1', 'd2', 'tme', 'tml'>."

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
        location = os.path.join(path, self.cell_name)

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