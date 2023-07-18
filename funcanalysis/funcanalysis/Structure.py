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
import numpy as np


class Cell:
    def __init__(self, cell_id: str, position: tuple, real: np.ndarray = None, nominal: bool = True) -> None:
        self.cell_id = cell_id
        self.real = real
        self.nominal = nominal

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
