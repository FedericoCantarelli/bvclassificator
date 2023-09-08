import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math


# Global parameters
DIMENSION = 128  #  Dimension of the grid
STRIDE = 21  # Stride of the laser


def laser_position():
    k = 1
    l = []
    v_dir = []
    for i in range(DIMENSION):
        temp = []
        for j in range(DIMENSION):
            temp.append((i, j))

        if k % 2 == 1:
            for t in temp:
                l.append(t)

        else:
            for t in temp[::-1]:
                l.append(t)

        k += 1
    return np.array([e for e in l[::STRIDE]])


#  Create global array for the laser position
position = laser_position()


class Profile:
    def __init__(self, x: int, y: int, n_frames: int, time_period: float) -> None:
        self.x = x
        self.y = y
        self.time_frames = np.linspace(0, time_period, n_frames)
        self.label = None
        self.profile = None
        self.errors = None
        self.func_profile = None
        self.err_params = None

    def simulate(self, loc: float, scale: float, with_noise: bool, label: int, tau=None) -> None:
        """_summary_

        Args:
            sim_params (dict): A dictionary containing a int variable "label" and a "funciton" simulation function
            loc (float): Mean of error normal distribution
            scale (float): Standard deviation of error normal distribution
        """
        self.label = label
        self.func_profile = sim(self.time_frames, tau)
        if with_noise:
            self.errors = np.random.normal(
                loc, scale, self.time_frames.shape[0])
            self.profile = self.func_profile + self.errors
            self.err_params = (loc, scale)

        else:
            self.profile = self.func_profile

    def plot(self) -> None:
        gs1 = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs1[0, :-2])
        ax2 = plt.subplot(gs1[1, :-2])
        ax3 = plt.subplot(gs1[0, -2:])
        ax4 = plt.subplot(gs1[1, -2:])

        ax1.plot(self.time_frames, self.profile, marker="D",
                 markerfacecolor="white", markersize=4, markeredgewidth=1)
        ax1.set_xticks([])
        ax1.set_title("Temperature Profile")

        ax2.scatter(self.time_frames, self.errors, s=12)
        ax2.axhline(y=self.err_params[0], color="orange",
                    linestyle="--", linewidth=1)
        ax2.set_title(
            f"Errors from a N({self.err_params[0]},{self.err_params[1]})")
        ax3.plot(self.func_profile)
        ax3.set_xticks([])
        ax3.yaxis.tick_right()
        ax3.set_title("Profile Function")

        ax4.hist(self.errors, "auto", edgecolor="black",
                 alpha=0.8, orientation="horizontal", density=True)
        ax4.set_title("Error Distribution")
        ax4.set_yticks([])

        plt.tight_layout()
        plt.show()

    @property
    def index(self) -> int:
        return self.x + self.y * DIMENSION

    @property
    def coordinates(self) -> tuple:
        return (self.x, self.y)


# def capacity(temp: float) -> float:
#     """Capacity for 17-4 PH. Heat capacity depends on material state.

#     Args:
#         temp (float): Temperature of the point

#     Returns:
#         float: Conductivity of the material.
#     """
#     if temp < 1404:
#         return 406.3 + 0.3055*temp
#     else:
#         return 834


# def conductivity(temp: float) -> float:
#     """Conductivity for 17-4 PH. The conductivity depends on material state.

#     Args:
#         temp (float): Temperature of the point

#     Returns:
#         float: Conductivity of the material.
#     """
#     if temp < 1404:
#         return 10.9385 + 0.01461*temp
#     else:
#         return 31.4


# def density(temp: float) -> float:
#     """Density for 17-4 PH. The density depends on material state.

#     Args:
#         temp (float): Temperature of the point

#     Returns:
#         float: Density of the material.
#     """
#     if temp < 1404:
#         return 7226.76
#     else:
#         return 7805.08 + 0.412*temp


# def compute_thermal_diffusion(temp: float) -> float:
#     cond = conductivity(temp)
#     cap = capacity(temp)
#     dens = density(temp)

#     return cond/(cap*dens)


# def in_control(arr: np.ndarray, x: int, y: int) -> np.ndarray:

#     result = np.zeros_like(arr)

#     P = 220         # Heat power W
#     v = 755.5       # Scan speed mm/s
#     T0 = 25         # Room temperature °C
#     alpha = 0.52    # Adjust the shape of the melting pool
#     z = 0           # Distance of the interested point from heat source.

#     for i, e in enumerate(arr):
#         r = math.sqrt((position[i][0] - x)**2 + (position[i][1]-y)**2)
#         azimuth = compute_azimuth(position[i], (x, y))

#         if i == 0:
#             k = compute_thermal_diffusion(T0)
#             cond = conductivity(T0)

#         else:
#             k = compute_thermal_diffusion(result[i-1])
#             cond  = conductivity(result[i-1])

#         coef = (alpha * P)/(2*math.pi*cond*math.sqrt(r**2+(alpha*z)**2))
#         exponent = -v * (math.sqrt(r**2+(alpha*z) **
#                          2+r*math.cos(azimuth)))/(2*k)
#         exp_res = math.exp(exponent)

#         result[i] = coef*exp_res+T0

#     return result


# def compute_azimuth(start_point, end_point):
#     x1, y1 = start_point
#     x2, y2 = end_point

#     # Calcola la differenza tra le coordinate x e y
#     delta_x = x2 - x1
#     delta_y = y2 - y1

#     # Calcola l'azimuth in radianti
#     azimuth_rad = math.atan2(delta_y, delta_x)
#     return azimuth_rad


def sim(arr: np.ndarray, tau):
    if tau is None:
        tau = 0
    return 255/(1+np.exp(0.2*(arr-0.95*tau)))


if __name__ == "__main__":
    c = Profile(5, 5, 60, 300)
    c.simulate(0, 1, True, 1, 50)
    c.plot()
