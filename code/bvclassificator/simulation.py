import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIMENSION = 3


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

    def simulate(self, sim_params: dict, loc: float, scale: float, with_noise: True) -> None:
        """_summary_

        Args:
            sim_params (dict): A dictionary containing a int variable "label" and a "funciton" simulation function
            loc (float): Mean of error normal distribution
            scale (float): Standard deviation of error normal distribution
        """
        self.label = sim_params["label"]
        self.func_profile = sim_params["function"](self.time_frames)

        if with_noise:
            self.errors = np.random.normal(
                loc, scale, self.time_frames.shape[0])
            self.profile = self.func_profile + self.errors
            self.err_params = (loc, scale)

        else:
            self.profile = self.func_profile

    def plot(self) -> None:
        assert self.profile is not None and self.errors is not None and self.func_profile is not None, "Not enough data to plot a summary for the profile"
        gs1 = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs1[0, :-2])
        ax2 = plt.subplot(gs1[1, :-2])
        ax3 = plt.subplot(gs1[0, -2:])
        ax4 = plt.subplot(gs1[1, -2:])

        ax1.plot(self.profile)
        ax1.set_xticks([])
        ax1.set_title("Temperature Profile")

        ax2.scatter(self.time_frames, self.errors, s=12)
        ax2.axhline(y=self.err_params[0], color="orange",
                    linestyle="--", linewidth=1.5)
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




def conductivity(temp:float) -> float:
    """Conductivity for 17-4 PH. The conductivity depends on material state.

    Args:
        temp (float): Temperature of the previous layer

    Returns:
        float: Conductivity of the material.
    """
    if temp < 1404:
        return 10.9385 + 0.01461*temp
    else:
        return 31.4
    







def in_control(arr: np.ndarray) -> np.ndarray:
    P = 220    # Heat power W
    v = 755.5  # Scan speed mm/s
    T0 = 25    # Room temperature °C
    alpha = 0.52
    


if __name__ == "__main__":
    c1 = Profile(0, 2, 600, 1)
    c1.simulate({"label": 0,
                 "function": in_control}, 0, 0.001, with_noise=True)
    c1.plot()
