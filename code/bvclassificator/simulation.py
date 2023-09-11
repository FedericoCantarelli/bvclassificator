import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Profile:
    def __init__(self, x: int, y: int, time_period: float, fps: int) -> None:
        self.x = x
        self.y = y
        self.time_frames = np.linspace(0, time_period-1, time_period*fps)

        self.label = None
        self.profile = None
        self.errors = None
        self.func_profile = None
        self.err_params = None

    def simulate(self, loc: float, scale: float, with_noise: bool, label: int, in_control: bool, tau = None) -> None:
        """_summary_

        Args:
            sim_params (dict): A dictionary containing a int variable "label" and a "funciton" simulation function
            loc (float): Mean of error normal distribution
            scale (float): Standard deviation of error normal distribution
        """
        self.label = label
        if in_control:
            self.func_profile = sim(self.time_frames, 0, 0.95)

        else:
            self.func_profile = sim(self.time_frames, tau, 0.95)

        if with_noise:
            self.errors = np.random.normal(
                loc, scale, self.time_frames.shape[0])
            self.profile = self.func_profile + self.errors
            self.err_params = (loc, scale)

        else:
            self.profile = self.func_profile

    def plot(self) -> None:
        gs = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs[0, :-2])
        ax2 = plt.subplot(gs[1, :-2])
        ax3 = plt.subplot(gs[0, -2:])
        ax4 = plt.subplot(gs[1, -2:])

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

    def index(self, dim: int) -> int:
        return self.y + self.x * dim


# def linear(arr: np.ndarray):
#     return arr


# def quadratic(arr: np.ndarray):
#     return np.power(arr, 2)


def sim(arr: np.ndarray, tau: int, H: float):
    return 255/(1+np.exp(0.2*(arr - H*tau)))


if __name__ == "__main__":
    profile_list = []
    for i in range(3):
        for j in range(3):
            c = Profile(i, j, 60, 1)
            if c.index(3) in [0, 1, 3, 4]:
                c.simulate(0, 1, True, 1, in_control=False)
            else:
                c.simulate(0, 1, True, 0, in_control=True)
            # print(f"(x,y) = {(i,j)}")
            # print(f"Index = {c.index}")

            profile_list.append(c)

    profile_list[0].plot()
    profile_list[2].plot()
