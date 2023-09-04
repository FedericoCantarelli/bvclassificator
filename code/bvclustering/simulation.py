from .analysis import Profile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tabulate import tabulate


class Simulation(Profile):
    def __init__(self, profile_id: str, x: float, y: float, beta0: tuple, beta1: tuple, beta2_coeff: float, beta2_slope: float, error: tuple, deviation_entity: float = None,
                 time: float = 60, fps: int = 30) -> None:

        super().__init__(profile_id=profile_id,
                         x=x,
                         y=y,
                         time=time,
                         fps=fps)

        self.is_in_control = True if deviation_entity is None else False

        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2_coeff = beta2_coeff
        self.beta2_slope = beta2_slope
        self.error = error
        self.deviation_entity = deviation_entity

        self.time_evolution = np.arange(0, self.time, 1/self.fps)

        self.profile = np.zeros_like(self.time_evolution)

        self.b0_eff = 0.0
        self.b1_eff = 0.0
        self.b2_eff = 0.0

        self.in_control_profile = None

        self.profile_function = np.zeros_like(self.time_evolution)

    def run(self):
        e = np.random.normal(
            loc=self.error[0], scale=self.error[1], size=len(self.time_evolution))

        self.b0_eff = np.random.normal(loc=self.beta0[0],
                                       scale=self.beta0[1])

        self.b1_eff = np.random.normal(loc=self.beta1[0],
                                       scale=self.beta1[1])

        self.b2_eff = self.beta2_coeff + self.beta2_slope * self.b1_eff

        if self.deviation_entity is not None:
            self.in_control_profile = self.b0_eff + self.b1_eff * \
                self.time_evolution + self.b2_eff*self.time_evolution**2
            self.in_control_profile = np.exp(self.in_control_profile)
            self.b1_eff = self.b1_eff + self.deviation_entity * self.beta1[1]

        self.profile_function = self.b0_eff + self.b1_eff * \
            self.time_evolution + self.b2_eff*self.time_evolution**2

        self.profile = self.profile_function + e

        self.profile_function = np.exp(self.profile_function)
        self.profile = np.exp(self.profile)

    def plot(self):

        legend_elements = [Line2D([0], [0], color='#FF0000', label='Profile function'),
                           Line2D([0], [0], marker='o', color='w', label='Real profile', markerfacecolor='cornflowerblue')]

        fig, ax = plt.subplots()

        ax.plot(self.time_evolution, self.profile_function, c="#FF0000")
        ax.scatter(self.time_evolution, self.profile,
                   c="cornflowerblue", alpha=0.8)

        if self.in_control_profile is not None:
            ax.plot(self.time_evolution, self.in_control_profile, c="darkorange")
            legend_elements.append(
                Line2D([0], [0], color="darkorange", label="In control profile"))

        ax.set_title("Profile plot")
        ax.set_ylim(np.min(self.profile)*0.98, np.max(self.profile)*1.01)
        ax.legend(handles=legend_elements, loc='best')
        plt.show()

    def summary(self):
        """Function to plot a simple summary of the thermal profile.
        """
        print(tabulate([["Position", str(self.position)], ["In control", self.is_in_control], ["beta_0", str(self.b0_eff)], ["beta_1", str(self.b1_eff)], [
              "beta_2", str(self.b2_eff)], ["Entity", str(self.deviation_entity)]], ["Summary of " + self.profile_id, "Value"], tablefmt="grid"))


def main():
    b_0 = (5.421, 0.0105)
    b_1 = (-2.165e-3, 1.773e-4)
    b_2 = (-0.0000001, -0.000523)
    error = (0, 0.00005)

    prova = Simulation("demo", 0, 0, b_0, b_1,
                       b_2[0], b_2[1], error, deviation_entity=3)
    prova.run()

    prova.plot()


if __name__ == "__main__":
    main()
