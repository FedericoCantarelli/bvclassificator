import numpy as np
from bvclassificator import simulation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True

    frames = np.arange(0, 60, 1)

    profile0 = simulation.sim(frames, tau=0, H=0.95)
    profile1 = simulation.sim(frames, tau=5, H=0.95)
    profile2 = simulation.sim(frames, tau=20, H=0.6)
    profile3 = simulation.sim(frames, tau=20, H=0.95)
    profile4 = simulation.sim(frames, tau=20, H=1.3)

    legend_element = [Line2D([0], [0], color="#e41a1c", ls=":", lw=1, label=r"$\tau= 0$, $H=0.95$"),
                      Line2D([0], [0], color="#377eb8", ls="--",
                             lw=1, label=r"$\tau= 5$, $H=0.95$"),
                      Line2D([0], [0], color="#4daf4a", ls="-.",
                             lw=1, label=r"$\tau= 20$, $H=0.6$"),
                      Line2D([0], [0], color="#984ea3", ls="--",
                             lw=1, label=r"$\tau= 20$, $H=0.95$"),
                      Line2D([0], [0], color="#ff7f00", ls="-.", lw=1, label=r"$\tau= 20$, $H=1.3$")]

    fig, ax = plt.subplots()
    ax.plot(frames, profile0, color="#e41a1c", ls=":", lw=1)
    ax.plot(frames, profile1, color="#377eb8", ls="--", lw=1)
    ax.plot(frames, profile2, color="#4daf4a", ls="-.", lw=1)
    ax.plot(frames, profile3, color="#984ea3", ls="--", lw=1)
    ax.plot(frames, profile4, color="#ff7f00", ls="-.", lw=1)
    ax.legend(handles=legend_element, fontsize="8")
    ax.set_title("Simulated Thermal Profiles", size=12, pad=10)
    ax.set_xlabel("Frame N")
    ax.set_ylabel("Intensity")

    plt.show()
