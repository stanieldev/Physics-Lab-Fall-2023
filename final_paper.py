# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Universal
plt.style.use('dark_background')
D = (1/600) * 1e6 # nm


# Class enum class
class Color:
    PINK = -1
    RED = 3
    CYAN = 4
    BLUE = 5
    VIOLET = 6

    def __init__(self, n: int):
        # Integer check
        if not isinstance(n, int): raise TypeError("Principle quantum number must be an integer")

        # Check if n is a valid integer (-1, 3, 4, 5, 6,...)
        if n < -1: raise ValueError("Principle quantum number must be greater than 2")
        if n > -1 and n < 3: raise ValueError("Principle quantum number must be greater than 2")
        if n > 6: raise NotImplementedError("Principle quantum number must be less than 7")

        # Set n to the principle quantum number
        self._n = n

    def __repr__(self):
        if self._n == -1: return "Pink"
        return ["Red", "Cyan", "Blue", "Violet"][self._n - 3]

    def toWavelength(self):
        EXPECTED = [656.279, 486.135, 434.0472, 410.1734]
        if self._n == -1: raise ValueError("Pink is not a valid color")
        return EXPECTED[self._n - 3]

    def toRGB(self):
        if self._n == -1: return 255/256, 192/256, 203/256
        return Color.RGB(self.toWavelength())
    
    def RGB(λ: float):

        # This is a port of javascript code from  http://stackoverflow.com/a/14917481
        gamma = 0.8
        intensity_max = 1

        if λ < 380:
            red, green, blue = 0, 0, 0
        elif λ < 440:
            red = -(λ - 440) / (440 - 380)
            green, blue = 0, 1
        elif λ < 490:
            red = 0
            green = (λ - 440) / (490 - 440)
            blue = 1
        elif λ < 510:
            red, green = 0, 1
            blue = -(λ - 510) / (510 - 490)
        elif λ < 580:
            red = (λ - 510) / (580 - 510)
            green, blue = 1, 0
        elif λ < 645:
            red = 1
            green = -(λ - 645) / (645 - 580)
            blue = 0
        elif λ <= 780:
            red, green, blue = 1, 0, 0
        else:
            red, green, blue = 0, 0, 0
    
        # let the intensity fall of near the vision limits
        if λ < 380:
            factor = 0
        elif λ < 420:
            factor = 0.3 + 0.7 * (λ - 380) / (420 - 380)
        elif λ < 700:
            factor = 1
        elif λ <= 780:
            factor = 0.3 + 0.7 * (780 - λ) / (780 - 700)
        else:
            factor = 0
    
        def _f(c):
            if c == 0:
                return 0
            else:
                return intensity_max * pow (c * factor, gamma)
    
        return _f(red), _f(green), _f(blue)


# Plotting functions
def plot_diffraction_spectrum(angles: list[float], colors: list[Color]) -> None:

    # Set up plot
    plt.title("Diffraction Spectrum")
    plt.xlabel("Angle (degrees)")
    plt.xlim(-90, 90)
    plt.xticks(np.arange(-80, 90, 10), abs(np.arange(-80, 90, 10)))
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.ylim(-1, 1)

    # Plot vertical lines with colors
    for angle, color in zip(angles, colors):
        plt.vlines(angle, ymin=0, ymax=1, color=color.toRGB(), linewidth=1)

    # Theoretical colors
    wavelengths = [656.279, 486.135, 434.0472, 410.1734]
    plt.vlines(0, ymin=-1, ymax=0, color=(255/256, 192/256, 203/256), linewidth=1)
    for λ in wavelengths:
        for m in range(1, int(np.ceil(D/λ))):
            plt.vlines(np.arcsin(m * λ / D) * (180/np.pi), ymin=-1, ymax=0, color=Color.RGB(λ), linewidth=1)
            plt.vlines(-np.arcsin(m * λ / D) * (180/np.pi), ymin=-1, ymax=0, color=Color.RGB(λ), linewidth=1)


# Doing statistics
def doing_statistics():
    files_list = ["./final_paper/test_file.txt"]
    data_list = []
    colors = [Color(int(color)) for color in np.loadtxt(files_list[0], skiprows=1)[:,2]]

    # Import data
    for file in files_list:
        data = np.loadtxt(file, skiprows=1)
        angles = data[:,0] + data[:,1]/60
        angles = angles - angles[int((len(angles) - 1)/2)]
        data_list.append(angles)

    # Transpose the datalist
    data_list = np.array(data_list).T

    # For every row, find the average and standard deviation
    averages = []
    standard_deviations = []
    for row in data_list:
        averages.append(np.average(row))
        standard_deviations.append(np.std(row))
    [print(k, i, j) for i, j, k in zip(averages, standard_deviations, colors)]





# Test file data
data = np.loadtxt("./final_paper/test_file.txt", skiprows=1)
angles = data[:,0] + data[:,1]/60
angles = angles - angles[int((len(angles) - 1)/2)]
colors = [Color(int(color)) for color in data[:,2]]

# Plot the measured diffraction pattern
plot_diffraction_spectrum(angles, colors)
plt.show()


# Use lots of datapoints to find X, dX for each line
# Take X/m for each line and θ, dθ for each wavelength
# Use θ, dθ to find λ, dλ


