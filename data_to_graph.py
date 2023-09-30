import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Color of the wavelength
def wave2rgb(wave):
    # This is a port of javascript code from  http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 1
 
    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0
 
    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0
 
    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow (c * factor, gamma)
 
    return f(red), f(green), f(blue)



#data = np.loadtxt("C:\\Users\\sfg99\\Code\\modern-physics-statistics\\atomic_spectra\\hydrogen_lights_off_high_saturation.txt", skiprows=1)
#data = np.loadtxt("C:\\Users\\sfg99\\Code\\modern-physics-statistics\\atomic_spectra\\hydrogen_lights_off_med_saturation.txt", skiprows=1)
# data = np.loadtxt("C:\\Users\\sfg99\\Code\\modern-physics-statistics\\atomic_spectra\\hydrogen_lights_on_low_saturation.txt", skiprows=1)
data = np.loadtxt("C:\\Users\\sfg99\\Code\\modern-physics-statistics\\atomic_spectra\\raw_helium.txt", skiprows=1)



# Modify data to be graphed
def initialize_data(data: np.ndarray, absolute=False) -> np.ndarray:

    # Remove points outside visible spectrum
    data = data[np.where(data[:, 0] >= 380)]
    data = data[np.where(data[:, 0] <= 780)]

    # Intensity picking
    if absolute:
        # Absolute value of intensity
        for i in range(len(data)):
            data[i, 1] = abs(data[i, 1])
    else:
        # Remove negative intensity
        data = data[np.where(data[:, 1] >= 0)]
    
    # Return data
    return data

# Take the log of the intensity
def log_intensity(data: np.ndarray, base=10) -> None:
    for i in range(len(data)):
        if data[i, 1] < 1: data[i, 1] = 1
        data[i, 1] = np.log(data[i, 1]) / np.log(base)

# Take the square root of the intensity
def power_intensity(data: np.ndarray, degree) -> None:
    for i in range(len(data)):
        data[i, 1] = data[i, 1] ** degree

# Normalize intensity
def normalize_intensity(data: np.ndarray) -> None:
    max_intensity = max(data[:, 1])
    for i in range(len(data)):
        data[i, 1] /= max_intensity


# Plot characteristics
plt.style.use('dark_background')


def plot_wavelength_distribution(fig: plt, data: np.ndarray):
    fig.title("Helium Spectrum")
    fig.xlim(380, 780)
    fig.ylim(0, 1.1)
    fig.ylabel("Relative Intensity")
    for i in range(len(data)):
        fig.vlines(data[i, 0], 0, data[i, 1], color=wave2rgb(data[i, 0]))

def plot_wavelength_spectral(fig: plt, data: np.ndarray):
    fig.title("Helium Spectrum")
    fig.xlim(380, 780)
    fig.xticks(np.arange(380, 780+1, 20))
    fig.yticks([])
    fig.xlabel("Wavelength (nm)")
    for i in range(len(data)):
        fig.axvline(data[i, 0], color=wave2rgb(data[i, 0]), alpha=data[i, 1])

# Initialize data
data = initialize_data(data, absolute=True)

# Extra data modifications
log_intensity(data)
normalize_intensity(data)

# Find maximum intensity in range
def find_maximinum_in_range(data, start, end):
    max_intensity = 0
    max_wavelength = 0
    for i in range(len(data)):
        if data[i, 0] >= start and data[i, 0] <= end:
            if data[i, 1] > max_intensity:
                max_intensity = data[i, 1]
                max_wavelength = data[i, 0]
    return max_wavelength, max_intensity

# Find maximum intensity in interval
def find_maximinum_in_interval(data, x, dx):
    return find_maximinum_in_range(data, x - dx, x + dx)

# Plot data
plot_wavelength_distribution(plt, data)

# Show
plt.show()


# List of wavelengths
nums = [443]

# For each wavelength, find the maximum intensity in the interval
for num in nums:
    print(find_maximinum_in_interval(data, num, 2)[0])


