# Import required libraries
import matplotlib.pyplot as plt
import numpy as np


# Universal settings
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#181818'

def constrain(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


# Define a function that takes in a wavelength in nm and returns the corresponding RGB value
def RGB(λ: float):

    # This is a port of javascript code from  http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 1
    K = 0.0025

    if λ < 380:
        red, green, blue = 1, 1 - np.exp(25*K*(λ - 380)), 1
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
        red, green, blue = 1, 0, 0

    # let the intensity fall of near the vision limits
    if λ < 380:
        factor = 0.3 * np.exp(-2*K*(λ - 380)**2)
    elif λ < 420:
        factor = 0.3 + 0.7 * (λ - 380) / (420 - 380)
    elif λ < 700:
        factor = 1
    elif λ <= 780:
        factor = 0.3 + 0.7 * (780 - λ) / (780 - 700)
    else:
        factor = 0.3 * np.exp(-0.5*K*(λ - 780)**2)

    def _f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow (c * factor, gamma)

    return _f(red), _f(green), _f(blue)


# Plot data
def bars_plot(wavelengths, intensities, log=False, filter=False):

    # Set up the figure
    plt.title('Emission Spectrum of Hydrogen')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (arb. units)')

    # Find the width of each bar using the adjacient wavelengths
    widths = np.diff(wavelengths)
    widths = np.append(widths, widths[-1])
    widths = widths * 4

    # If log is true, take the log of the intensities
    if log:
        intensities[intensities < 1] = 1
        intensities = np.log(intensities)
        plt.ylabel('Intensity (log scale)')
    
    # If filter is true, apply a low-pass filter to the intensities
    if filter:
        intensities = np.convolve(intensities, np.ones(10)/10, mode='same')

    
    # # Create a list of intensities
    # normalized_intensity = intensities.copy() / np.max(intensities)

    # Set the x and y limits
    plt.xlim(min(wavelengths), max(wavelengths))
    plt.ylim(0, max(intensities)*1.1)

    # Plot bars
    plt.bar(wavelengths, intensities, width=widths, color=[RGB(l) for l in wavelengths])

# Plot data
def emissions_plot(wavelengths, intensities):

    # Set the x and y limits
    plt.xlim(min(wavelengths), max(wavelengths))
    plt.ylim(0, 1)

    # Create a list of intensities
    normalized_intensity = intensities.copy() / np.max(intensities)

    # Plot vlines
    plt.vlines(wavelengths, ymin=0, ymax=1, color=[RGB(l) for l in wavelengths], alpha=np.sqrt(normalized_intensity))




# Generate better data
def weighted_data():

    # Constants
    K_med = 64320/16755
    K_low = 64320/4817

    # Load data
    data_high = np.loadtxt('H_high_on.txt', unpack=True, skiprows=1)
    data_med = np.loadtxt('H_med_on.txt', unpack=True, skiprows=1)
    data_low = np.loadtxt('H_low_on.txt', unpack=True, skiprows=1)

    # Create a weighed data array
    p = data_high[1]/np.max(data_high[1])
    better_data = data_high[1] * (1 - p) ** 2 + \
                data_med[1] * K_med * 2 * (1 - p) * p + \
                data_low[1] * K_low * p ** 2

    # Return the wavelengths and the weighed data
    better_data = np.abs(better_data)
    better_data[better_data < 0] = 0
    return data_high[0], better_data




wavelengths, better_data = weighted_data()



# emissions_plot(wavelengths, better_data)
# plt.show()

bars_plot(wavelengths, better_data, log=True)
plt.show()



# bars_plot(data_low[0], better_data)





# # Plot vertical lines
# plt.xlim(min(data_low[0]), max(data_low[0]))
# plt.ylim(min(better_data), max(better_data)*1.1)
# plt.show()







# # Load H_high_off.txt data
# data_high = np.loadtxt('H_low_on.txt', unpack=True, skiprows=1)
# wavelengths = data_high[0]
# intensities = data_high[1]

# # Modify intensities

# # Take the absolute values of intensities
# intensities = np.abs(intensities)
# intensities[intensities < 1] = 1
# intensities = np.log(intensities)
# intensities = intensities / np.max(intensities)


