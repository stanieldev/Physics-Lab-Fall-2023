# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# LAMBDA = 656e-6 # Wavelength of laser (mm)
LAMBDA = 545e-6 # Wavelength of laser (mm)


# Double slit diffraction function
def double_slit(x, a, d, Δθ):
    alpha = (np.pi/LAMBDA) * a * np.sin((x + Δθ)*np.pi/180)
    beta = (np.pi/LAMBDA) * d * np.sin((x + Δθ)*np.pi/180)
    return ( np.cos(beta)**2 ) * ( np.sin(alpha)**2 ) / alpha**2

# Single slit diffraction function
def single_slit(x, a, Δθ):
    alpha = (np.pi/LAMBDA) * a * np.sin((x + Δθ)*np.pi/180)
    return ( np.sin(alpha)**2 ) / alpha**2

# Double slit lambda function
def double_slit_lambda(x, wavelength, Δθ):
    a = 91.391/1000
    d = 448.980/1000
    alpha = (np.pi/wavelength) * a * np.sin((x + Δθ)*np.pi/180)
    beta = (np.pi/wavelength) * d * np.sin((x + Δθ)*np.pi/180)
    return ( np.cos(beta)**2 ) * ( np.sin(alpha)**2 ) / alpha**2


# Fit double slit data
def fit_double_slit(theta, intensity):
    popt, pcov = curve_fit(double_slit, theta, intensity, p0=[0.5, 0.5, 0])
    return popt, pcov

# Fit single slit data
def fit_single_slit(theta, intensity):
    popt, pcov = curve_fit(single_slit, theta, intensity, p0=[0.5, 0])
    return popt, pcov



# Plot double slit data
def plot_double_slit(θ_deg, δθ_deg, I, δI, plot_fit=True, s=False):

    # Set up graph
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Intensity vs. Angle')

    # Plot intensity vs. position
    plt.errorbar(θ_deg, I, xerr=δθ_deg, yerr=δI, fmt='x', color='black')

    # Plot fit
    if plot_fit:

        # Fit data
        popt, pcov = fit_double_slit(θ_deg, I)

        # Plot fit
        SPACE = np.linspace(min(θ_deg), max(θ_deg), 1000)
        plt.plot(SPACE, double_slit(SPACE, *popt), color='green')

        # Print fit parameters
        print(f"a = {popt[0]} ± {np.sqrt(pcov[0, 0])}")
        print(f"d = {popt[1]} ± {np.sqrt(pcov[1, 1])}")
        print(f"Δθ = {popt[2]} ± {np.sqrt(pcov[2, 2])}")

        # Calculate r² value
        residuals = I - double_slit(θ_deg, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I - np.mean(I))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"r² = {r_squared}")

    # Special plot for part 2
    if s:
        # Fit data
        popt, pcov = curve_fit(double_slit_lambda, θ_deg, I, p0=[LAMBDA, 0])

        # Plot fit
        SPACE = np.linspace(min(θ_deg), max(θ_deg), 1000)
        plt.plot(SPACE, double_slit_lambda(SPACE, *popt), color='green')

        # Print fit parameters
        print(f"λ = {popt[0]} ± {np.sqrt(pcov[0, 0])}")
        print(f"Δθ = {popt[1]} ± {np.sqrt(pcov[1, 1])}")

        # Calculate r² value
        residuals = I - double_slit_lambda(θ_deg, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I - np.mean(I))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"r² = {r_squared}")

    # Show plot
    plt.show()

# Plot single slit data
def plot_single_slit(θ_deg, δθ_deg, I, δI, plot_fit=True):

    # Set up graph
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Intensity vs. Angle')

    # Plot intensity vs. position
    plt.errorbar(θ_deg, I, xerr=δθ_deg, yerr=δI, fmt='x', color='black')

    # Plot fit
    if plot_fit:

        # Fit data
        popt, pcov = fit_single_slit(θ_deg, I)

        # Plot fit
        SPACE = np.linspace(min(θ_deg), max(θ_deg), 1000)
        plt.plot(SPACE, single_slit(SPACE, *popt), color='red')

        # Print fit parameters
        print(f"a = {popt[0]} ± {np.sqrt(pcov[0, 0])}")
        print(f"Δθ = {popt[1]} ± {np.sqrt(pcov[1, 1])}")

        # Calculate r² value
        residuals = I - single_slit(θ_deg, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I - np.mean(I))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"r² = {r_squared}")

    # Show plot
    plt.show()

# Plot signal to noise ratio
def plot_signal_noise(V, δV, SNR, δSNR):
    
    # Set up graph
    plt.xlabel('Voltage (kV)')
    plt.ylabel('Signal to Noise Ratio')
    plt.title('Signal to Noise Ratio vs. Voltage')

    # Plot signal to noise ratio vs. voltage
    plt.errorbar(V, SNR, xerr=δV, yerr=δSNR, fmt='x', color='black')

    # Show plot
    plt.show()



# Load data function
def load_data(filename):

    # Constants
    δx = 0.01    # mm
    δI = 0.0005  # volts
    D = 500      # mm

    # Load data
    data = np.loadtxt(filename, skiprows=1)

    # Split data
    X, I = data[:, 0], data[:, 1]

    # Position adjustments
    X -= X[np.argmax(I)]
    θ_rad = np.arctan(X/D)
    δθ_rad = (np.cos(θ_rad)**2 / D) * δx
    θ_deg = θ_rad * (180 / np.pi) + 1e-12  # Add small value to avoid division by zero
    δθ_deg = δθ_rad * (180 / np.pi)

    # Intensity adjustments
    I -= np.min(I)
    δI /= np.max(I)
    I /= np.max(I)

    # Return adjusted data
    return (θ_deg, δθ_deg), (I, δI)

# Load Signal-Noise data
def load_signal_noise(filename):

    # Constants
    δV = 1e-3  # kV

    # Load data
    data = np.loadtxt(filename, skiprows=1)

    # Split data
    voltage, dark, light = data[:, 0], data[:, 1], data[:, 2]

    # Calculate signal and noise
    SNR = (light - dark) / dark
    δSNR = np.sqrt(light + (light**2 / dark)) / dark

    # Return adjusted data
    return (voltage, δV), (SNR, δSNR)
    



# # Signal to noise ratio
# (V, δV), (SNR, δSNR) = load_signal_noise('./double_slit/w2_signal.txt')
# plot_signal_noise(V, δV, SNR, δSNR)

# # Single-Photon Double slit 
# (θ_deg, δθ_deg), (I, δI) = load_data('./double_slit/w2_double.txt')
# plot_double_slit(θ_deg, δθ_deg, I, δI, plot_fit=False, s=True)


# # Front single slit
# (θ_deg, δθ_deg), (I, δI) = load_data('./double_slit/w1_front_single.txt')
# plot_single_slit(θ_deg, δθ_deg, I, δI, plot_fit=True)

# # Back single slit
# (θ_deg, δθ_deg), (I, δI) = load_data('./double_slit/w1_back_single.txt')
# plot_single_slit(θ_deg, δθ_deg, I, δI, plot_fit=True)

# # Middle double slit
# (θ_deg, δθ_deg), (I, δI) = load_data('./double_slit/w1_middle_double.txt')
# plot_double_slit(θ_deg, δθ_deg, I, δI, plot_fit=True)

