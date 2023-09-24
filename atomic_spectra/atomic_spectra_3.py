import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Gaussian fit function
def gaussian(x, A, μ, σ):
    return A*np.exp(-(x-μ)**2/(2*σ**2))

# Inverse Gaussian fit function
def inverse_gaussian(x, A, μ, σ):
    return 1 - gaussian(x, A, μ, σ) / A

# Create subinterval function
def subinterval(data, x, Δx):
    return data[np.where(np.logical_and(data[:,0] >= x-Δx, data[:,0] <= x+Δx))]

# Find the best fit
def find_best_fit(data, A: float, μ: float):

    # Initialize the best fit
    best_fit = [0, 0, 0] # [Δx, σ, R^2]

    # Iterate through a range of Δx values
    for Δx in range(2, 20):

        # Create a subinterval of data of max_x +- Δx
        subset = subinterval(data, μ, Δx)

        # Using the max value and x value, estimate the standard deviation of the Gaussian fit with A = max_y and μ = max_x
        def restricted_gaussian(x, σ):
            return gaussian(x, A, μ, σ)
        params, cov_matrix = curve_fit(restricted_gaussian, subset[:,0], subset[:,1], p0=[1])

        # Calculate the R^2 value
        residuals = subset[:,1] - restricted_gaussian(subset[:,0], *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((subset[:,1] - np.mean(subset[:,1]))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # If the R^2 value is better than the previous best fit, update the best fit
        if r_squared > best_fit[2]:
            best_fit = [Δx, params[0], r_squared]
    
    # Return the standard deviation of the best fit
    return best_fit


# Different ways to modify the data
def modify_data_v1(data, max_x, Δx):
    # If the wavelength is within 3σ of max_x, set the intensity to 0
    for i in range(len(data)):
        if data[i,0] >= max_x-Δx and data[i,0] <= max_x+Δx:
            data[i,1] = 0

def modify_data_v2(data, A, μ, σ):
    # Subtract the Gaussian fit from the data
    data[:,1] -= gaussian(data[:,0], A, μ, σ)

def modify_data_v3(data, A, μ, σ):
    # Inverse Gaussian by 5 sigma to the data
    return data[:,1] * inverse_gaussian(data[:,0], A, μ, σ)



# Iterate one step
def iterate_one_step(data):

    # Find the max value and its corresponding x value
    max_y = np.max(data[:,1])
    max_x = data[np.argmax(data[:,1]), 0]

    # Find the best fit
    Δx, σ, p = find_best_fit(data, max_y, max_x)
    if max_x <= 700:
        print(f"λ = {max_x}nm, ({p})")

    # Modify the data
    # modify_data_v1(data, max_x, 5*σ)
    # modify_data_v2(data, max_y, max_x, σ)
    new_data = modify_data_v3(data, max_y, max_x, 5*σ)

    # Plot the data
    # plt.plot(data[:,0], data[:,1], label="Original data")
    # plt.plot(data[:,0], new_data, label="Modified data")
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Intensity (a.u.)")
    # plt.legend()
    #plt.show()

    # Swap data
    data[:,1] = new_data

    # Return wavelength
    return max_x



# Load data
data_low_on = np.loadtxt('./hydrogen_lights_on_low_saturation.txt', skiprows=1)
data_high_on = np.loadtxt('./hydrogen_lights_on_high_saturation.txt', skiprows=1)
data_low_off = np.loadtxt('./hydrogen_lights_off_low_saturation.txt', skiprows=1)
data_high_off = np.loadtxt('./hydrogen_lights_off_high_saturation.txt', skiprows=1)
data_med_off = np.loadtxt('./hydrogen_lights_off_med_saturation.txt', skiprows=1)
data_med_on = np.loadtxt('./hydrogen_lights_on_med_saturation.txt', skiprows=1)

# Modify data
for i in range(len(data_high_on)):
    if data_high_on[i,0] >= 640 and data_high_on[i,0] <= 670:
        data_high_on[i,1] = 0
for i in range(len(data_high_off)):
    if data_high_off[i,0] >= 640 and data_high_off[i,0] <= 670:
        data_high_off[i,1] = 0



# Plot the data
plt.plot(data_med_off[:,0], data_med_off[:,1])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (a.u.)")
plt.show()

def run_through(data):
    wavelengths = []
    while True:
        try:
            wavelengths.append(iterate_one_step(data))
        except ValueError:
            break
    return wavelengths

# Collect wavelengths
dlo = run_through(data_low_on)
dho = run_through(data_high_on)
dlof = run_through(data_low_off)
dhof = run_through(data_high_off)
dmo = run_through(data_med_on)
dmof = run_through(data_med_off)
experimental = [656.279, 486.135, 434.0472, 410.1734]

# Plot the wavelengths as a scatter plot
plt.scatter(dlo, 3 * np.ones(len(dlo)), label="Low on")
plt.scatter(dho, 2 * np.ones(len(dho)), label="High on")
plt.scatter(dlof, 1 * np.ones(len(dlof)), label="Low off")
plt.vlines(experimental, label="Experimental", ymin=-4, ymax=4, color="red")
plt.scatter(dhof, -1 * np.ones(len(dhof)), label="High off")
plt.scatter(dmo, -2 * np.ones(len(dmo)), label="Med on")
plt.scatter(dmof, -3 * np.ones(len(dmof)), label="Med off")
plt.xlabel("Wavelength (nm)")
plt.yticks([])
plt.legend()
plt.show()