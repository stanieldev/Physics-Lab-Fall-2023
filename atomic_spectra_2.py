import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Gaussian fit function
def gaussian(x, A, μ, σ):
    return A*np.exp(-(x-μ)**2/(2*σ**2))

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

# Iterate one step
def iterate_one_step(data):

    # Find the max value and its corresponding x value
    max_y = np.max(data[:,1])
    max_x = data[np.argmax(data[:,1]), 0]
    # print(f"Max value: {max_y} counts at {max_x}nm")

    # Find the best fit
    Δx, σ, p = find_best_fit(data, max_y, max_x)
    # print(f"Best fit: σ = {σ}nm, R^2 = {p}")
    print(f"λ = {max_x}nm, ({p})")

    # Print the coefficients
    # print(f"A = {max_y} counts")
    # print(f"μ = {max_x}nm")
    # print(f"σ = {σ}nm")

    # Subtract the best fit from the data
    data[:,1] -= gaussian(data[:,0], max_y, max_x, σ)




# Load data
data = np.loadtxt('./atomic_spectra/hydrogen_lights_on_low_saturation.txt', skiprows=1)

# Iterate through the data
for i in range(30):
    iterate_one_step(data)