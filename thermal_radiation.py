import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Plotting done in Week 1's Prelab
def prelab():

    # Load the data
    tungsten_data = np.loadtxt('thermal_radiation/tungsten.txt', skiprows=2)

    # Split the data into two arrays
    W_r = tungsten_data[:,0]
    W_T = tungsten_data[:,1]

    # Modify data
    x = np.log(W_r)
    y = np.log(W_T) - np.log(300)  # Subtract log(300 K) from all temperatures

    # Find the slope of the line
    def linear_regression(x, a): return a * x

    # Fit the data
    c, cov = curve_fit(linear_regression, x, y)
    print(f"c +/- δc = {c[0]:.6f} +/- {np.sqrt(cov[0,0]):.6f}")

    # Calculate r^2
    residuals = y - linear_regression(x, c[0])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"r^2 = {r_squared*100:.4f}%")

    # Plot the data
    plt.plot(x, y, 'o')
    plt.plot(x, linear_regression(x, c[0]), label=f"c = {c[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}")
    plt.xlabel('ln(R/R0)')
    plt.ylabel('ln(T) - ln(300 K)')
    plt.title('Tungsten')
    plt.legend()
    plt.show()




