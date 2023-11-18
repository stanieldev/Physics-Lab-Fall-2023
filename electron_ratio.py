# Import libaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
S1 = 7.772527472542244
dS1 = 0.06713279305611669


# Runnable functions
def calibration_plot():

    # Load the calibration data
    calibration = np.loadtxt("electron_ratio/calibration.txt", skiprows=1)

    # Calculate linear regression
    def linear_regression(x, a):
        return a * x

    # Optimize the linear regression using curve_fit
    popt, pcov = curve_fit(linear_regression, calibration[:,0], calibration[:,1])

    # Plot the calibration data and the linear regression
    plt.plot(calibration[:,0], calibration[:,1], "o")
    plt.plot(calibration[:,0], linear_regression(calibration[:,0], *popt))
    plt.xlabel("Current (A)")
    plt.ylabel("Magnetic Field (gauss)")
    plt.title("Calibration of the Helmholtz Coils")

    # Calculate r^2
    residuals = calibration[:,1] - linear_regression(calibration[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((calibration[:,1] - np.mean(calibration[:,1]))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Console output
    print("Calibration:")
    print(f"s¹ = {popt[0]} ± {np.sqrt(pcov[0][0])}")
    print(f"r² = {r_squared}")

    # Show the graph
    plt.grid()
    plt.show()

def xy_uniformity_plot(derivative=0):

    # Load the data
    xy_uniformity = np.loadtxt("electron_ratio/xy_uniformity.txt", skiprows=1)

    # Universal variables
    plt.xlabel("x (m)")
    plt.grid()

    # Derivative if cases
    if derivative == 0:
        plt.ylabel("B (Gauss)")
        plt.title("Magnetic Field in the XY Plane")
        xy_uniformity[:,1] = xy_uniformity[:,1]
        mean = np.mean(xy_uniformity[:,1])
        plt.plot(xy_uniformity[:,0], [mean for i in range(len(xy_uniformity[:,1]))])
        print(mean)
        plt.ylim(0, max(xy_uniformity[:,1])*2)
    
    elif derivative == 1:
        plt.ylabel("dB/dx (Gauss/m)")
        plt.title("Derivative of Magnetic Field in the XY Plane")
        xy_uniformity[:,1] = np.gradient(xy_uniformity[:,1], xy_uniformity[:,0])
        plt.ylim(min(xy_uniformity[:,1])*2, max(xy_uniformity[:,1])*2)
    
    elif derivative == 2:
        plt.ylabel("d²B/dx² (Gauss/m²)")
        plt.title("Second Derivative of Magnetic Field in the XY Plane")
        xy_uniformity[:,1] = np.gradient(np.gradient(xy_uniformity[:,1], xy_uniformity[:,0]), xy_uniformity[:,0])
        plt.ylim(min(xy_uniformity[:,1])*2, max(xy_uniformity[:,1])*2)

    # Finish off
    plt.plot(xy_uniformity[:,0], xy_uniformity[:,1], "o")

    # Show the graph
    plt.show()

def z_uniformity_plot():

    # Load the data
    z_uniformity = np.loadtxt("electron_ratio/z_uniformity.txt", skiprows=1)

    # Universal variables
    plt.xlabel("z (m)")
    plt.ylabel("B (Gauss)")
    plt.title("Magnetic Field in the Z Direction")
    plt.ylim(0, max(z_uniformity[:,1])*2)
    plt.grid()

    # Plot the data
    plt.plot(z_uniformity[:,0], z_uniformity[:,1], "o")

    # Find the mean and plot it
    mean = np.mean(z_uniformity[:,1])
    plt.plot(z_uniformity[:,0], [mean for i in range(len(z_uniformity[:,1]))])
    print(mean)

    # Plot a parabola regression
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    popt, pcov = curve_fit(parabola, z_uniformity[:,0], z_uniformity[:,1])
    plt.plot(z_uniformity[:,0], parabola(z_uniformity[:,0], *popt))
    
    # Finish graph
    plt.show()

def e_over_m_plot():

    # Load the data
    e_over_m = np.loadtxt("electron_ratio/e_over_m.txt", skiprows=1)

    # Define lists
    x_list = np.sqrt(e_over_m[:,0])/e_over_m[:,1]
    y_list = e_over_m[:,2]

    # Set up plot
    plt.xlabel("sqrt(V)/I (√V/A)")
    plt.ylabel("Electron Orbit Diameter (m)")
    plt.title("Electron Orbit Diameter vs. √V/I")

    # Remove points from both lists whos y value is -1
    x_list = np.delete(x_list, np.where(y_list == -1))
    y_list = np.delete(y_list, np.where(y_list == -1))
    
    # Calculate linear regression
    def linear_regression(x, a, b):
        return a * x + b
    
    # Optimize the linear regression using curve_fit
    popt, pcov = curve_fit(linear_regression, x_list, y_list)

    # Plot the data and the linear regression
    plt.plot(x_list, y_list, "o")
    plt.plot(x_list, linear_regression(x_list, *popt))

    # Calculate r^2
    residuals = y_list - linear_regression(x_list, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_list - np.mean(y_list))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)

    # Console output
    print("Calibration:")
    print(f"s² = {popt[0]} ± {np.sqrt(pcov[0][0])}")
    print(f"r² = {r_squared}")

    # Show the graph
    plt.grid()
    plt.show()



calibration_plot()
xy_uniformity_plot(derivative=0)
z_uniformity_plot()
e_over_m_plot()