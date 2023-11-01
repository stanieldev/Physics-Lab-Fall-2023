# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



# Define regression models

# Intersecting tangents method
def intersecting_tangents(x, slope, x_offset):
    return [(0 if _x < x_offset else slope * (_x - x_offset)) for _x in x]

# Sigmoid function method
def sigmoid(x, A, k, x_offset):
    return A / (1 + np.exp(-k * (x - x_offset)))



# Constants
dV = 0.05  # Voltage uncertainty (V)
dI = 0.10  # Current uncertainty (nA)



# Calculate regression of data
def regression_data(data, print_results=True, regression_model=intersecting_tangents):

    # Modify the data
    y_offset = data[:,1][0]  # Find current at endpoint
    _data = data - y_offset  # Set endpoint to 0 nanoamps
    _data = _data[_data[:,0].argsort()]  # Sort by voltage

    # Calculate the regression parameters
    popt, pcov = curve_fit(regression_model, _data[:,0], _data[:,1], p0=[1, 0])

    # Print the regression parameters
    if print_results:
        print(f"Model: {regression_model.__name__}")
        print(f"Slope: {popt[0]:.3f} ± {np.sqrt(pcov[0,0]):.3f} nA/V")
        print(f"V_s:  {popt[1]:.3f} ± {np.sqrt(pcov[1,1]):.3f} V")

    # Return the regression parameters
    return popt, pcov, y_offset

# Plot the data function
def plot_data(data, popt, pcov, y_offset=0, regression_model=intersecting_tangents):

    # Modify the data
    _data = data[data[:,0].argsort()]  # Sort by voltage

    # Set up graph
    plt.xlabel("Voltage (V)")
    plt.ylabel("Photocurrent (nA)")
    plt.title("Photocurrent vs. Voltage")

    # Some pre-calculated values
    SPACE = np.linspace(_data[0,0], _data[-1,0], 10_000)
    V_s, dV_s = popt[1], np.sqrt(pcov[1,1])
    V_s_upper = V_s + dV_s
    V_s_lower = V_s - dV_s

    # Plot the data
    plt.errorbar(_data[:,0], _data[:,1], xerr=dV, yerr=dI, label="Experimental Data", ls='none')
    plt.plot(SPACE, regression_model(SPACE, *popt) + y_offset, label=f"Regression Model: {regression_model.__name__}")
    plt.axvline(x=popt[1], color="red", linestyle="--", label=f"Vs = {popt[1]:.4f} ± {np.sqrt(pcov[1,1]):.4f} V")
    plt.axvline(x=V_s_upper, color="red", linestyle="--", alpha=0.5)
    plt.axvline(x=V_s_lower, color="red", linestyle="--", alpha=0.5)

    # Finalize graph
    plt.legend(loc="upper left")
    plt.show()



# Calculate regression of data gradient
def regression_gradient_data(data, print_results=True, regression_model=sigmoid):

    # Find duplicate voltages and take the average of their currents
    voltages = np.unique(data[:,0])
    currents = np.zeros_like(voltages)
    for i, voltage in enumerate(voltages):
        currents[i] = np.mean(data[data[:,0] == voltage, 1])
    data = np.column_stack((voltages, currents))
    data[:,1] = np.gradient(data[:,1], data[:,0])

    # Calculate the regression parameters
    popt, pcov = curve_fit(regression_model, data[:,0], data[:,1], p0=[1, 0, 0])

    # Print the regression parameters
    if print_results:
        print(f"Model: {regression_model.__name__}")
        print(f"Slope: {popt[0]:.3f} ± {np.sqrt(pcov[0,0]):.3f} nA/V")
        print(f"V_s:  {popt[2]:.3f} ± {np.sqrt(pcov[2,2]):.3f} V")

    # Return the regression parameters
    return popt, pcov

# Plot the data function
def plot_gradient_data(data, popt, pcov, y_offset=0, regression_model=sigmoid):

    # Modify the data
    _data = data[data[:,0].argsort()]  # Sort by voltage

    # Find duplicate voltages and take the average of their currents
    voltages = np.unique(_data[:,0])
    currents = np.zeros_like(voltages)
    for i, voltage in enumerate(voltages):
        currents[i] = np.mean(_data[_data[:,0] == voltage, 1])
    _data = np.column_stack((voltages, currents))
    _data[:,1] = np.gradient(_data[:,1], _data[:,0])
    
    # Set up graph
    plt.xlabel("Voltage (V)")
    plt.ylabel("dI/dV (nA/V)")
    plt.title("Change in Photocurrent vs. Voltage")

    # Some pre-calculated values
    SPACE = np.linspace(_data[0,0], _data[-1,0], 10_000)
    V_s, dV_s = popt[2], np.sqrt(pcov[2,2])
    V_s_upper = V_s + dV_s
    V_s_lower = V_s - dV_s

    # Plot the data
    plt.errorbar(_data[:,0], _data[:,1], xerr=dV, yerr=dI, label="Experimental Data", ls='none')
    plt.plot(SPACE, regression_model(SPACE, *popt) + y_offset, label=f"Regression Model: {regression_model.__name__}")
    plt.axvline(x=popt[2], color="red", linestyle="--", label=f"Vs = {popt[2]:.4f} ± {np.sqrt(pcov[2,2]):.4f} V")
    plt.axvline(x=V_s_upper, color="red", linestyle="--", alpha=0.5)
    plt.axvline(x=V_s_lower, color="red", linestyle="--", alpha=0.5)

    # Finalize graph
    plt.legend(loc="upper left")
    plt.show()



# Compile stopping voltage data
def compile_stopping_voltages(plot=False):

    # Data to process
    data_processing = {
        "4358": "photoelectric_effect/4358A_raw.txt",
        "5461": "photoelectric_effect/5461A_raw.txt",
        "5779": "photoelectric_effect/5779A_raw.txt",
        "5461.0": "photoelectric_effect/5461A_low.txt",
        "5779.0": "photoelectric_effect/5779A_low.txt",
    }

    # Compile data
    waves_vs_stopping_voltages = []
    for wavelength, data_file in data_processing.items():

        # Load data from file
        data = np.loadtxt(data_file, skiprows=1)
        data[:,1] *= 10  # Convert to nanoamps

        # Calculate regression of data
        popt, pcov, y_offset = regression_data(data, print_results=False)
        if plot: plot_data(data, popt, pcov, y_offset)
        waves_vs_stopping_voltages.append([float(wavelength)/10, popt[1], np.sqrt(pcov[1,1])])

        # Calculate regression of data gradient
        popt, pcov = regression_gradient_data(data, print_results=False)
        if plot: plot_gradient_data(data, popt, pcov)
        waves_vs_stopping_voltages.append([float(wavelength)/10, popt[2], np.sqrt(pcov[2,2])])
    
    # Save to file
    np.savetxt("photoelectric_effect/stop_voltages.txt", waves_vs_stopping_voltages, header="Wavelength (nm) | V_s (V) | V_s Uncertainty (V)")



# Plot stopping voltage data vs inverse wavelength
def plot_final_graph():

    # Load data from file
    data = np.loadtxt("photoelectric_effect/stop_voltages.txt", skiprows=1)

    # Take the average of the stopping voltages at each wavelength
    wavelengths = np.unique(data[:,0])
    stopping_voltages = np.zeros_like(wavelengths)
    stopping_voltages_uncertainty = np.zeros_like(wavelengths)
    for i, wavelength in enumerate(wavelengths):
        stopping_voltages[i] = np.mean(data[data[:,0] == wavelength, 1])
        stopping_voltages_uncertainty[i] = np.max(data[data[:,0] == wavelength, 2])

    # Plot the data
    plt.xlabel("1/Wavelength (nm^-1)")
    plt.ylabel("Stopping Voltage (V)")
    plt.title("Stopping Voltage vs. Inverse Wavelength")

    # Some pre-calculated values
    inv_wavelengths = 1/wavelengths
    SPACE = np.linspace(inv_wavelengths[0], inv_wavelengths[-1], 10_000)

    # Plot the data
    plt.errorbar(inv_wavelengths, stopping_voltages, yerr=stopping_voltages_uncertainty, label="Experimental Data", ls='none', marker='o', color='black')

    # Plot a linear regression
    popt, pcov = curve_fit(lambda x, m, b: m*x + b, inv_wavelengths, stopping_voltages)
    plt.plot(SPACE, popt[0]*SPACE + popt[1], label=f"Linear Regression: V_s = {popt[0]:.3f} * 1/λ + {popt[1]:.3f} V")

    # Print the regression parameters
    print(f"Model: Linear Regression")
    print(f"hc: {popt[0]:.3f} ± {np.sqrt(pcov[0,0]):.3f} eV nm")
    print(f"Work Function: {popt[1]:.3f} ± {np.sqrt(pcov[1,1]):.3f} eV")

    # Finalize graph
    plt.legend(loc="upper left")
    plt.show()





# Main function
def main():
    plot_final_graph()
    

    


if __name__ == "__main__":
    main()
