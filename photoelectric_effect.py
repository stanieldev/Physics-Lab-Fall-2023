# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Constants
dV = 0.05  # Voltage uncertainty (V)
dI = 0.10  # Current uncertainty (nA)
C = 299_792_458  # Speed of light (m/s)
E = 1.602176634e-19  # Elementary charge (C)


# Statistical number class
class StatNum:
    def __init__(self, value, uncertainty):
        self.value = value
        self.uncertainty = uncertainty
    def __str__(self):
        return f"{self.value:.4f} ± {self.uncertainty:.4f}"
    def __repr__(self):
        return f"StatNum({self.value}, {self.uncertainty})"

# Remove duplicate inputs and take the average of their outputs
def merge_duplicates(data):
    x = np.unique(data[:,0])
    y = np.zeros_like(x)
    for i, j in enumerate(x):
        y[i] = np.mean(data[data[:,0] == j, 1])
    return np.column_stack((x, y))

def merge_duplicates_3(data):
    x = np.unique(data[:,0])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    for i, j in enumerate(x):
        y[i] = np.mean(data[data[:,0] == j, 1])
        z[i] = np.max(data[data[:,0] == j, 2])
    return np.column_stack((x, y, z))



# Intersecting tangents functions
def ReLU(x, a, b, c=0):
    return [(c if _x < b else a * (_x - b) + c) for _x in x]
def ReLU_zeroed(x, a, b):
    return [(0 if _x < b else a * (_x - b) + 0) for _x in x]

# Sigmoid function
def sigmoid(x, A, k, x_offset):
    return A / (1 + np.exp(-k * (x - x_offset)))



# ReLU regression function
def ReLU_regression(data, print_results=True):

    # Modify the data
    points = data.copy()
    current_offset = points[:,1][0]  # Find current at endpoint
    points[:,1] -= current_offset    # Subtract current at endpoint from all currents

    # Find the best fit for the ReLU regression
    popt, pcov = curve_fit(ReLU_zeroed, points[:,0], points[:,1], p0=[1, -2])

    # Print the regression parameters
    if print_results:
        print(f"Model: ReLU Regression")
        print(f"Slope: {popt[0]:.3f} ± {np.sqrt(pcov[0,0]):.3f} nA/V")
        print(f"V_s:  {popt[1]:.3f} ± {np.sqrt(pcov[1,1]):.3f} V")

    # Return the regression parameters and error
    params = [popt[0], popt[1], current_offset]
    error = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), 0]
    return params, error

# Sigmoid regression function
def sigmoid_regression(data, print_results=True):

    # Find the best fit for the Sigmoid regression
    params, pcov = curve_fit(sigmoid, data[:,0], data[:,1], p0=[1, 0, 0])

    # Print the regression parameters
    if print_results:
        print(f"Model: Sigmoid Regression")
        print(f"Slope: {params[0]:.3f} ± {np.sqrt(pcov[0,0]):.3f} nA/V")
        print(f"V_s:  {params[2]:.3f} ± {np.sqrt(pcov[2,2]):.3f} V")

    # Return the regression parameters and error
    error = [np.sqrt(pcov[i,i]) for i in range(len(params))]
    return params, error

# Stopping voltage function here
def calc_stopping_voltage(data, *, model):
    if model == ReLU:
        params, error = ReLU_regression(data, print_results=False)
        return params[1], error[1]
    elif model == sigmoid:
        params, error = sigmoid_regression(data, print_results=False)
        return params[2], error[2]
    else:
        raise NotImplementedError(f"Regression model {model.__name__} not implemented")



# Plot the data
def plot_datapoints(data, color):
    currents = merge_duplicates(data)
    plt.errorbar(currents[:,0], currents[:,1], color=color, 
                 xerr=dV, yerr=dI, ls='none',
                 marker='o', markersize=4, alpha=0.50)

# Plot the activation function
def plot_regression(data, color, model=ReLU):

    # Some pre-calculated values
    SPACE = np.linspace(-4.0, 4.0, 1000)

    # Calculate the regression
    if model == ReLU:
        params, _ = ReLU_regression(data, print_results=False)
    elif model == sigmoid:
        params, _ = sigmoid_regression(data, print_results=False)
    else:
        raise NotImplementedError(f"Regression model {model.__name__} not implemented")
    
    # Plot the regression
    plt.plot(SPACE, model(SPACE, *params), color=color)

# Plot the stopping voltage
def plot_stopping_voltage(data, color, model):

    # Calculate the stopping voltage
    V_s, dV_s = calc_stopping_voltage(data, model=model)

    # Plot the stopping voltage
    plt.axvline(x=V_s, color=color, linestyle="--", label=f"Vs = {V_s:.4f} ± {dV_s:.4f} V")
    plt.axvline(x=V_s+dV_s, color=color, linestyle="--", alpha=0.5)
    plt.axvline(x=V_s-dV_s, color=color, linestyle="--", alpha=0.5)



# Plot file function
def plot_file(file, color="blue", gradient=False, show_points=True, show_regression=True, show_bound=False):

    # Import data and sort by voltage
    rough_data = np.loadtxt(file[:-4] + "_rough.txt", skiprows=1);  rough_data[:,1] *= 10  # 10^-8 A to nA
    fine_data  = np.loadtxt(file[:-4] + "_fine.txt",  skiprows=1);  fine_data[:,1] *= 10  # 10^-8 A to nA
    data = np.concatenate((rough_data, fine_data))

    # Sort by voltage
    fine_data = fine_data[fine_data[:,0].argsort()]
    data = data[data[:,0].argsort()]

    # Converts the data to gradient if gradient is True
    if gradient:
        fine_data = merge_duplicates(fine_data)
        data = merge_duplicates(data)
    if gradient:
        fine_data[:,1] = np.gradient(fine_data[:,1], fine_data[:,0])
        data[:,1] = np.gradient(data[:,1], data[:,0])

    # Set up graph
    wavelength = (file.split("_")[1].split(".")[0])[-5:-1]
    plt.xlabel("Voltage (V)")
    plt.ylabel("Photocurrent (nA)") if not gradient else plt.ylabel("Photocurrent per Volt (nA/V)")
    # plt.title(f"Photocurrent vs. Voltage ({int(wavelength)/10}nm)") if not gradient else plt.title("Change in Photocurrent vs. Voltage")

    # Plot the data
    if show_points:
        plot_datapoints(data, color=color)

    # Plot the regression
    if show_regression:
        if not gradient: plot_regression(fine_data, color=color, model=ReLU)
        elif gradient: plot_regression(fine_data, color=color, model=sigmoid)

    # Plot the stopping voltage
    if show_bound:
        if not gradient: plot_stopping_voltage(fine_data, color=color, model=ReLU)
        elif gradient: plot_stopping_voltage(fine_data, color=color, model=sigmoid)
    
    # Make sure the legend is visible
    plt.legend(loc="upper left")

# Calculate file function
def calc_file(file, print_results=True):

    # Import data and sort by voltage
    rough_data = np.loadtxt(file[:-4] + "_rough.txt", skiprows=1);  rough_data[:,1] *= 10  # 10^-8 A to nA
    fine_data  = np.loadtxt(file[:-4] + "_fine.txt",  skiprows=1);  fine_data[:,1] *= 10  # 10^-8 A to nA
    data = np.concatenate((rough_data, fine_data))

    # Sort by voltage
    fine_data = fine_data[fine_data[:,0].argsort()]
    data = data[data[:,0].argsort()]

    # Calculate stopping voltage for ReLU regression
    V_ReLU = StatNum(*calc_stopping_voltage(fine_data, model=ReLU))

    # Convert the data to its gradient
    fine_data = merge_duplicates(fine_data)
    data = merge_duplicates(data)
    fine_data[:,1] = np.gradient(fine_data[:,1], fine_data[:,0])
    data[:,1] = np.gradient(data[:,1], data[:,0])

    # Calculate stopping voltage for sigmoid regression
    V_sigmoid = StatNum(*calc_stopping_voltage(fine_data, model=sigmoid))

    # Print a table with the data
    if print_results:
        print(f"\nFile: {file}")
        print( "  Model  |  Stopping Voltage (V)  |  Uncertainty (V)   |")
        print( "---------|------------------------|--------------------|")
        print(f"ReLU     |  {V_ReLU.value:.16f}   | {V_ReLU.uncertainty:.16f} |")
        print(f"Sigmoid  |  {V_sigmoid.value:.16f}   | {V_sigmoid.uncertainty:.16f} |")

    # Return the stopping voltages
    return V_ReLU, V_sigmoid

# Calculate files function
def calc_files(files: list, wavelengths: list, save_linear=True, save_sigma=False):
    save_lines = []
    for file, wavelength in zip(files, wavelengths):

        # Calculate the stopping voltages
        a, b = calc_file(file, print_results=False)

        # Save the data
        if save_linear: save_lines.append(f"{wavelength} {a.value} {a.uncertainty}")
        if save_sigma: save_lines.append(f"{wavelength} {b.value} {b.uncertainty}")

    # Save the data to a file using numpy
    np.savetxt("photoelectric_effect/stop_voltages.txt", np.array(save_lines), fmt="%s", header="Wavelength (nm) | Stopping_Voltage (Vs) | Uncertainty (Vs)")







# Plot stopping voltage data vs frequency
def plot_final_graph(mergedata=False):

    # Load data from file

    data = np.loadtxt("photoelectric_effect/stop_voltages.txt", skiprows=1)
    if mergedata: data = merge_duplicates_3(data)

    # Calculate useful lists
    frequencies = (C*10**9)/data[:,0]
    stopping_voltage = data[:,1]
    stopping_uncertainty = data[:,2]

    # Calculate the best fit line
    params, covar = curve_fit(lambda x, m, b: m*x + b, frequencies, stopping_voltage, p0=[1, 0])

    # Set up graph
    plt.xlabel("Light Frequency (Hz)")
    plt.ylabel("Stopping Voltage (V)")
    plt.title("Stopping Voltage vs. Light Frequency")

    # Plot the data
    # colors = ["blue", "green", "yellow"]
    # [plt.errorbar(frequencies[i], stopping_voltage[i], xerr=0, yerr=stopping_uncertainty[i], 
    #              color=colors[i], ls='none', marker='o', markersize=4, alpha=0.50) for i in range(3)]
    plt.errorbar(frequencies, stopping_voltage, xerr=0, yerr=stopping_uncertainty, 
                 color="white", ls='none', marker='o', markersize=4, alpha=0.50)
    plt.plot(frequencies, params[0]*frequencies + params[1], color="white")

    # Plot extra x ticks at each frequency with the wavelength as the label
    plt.xticks(frequencies, [f"{i}nm" for i in data[:,0]], ha="center")

    # Pretty format the parameters
    def pretty_format(x, dx, N=15):

        # Find the exponent of x
        exponent = int(np.floor(np.log10(abs(x))))

        # Multiply x and dx by 10^-exponent
        x *= 10**-exponent
        dx *= 10**-exponent

        # Round x and dx to N significant figures
        x = np.round(x, N - int(exponent))
        dx = np.round(dx, N - int(exponent))

        # If exponent is less than 10, add a leading zero
        if abs(exponent) < 10: 
            if exponent < 0: exponent = f"-0{abs(int(exponent))}"
            elif exponent > 0: exponent = f"+0{int(exponent)}"

        # Return the formatted string
        return f"{x:.{N}f}e{exponent}", f"{dx:.{N}f}e{exponent}"

    # Create the formatted strings
    H, dH = pretty_format(-params[0], np.sqrt(covar[0,0]))
    φ, dφ = pretty_format(params[1], np.sqrt(covar[1,1]))
    H_C, dH_C = pretty_format(-params[0]*C*10**9, np.sqrt(covar[0,0])*C*10**9)

    # Print the best fit line
    print(f"Model: Linear Regression")
    print("Expected Distribution: V_s = (h/e)f - (φ/e)")
    print( " Parameter |         Value         |      Uncertainty      |")
    print( "-----------|-----------------------|-----------------------|")
    print(f" h/e (V s) | {         H         } | {        dH         } |")
    print(f"  φ/e (V)  | {         φ         }   | {        dφ         }   |")
    print( "-----------|-----------------------|-----------------------|")
    print(f" h (eV s)  | {         H         } | {        dH         } |")
    print(f"  φ (eV)   | {         φ         }   | {        dφ         }   |")
    print( "-----------|-----------------------|-----------------------|")
    print(f"hc (eV nm) | {        H_C        } | {       dH_C        } |")




# Main function
def main():

    # Plot style
    plt.style.use('dark_background')

    ### Individual Plots ###

    # # Plot the higher ultraviolet data
    # plot_file("photoelectric_effect/3131A.txt", color="magenta", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/3131A.txt")
    # plt.show()

    # # Plot the lower ultraviolet data
    # plot_file("photoelectric_effect/3655A.txt", color="white", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/3655A.txt")
    # plt.show()

    # # Plot the violet light data
    # plot_file("photoelectric_effect/4047A.txt", color="purple", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/4047A.txt")
    # plt.show()

    # # Plot the blue light data
    # plot_file("photoelectric_effect/4358A.txt", color="blue", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/4358A.txt")
    # plt.show()

    # # Plot the green light data
    # plot_file("photoelectric_effect/5461A.txt", color="green", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/5461A.txt")
    # plt.show()

    # # Plot the yellow light data
    # plot_file("photoelectric_effect/5779A.txt", color="yellow", gradient=False,
    #           show_bound=True, show_regression=True, show_points=True)
    # calc_file("photoelectric_effect/5779A.txt")
    # plt.show()


    
    # Plot the violet light data
    plot_file("photoelectric_effect/4047A.txt", color="red", gradient=False,
              show_bound=True, show_regression=True, show_points=True)
    plot_file("photoelectric_effect/D0.3_4047A.txt", color="green", gradient=False,
              show_bound=True, show_regression=True, show_points=True)
    plot_file("photoelectric_effect/D0.5_4047A.txt", color="blue", gradient=False,
                show_bound=True, show_regression=True, show_points=True)
    plt.show()
    







    plot_final_graph(mergedata=False)
    plt.show()


    # Calculate all the files
    files_list = [
        "photoelectric_effect/3131A.txt",
        "photoelectric_effect/3655A.txt",
        "photoelectric_effect/4047A.txt",
        "photoelectric_effect/4358A.txt",
        "photoelectric_effect/5461A.txt",
        "photoelectric_effect/5779A.txt",
        # "photoelectric_effect/D0.3_4047A.txt",
        # "photoelectric_effect/D0.5_4047A.txt",
        # "photoelectric_effect/low_5461A.txt",
        # "photoelectric_effect/low_5779A.txt"
    ]
    wavelength_list = [313.1, 365.5, 404.7, 435.8, 546.1, 577.9, 546.1, 577.9, 404.7, 404.7, 546.1, 577.9]

    calc_files(files_list, wavelength_list, save_linear=True, save_sigma=True)
    

    

# Main guard
if __name__ == "__main__":
    main()



