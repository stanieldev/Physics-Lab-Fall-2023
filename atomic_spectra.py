import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
import scipy.stats as stats
from typing import Callable
import math

# Constants
dTheta = 0.5 / 60  # sin(x) ~ x for small values of theta
D = 1/600    # 600 lines per mm
D = D*10**6  # Convert to nanometers^-1

# Color dictionary
COLOR_INDEX_TO_HEX = {
    "0": "#FFC0CB",  # Pink
    "1": "#7d00db",  # Violet
    "2": "#2800ff",  # Blue-Violet
    "3": "#00efff",  # Blue-Green
    "4": "#ff0000"   # Red
}

# Data point class
class DataPoint:
    def __init__(self, x, y, x_err, y_err, color) -> None:
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err
        self.color = color
    def __repr__(self) -> str:
        return f"DataPoint({self.x}, {self.y}, {self.x_err}, {self.y_err}, {self.color})"

# Regression lines class
class RegressionFunctions:
    class Linear:
        def __eval__(x, a):
            return a * x
        def __form__() -> str:
            return f"ax"
    class FullLinear:
        def __eval__(x, a, b):
            return a * x + b
        def __form__() -> str:
            return f"ax + b"

# Points class
class Points:
    def __init__(self, data):
        # All data
        self.data = data

        # Sorted by color
        self.Pink = self.data.transpose()[self.data.transpose()[:, 1] == 0]
        self.Violet = self.data.transpose()[self.data.transpose()[:, 1] == 1]
        self.Blue = self.data.transpose()[self.data.transpose()[:, 1] == 2]
        self.Cyan = self.data.transpose()[self.data.transpose()[:, 1] == 3]
        self.Red = self.data.transpose()[self.data.transpose()[:, 1] == 4]




# Import data
path = "./atomic_spectra/manual_spectrometer.txt"
data = np.loadtxt(path, unpack=True)

# Modify data to proper sets
data[0] = data[0] + data[1]/60  # Add arc seconds to degrees
data = np.delete(data, 1, 0)    # Remove arc seconds column
data[0] -= data[0][0]           # Set first angle to 0 degrees

# Create all points lists
POINTS = Points(data)

# Create figure with 4 subplots in a 2x2 grid
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
fig.patch.set_facecolor('grey')





# Plot the measured diffraction pattern
def manually_measured_diffraction_plot(plot: plt) -> None:

    # Set up plot
    plot.get_yaxis().set_visible(False)
    plot.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plot.set_facecolor('black')
    plot.title.set_text("Manually Measured Atomic Spectrum of Hydrogen")

    # Define points as lists
    diffraction_angles = data[0]
    spectral_color = np.array([COLOR_INDEX_TO_HEX[str(int(i))] for i in data[1]])

    # Plot lines
    [plot.axvline(x = angle, color = color, label = 'axvline - full height') for angle, color in zip(diffraction_angles, spectral_color)]

manually_measured_diffraction_plot(ax1)


# Plot the theoretical diffraction pattern
def theoretical_diffraction_plot(plot: plt) -> None:

    # Set up plot
    plot.get_yaxis().set_visible(False)
    plot.set_facecolor('black')
    plot.title.set_text("Theoretical Atomic Spectrum of Hydrogen")
    plot.get_shared_x_axes().join(plot, ax1)

    # Known wavelengths of light
    violet_wavelength = 410.2
    blue_violet_wavelength = 434.0
    blue_green_wavelength = 486.1
    red_wavelength = 656.3

    # Unrefracted lines
    plot.axvline(x = 0, color = COLOR_INDEX_TO_HEX["0"], label = 'axvline - full height')

    # Violet lines
    violet_angles = [np.arcsin(violet_wavelength*m/D)*180/np.pi for m in range(1, math.floor(D/violet_wavelength) + 1)]
    [plot.axvline(x = angle, color = COLOR_INDEX_TO_HEX["1"], label = 'axvline - full height') for angle in violet_angles]
    [plot.axvline(x = -angle, color = COLOR_INDEX_TO_HEX["1"], label = 'axvline - full height') for angle in violet_angles]

    # Blue-Violet lines
    blue_violet_angles = [np.arcsin(blue_violet_wavelength*m/D)*180/np.pi for m in range(1, math.floor(D/blue_violet_wavelength) + 1)]
    [plot.axvline(x = angle, color = COLOR_INDEX_TO_HEX["2"], label = 'axvline - full height') for angle in blue_violet_angles]
    [plot.axvline(x = -angle, color = COLOR_INDEX_TO_HEX["2"], label = 'axvline - full height') for angle in blue_violet_angles]

    # Blue-Green lines
    blue_green_angles = [np.arcsin(blue_green_wavelength*m/D)*180/np.pi for m in range(1, math.floor(D/blue_green_wavelength) + 1)]
    [plot.axvline(x = angle, color = COLOR_INDEX_TO_HEX["3"], label = 'axvline - full height') for angle in blue_green_angles]
    [plot.axvline(x = -angle, color = COLOR_INDEX_TO_HEX["3"], label = 'axvline - full height') for angle in blue_green_angles]

    # Red lines
    red_angles = [np.arcsin(red_wavelength*m/D)*180/np.pi for m in range(1, math.floor(D/red_wavelength) + 1)]
    [plot.axvline(x = angle, color = COLOR_INDEX_TO_HEX["4"], label = 'axvline - full height') for angle in red_angles]
    [plot.axvline(x = -angle, color = COLOR_INDEX_TO_HEX["4"], label = 'axvline - full height') for angle in red_angles]

theoretical_diffraction_plot(ax2)


# Plot the sin(angle) vs diffraction order
def sine_angle_vs_diffraction_order(plot: plt) -> None:

    # Set up plot
    plot.title.set_text("sin(Angle) vs Diffraction Order")
    plot.set_ylabel("sin(Angle)")
    plot.set_xlabel("Diffraction Order")

    # Define points as lists
    sine_diffraction_angle = np.sin(data[:, 0] * np.pi / 180)
    diffraction_order = data[:, 2]
    spectral_color = [matplotlib.colors.to_rgb(COLOR_INDEX_TO_HEX[str(int(i))]) for i in data[1]]

    # Plot points by color
    plot.errorbar(x = 0, y = 0, yerr=dTheta, color = matplotlib.colors.to_rgb(COLOR_INDEX_TO_HEX[str(int(0))]), marker = 'o', linestyle = 'None')
    # [plot.errorbar(x = order, y = sine_angle, yerr=dTheta, color = color, marker = 'o', linestyle = 'None') for order, sine_angle, color in zip(diffraction_order, sine_diffraction_angle, spectral_color)]

    # Plot each wavelength of light
    plot_sine_dataset(plot, POINTS.Violet, "Violet")
    plot_sine_dataset(plot, POINTS.Blue, "Blue-Violet")
    plot_sine_dataset(plot, POINTS.Cyan, "Blue-Green")
    plot_sine_dataset(plot, POINTS.Red, "Red")
    plot.legend()

saved_data = []
def plot_sine_dataset(plot: plt, datum, color_str: str = None) -> tuple:

    # Define points as lists
    sine_diffraction_angle = np.sin(datum[:, 0] * np.pi / 180)
    diffraction_order = datum[:, 2]
    spectral_color = [matplotlib.colors.to_rgb(COLOR_INDEX_TO_HEX[str(int(i))]) for i in datum[:, 1]]

    # Plot points by color
    [plot.errorbar(x = order, y = sine_angle, yerr=dTheta, color = color, marker = 'o', linestyle = 'None') for order, sine_angle, color in zip(diffraction_order, sine_diffraction_angle, spectral_color)]

    # Calculate regression
    params, covariance = opt.curve_fit(RegressionFunctions.Linear.__eval__, diffraction_order, sine_diffraction_angle, sigma=np.full(len(datum), dTheta))
    param_errors = np.sqrt(np.diag(covariance))

    # Plot regression line
    plot.plot(diffraction_order, RegressionFunctions.Linear.__eval__(diffraction_order, *params), label = f"s +/- δs = {params[0]:.3f} +/- {param_errors[0]:.3f}", color = COLOR_INDEX_TO_HEX[str(int(datum[0, 1]))])

    # Calculate statistics
    residuals = sine_diffraction_angle - RegressionFunctions.Linear.__eval__(diffraction_order, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sine_diffraction_angle - np.mean(sine_diffraction_angle))**2)

    # Calculate probabilities
    r_squared = 1 - (ss_res / ss_tot)

    # Prepare for printing
    r_squared = math.floor(r_squared * 10000) / 10000

    # Print statistics
    print(f"{color_str} Statistics:")
    print(f"s +/- δs = {params[0]:.3f} +/- {param_errors[0]:.3f}")
    print(f"λ +/- δλ = {D * params[0]:.3f}nm +/- {D * param_errors[0]:.3f}nm")
    print(f"r² = {r_squared:.3f} ({r_squared*100:.2f}%)")

    # Save data
    saved_data.append((D * params[0], D * param_errors[0]))

sine_angle_vs_diffraction_order(ax3)


# Plot the wavelength as vertical bars with error bars
def wavelength_plot(plot: plt) -> None:
    
    # Set up plot
    plot.set_xlabel("Wavelength (nm)")
    plot.get_yaxis().set_visible(False)

    # Define points as lists
    wavelengths = np.array([i[0] for i in saved_data])
    errors = np.array([i[1] for i in saved_data])
    colors = [matplotlib.colors.to_rgb(COLOR_INDEX_TO_HEX[str(int(i))]) for i in range(1, 5)]

    # Plot lines by color
    [plot.axvline(x = wavelength, color = color, label = 'axvline - full height') for wavelength, color in zip(wavelengths, colors)]
    [plot.errorbar(x = wavelength, y = 0, xerr=error, color = color, marker = None, linestyle = 'None') for wavelength, error, color in zip(wavelengths, errors, colors)]
    [plot.axvline(x = wavelength + error, color = color, label = 'axvline - full height', linestyle='dashed') for wavelength, error, color in zip(wavelengths, errors, colors)]
    [plot.axvline(x = wavelength - error, color = color, label = 'axvline - full height', linestyle='dashed') for wavelength, error, color in zip(wavelengths, errors, colors)]


# wavelength_plot(ax4)


def rydberg_constant(plot: plt):

    # Set up plot
    plot.set_xlabel("1/n²")
    plot.set_ylabel("1 / Wavelength (nm^-1)")

    # Define points as lists
    x_list = [3,4,5,6]
    y_list = [660.589, 488.869, 440.637, 401.049]
    y_error = [3.467, 1.965, 3.882, 6.079]

    # Modify data
    x_list = np.array([1 / (i**2) for i in x_list])
    y_error = np.array([i / j**2 for i, j in zip(y_error, y_list)])
    y_list = np.array([1 / i for i in y_list])

    # Plot points
    plot.errorbar(x_list, y_list, yerr=y_error, marker = 'o', linestyle = 'None')

    # Calculate regression
    params, covariance = opt.curve_fit(RegressionFunctions.FullLinear.__eval__, x_list, y_list, sigma=y_error)
    param_errors = np.sqrt(np.diag(covariance))

    # Plot regression line
    plot.plot(x_list, RegressionFunctions.FullLinear.__eval__(x_list, *params), label = f"s +/- δs = {params[0]:.3f} +/- {param_errors[0]:.3f}")

    # Calculate statistics
    residuals = y_list - RegressionFunctions.FullLinear.__eval__(x_list, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_list - np.mean(y_list))**2)

    # Calculate probabilities
    r_squared = 1 - (ss_res / ss_tot)

    # Prepare for printing
    r_squared = math.floor(r_squared * 10000) / 10000

    # Print statistics
    RD = 0.0109737316
    print(f"Rydberg Statistics:")
    print(f"s +/- δs = {-params[0]:.4f}nm^-1 +/- {-param_errors[0]:.4f}nm^-1")
    print(f"b +/- δb = {4*params[1]:.4f}nm^-1 +/- {4*param_errors[1]:.4f}nm^-1")
    print(f"r² = {r_squared:.3f} ({r_squared*100:.2f}%)")
    print((-params[0] - RD)/(-param_errors[0]))

rydberg_constant(ax4)













# # Top right subplot


# # Error bars
# ax3.errorbar(pink_x, np.sin(pink_y * C), yerr=dTheta, color = matplotlib.colors.to_rgb(color_dict["0"]), marker = 'o', linestyle = 'None', label = "Pink")
# ax3.errorbar(violet_x, np.sin(violet_y * C), yerr=dTheta, color = matplotlib.colors.to_rgb(color_dict["1"]), marker = 'o', linestyle = 'None', label = "Violet")
# ax3.errorbar(blue_violet_x, np.sin(blue_violet_y * C), yerr=dTheta, color = matplotlib.colors.to_rgb(color_dict["2"]), marker = 'o', linestyle = 'None', label = "Blue-Violet")
# ax3.errorbar(blue_green_x, np.sin(blue_green_y * C), yerr=dTheta, color = matplotlib.colors.to_rgb(color_dict["3"]), marker = 'o', linestyle = 'None', label = "Blue-Green")
# ax3.errorbar(red_x, np.sin(red_y * C), yerr=dTheta, color = matplotlib.colors.to_rgb(color_dict["4"]), marker = 'o', linestyle = 'None', label = "Red")

# # Regression lines
# violet_params, violet_covariance = opt.curve_fit(RegressionFunctions.Linear.__eval__, violet_x, np.sin(violet_y * C), sigma=np.full(len(violet_points), dTheta))
# blue_violet_params, blue_violet_covariance = opt.curve_fit(RegressionFunctions.Linear.__eval__, blue_violet_x, np.sin(blue_violet_y * C), sigma=np.full(len(blue_violet_points), dTheta))
# blue_green_params, blue_green_covariance = opt.curve_fit(RegressionFunctions.Linear.__eval__, blue_green_x, np.sin(blue_green_y * C), sigma=np.full(len(blue_green_points), dTheta))
# red_params, red_covariance = opt.curve_fit(RegressionFunctions.Linear.__eval__, red_x, np.sin(red_y * C), sigma=np.full(len(red_points), dTheta))


# # Print regression line parameters
# print(f"Blue-Violet Slope: {blue_violet_params[0]:.3f} +/- {np.sqrt(np.diag(blue_violet_covariance))[0]:.3f}")
# print(f"Blue-Green Slope: {blue_green_params[0]:.3f} +/- {np.sqrt(np.diag(blue_green_covariance))[0]:.3f}")
# print(f"Red Slope: {red_params[0]:.3f} +/- {np.sqrt(np.diag(red_covariance))[0]:.3f}")
# print()
# blue_violet_wavelength = D * blue_violet_params[0]
# blue_green_wavelength = D * blue_green_params[0]
# red_wavelength = D * red_params[0]
# print(f"Blue-Violet Wavelength: {blue_violet_wavelength:.3f} +/- {np.sqrt(np.diag(blue_violet_covariance))[0] * D:.3f}")
# print(f"Blue-Green Wavelength: {blue_green_wavelength:.3f} +/- {np.sqrt(np.diag(blue_green_covariance))[0] * D:.3f}")
# print(f"Red Wavelength: {red_wavelength:.3f} +/- {np.sqrt(np.diag(red_covariance))[0] * D:.3f}")


# print(f"Violet Slope: {violet_params[0]:.3f} +/- {np.sqrt(np.diag(violet_covariance))[0]:.3f}")
# violet_wavelength = D * violet_params[0]
# print(f"Violet Wavelength: {violet_wavelength:.3f} +/- {np.sqrt(np.diag(violet_covariance))[0] * D:.3f}")










# # Calculate statistics
# fig_param_errors = np.sqrt(np.diag(covariance))
# chi_squared = np.sum(((independent_column.data - regression.__eval__(dependent_column.data, *fig_params)) / independent_error_column.data)**2)
# degrees_of_freedom = len(dependent_column.data) - len(fig_params)
# reduced_chi_squared = chi_squared / degrees_of_freedom

# # Calculate r^2
# residuals = independent_column.data - regression.__eval__(dependent_column.data, *fig_params)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((independent_column.data - np.mean(independent_column.data))**2)
# r_squared = 1 - (ss_res / ss_tot)

# # Calculate p-values
# p_value_chi_squared = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
# p_value_r_squared = 1 - stats.f.cdf(r_squared, 1, degrees_of_freedom)

# # Print statistics
# print(f"Regression form: {regression.__form__()}")
# [print(f"{fig_params[i]=:.3f} +/- {fig_param_errors[i]:.3f}") for i in range(len(fig_params))]
# print(f"χ² = {chi_squared:.3f} (p={p_value_chi_squared*100}%)")
# print(f"Degrees of Freedom = {degrees_of_freedom}")
# print(f"Reduced χ² = {reduced_chi_squared:.3f}")
# print(f"r² = {r_squared:.3f} (p={p_value_r_squared*100}%)")








 
# rendering plot
plt.show()



# plot data


# theta_error = DataColumn("Theta Error", "σ_θ", "degrees", data[2] + data[3]/60)
# wavelength = DataColumn("Wavelength", "λ", "nm", data[0])









# # Application class
# class Application:






#     def user_plot_scatter(self) -> None:

#         # Plot text objects
#         x_label = f"{dependent_column.name}\n{dependent_column.var}({dependent_column.unit})"
#         y_label = f"{independent_column.var}({independent_column.unit})\n{independent_column.name}"
#         title = f"{independent_column.name} vs. {dependent_column.name}"

#         # Create plot
#         fig = plt
#         fig.errorbar(dependent_column.data, independent_column.data, yerr=independent_error_column.data, capsize=5, marker='o', linestyle='None')
#         fig.xlabel(x_label)
#         fig.ylabel(y_label)
#         fig.title(title)

#         # Show figure
#         fig.show()
#         self.last_console_message = "Scatter plot plotted successfully!"    

#     def user_plot_regression(self) -> None:
        
#         # Print instructions
#         print("Scatter Plot w/ Regression Protocol")
#         print("> Enter the index of the columns you want to plot")
#         print("> Enter \"exit\" to go back to the menu\n")
#         self.list_available_columns()
        
#         # Query data
#         try:
#             dependent_column, independent_column, independent_error_column = self.user_query_scatterplot()
#             regression = self.user_query_regression()
#         except:
#             return

#         # Plot text objects
#         x_label = f"{dependent_column.name}\n{dependent_column.var}({dependent_column.unit})"
#         y_label = f"{independent_column.var}({independent_column.unit})\n{independent_column.name}"
#         title = f"{independent_column.name} vs. {dependent_column.name}"

#         # Create plot
#         fig = plt
#         fig.errorbar(dependent_column.data, independent_column.data, yerr=independent_error_column.data, capsize=5, marker='o', linestyle='None')
#         fig.xlabel(x_label)
#         fig.ylabel(y_label)
#         fig.title(title)


#         # Regression
#         fig_params, covariance = opt.curve_fit(regression.__eval__, dependent_column.data, independent_column.data, sigma = independent_error_column.data)
#         fig.plot(dependent_column.data, regression.__eval__(dependent_column.data, *fig_params), label = 'fit')
        
#         # Calculate statistics
#         fig_param_errors = np.sqrt(np.diag(covariance))
#         chi_squared = np.sum(((independent_column.data - regression.__eval__(dependent_column.data, *fig_params)) / independent_error_column.data)**2)
#         degrees_of_freedom = len(dependent_column.data) - len(fig_params)
#         reduced_chi_squared = chi_squared / degrees_of_freedom

#         # Calculate r^2
#         residuals = independent_column.data - regression.__eval__(dependent_column.data, *fig_params)
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((independent_column.data - np.mean(independent_column.data))**2)
#         r_squared = 1 - (ss_res / ss_tot)

#         # Calculate p-values
#         p_value_chi_squared = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
#         p_value_r_squared = 1 - stats.f.cdf(r_squared, 1, degrees_of_freedom)

#         # Print statistics
#         print(f"Regression form: {regression.__form__()}")
#         [print(f"{fig_params[i]=:.3f} +/- {fig_param_errors[i]:.3f}") for i in range(len(fig_params))]
#         print(f"χ² = {chi_squared:.3f} (p={p_value_chi_squared*100}%)")
#         print(f"Degrees of Freedom = {degrees_of_freedom}")
#         print(f"Reduced χ² = {reduced_chi_squared:.3f}")
#         print(f"r² = {r_squared:.3f} (p={p_value_r_squared*100}%)")


#         # Show figure
#         fig.show()
#         self.last_console_message = "Regression plotted successfully!"



#     def user_statistics(self) -> None:
        
#         # Validation check
#         if len(self.stored_data) == 0:
#             self.last_console_message = "Stored data is empty!"
#             return
        
#         # Print instructions
#         print("Scatter Plot w/ Regression Protocol")
#         print("> Enter the index of the columns you want to plot")
#         print("> Enter \"exit\" to go back to the menu\n")
#         self.list_available_columns()
        
#         # Query data
#         try:
#             dependent_column, independent_column, independent_error_column = self.user_query_scatterplot()
#             regression = self.user_query_regression()
#         except ReturnToMenuException:
#             return

#         # Plot text objects
#         x_label = f"{dependent_column.name}\n{dependent_column.var}({dependent_column.unit})"
#         y_label = f"{independent_column.var}({independent_column.unit})\n{independent_column.name}"
#         title = f"{independent_column.name} vs. {dependent_column.name}"

#         # Regression statistics
#         fig_params, covariance = opt.curve_fit(regression.__eval__, dependent_column.data, independent_column.data, sigma = independent_error_column.data)
#         fig_param_errors = np.sqrt(np.diag(covariance))
#         residuals = independent_column.data - regression.__eval__(dependent_column.data, *fig_params)
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((independent_column.data - np.mean(independent_column.data))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#         p_value_r_squared = 1 - stats.f.cdf(r_squared, 1, degrees_of_freedom)

#         # Chi squared statistics
#         chi_squared = np.sum(((independent_column.data - regression.__eval__(dependent_column.data, *fig_params)) / independent_error_column.data)**2)
#         degrees_of_freedom = len(dependent_column.data) - len(fig_params)
#         reduced_chi_squared = chi_squared / degrees_of_freedom
#         p_value_chi_squared = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)

#         # Print statistics
#         print(f"Regression form: {regression.__form__()}")
#         [print(f"{fig_params[i]=:.3f} +/- {fig_param_errors[i]:.3f}") for i in range(len(fig_params))]
#         print(f"r² = {r_squared:.3f} (p={p_value_r_squared*100}%)")
#         print(f"χ² = {chi_squared:.3f} (p={p_value_chi_squared*100}%)")
#         print(f"Reduced χ² = {reduced_chi_squared:.3f}")
#         print(f"Degrees of Freedom = {degrees_of_freedom}")

#         # Print success
#         self.last_console_message = "Regression plotted successfully!"
