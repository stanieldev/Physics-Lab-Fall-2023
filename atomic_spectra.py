import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import scipy.optimize as opt
from enum import Enum
import numpy as np

# Constants
D = 1/600    # 600 lines per mm
D = D*10**6  # Convert to nanometers^-1



# Color enum
class Color(Enum):
    Pink = 0
    Violet = 1
    Blue_Violet = 2
    Blue_Green = 3
    Red = 4
    def toLambda(self):
        EXPECTED = [410.1734, 434.0472, 486.135, 656.279]
        if self.value == 0: return 0
        return EXPECTED[self.value - 1]
    def toRGB(self):
        if self.value == 0: return 255/256, 192/256, 203/256
        return Color.RGB(self.toLambda())
    def RGB(λ: float):

        # This is a port of javascript code from  http://stackoverflow.com/a/14917481
        gamma = 0.8
        intensity_max = 1

        if λ < 380:
            red, green, blue = 0, 0, 0
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
            red, green, blue = 0, 0, 0
    
        # let the intensity fall of near the vision limits
        if λ < 380:
            factor = 0
        elif λ < 420:
            factor = 0.3 + 0.7 * (λ - 380) / (420 - 380)
        elif λ < 700:
            factor = 1
        elif λ <= 780:
            factor = 0.3 + 0.7 * (780 - λ) / (780 - 700)
        else:
            factor = 0
    
        def _f(c):
            if c == 0:
                return 0
            else:
                return intensity_max * pow (c * factor, gamma)
    
        return _f(red), _f(green), _f(blue)

def f(label: str, x: float, dx: float, units=None) -> str:
    dx = np.ceil(dx * 1000) / 1000
    u = f" {units}" if units is not None else ""
    return f"{label} +/- δ{label} = {x:.3f}{u} +/- {dx:.3f}{u}"

def r2(r_squared: float):
    r_squared = np.floor(r_squared * 1_000_000) / 1_000_000
    return f"(r²={r_squared*100:.4f}%)"

def constrain(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x



# Calculation functions
def calculate_wavelengths(orders: list[int], angles: list[float], colors: list[Color]):

    # Constants
    D = 1/600    # 600 lines per mm
    D = D*10**6  # Convert to nanometers^-1

    # Define linear regression function
    def linear_regression(x, a): return a * x

    # Calculate regression for each color
    ordered_pairs = []
    for selected_color in list(Color)[1:]:
        color_data = np.array([[order, angle] for order, angle, color in zip(orders, angles, colors) if color == selected_color])
        param, covariance = opt.curve_fit(linear_regression, color_data[:, 0], color_data[:, 1])
        param_error = np.sqrt(np.diag(covariance))
        ordered_pairs.append([D * param, D * param_error, selected_color])

    # Return ordered pairs
    print(ordered_pairs)
    return ordered_pairs

def calculate_around_region(data, λ, dλ=10) -> tuple[float, float]:

    # Gaussian function
    def gaussian(x, A, μ, σ): return A*np.exp(-(x-μ)**2/(2*σ**2))

    # Plot data around a certain λ
    subset = data[(data[:,0] > λ - dλ) & (data[:,0] < λ + dλ)]
    wavelengths = subset[:,0]
    intensities = subset[:,1]

    # Fit gaussian to data
    popt, pcov = opt.curve_fit(gaussian, wavelengths, intensities, p0=[1, λ, 1])
    # residuals = intensities - gaussian(wavelengths, *popt)
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((intensities-np.mean(intensities))**2)
    # r_squared = 1 - (ss_res / ss_tot)

    # Return gaussian parameters
    return popt[1], np.sqrt(pcov[1,1])



# Plotting functions
def plot_diffraction_spectrum(plot: plt, angles: list[float], colors: list[Color]) -> None:

    # Set up plot
    plot.get_yaxis().set_visible(False)
    plot.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plot.set_facecolor('black')
    plot.set_xlim(-90, 90)
    plot.set_xticks(np.arange(-80, 90, 20))

    # Plot vertical lines
    [plot.axvline(x = angle, color = color, label = 'axvline - full height') for angle, color in zip(angles, np.array([color.toRGB() for color in colors]))]

def plot_colors(plot: plt, orders: list[int], angles: list[float], colors: list[Color]):

    # Plot points by color
    [plot.errorbar(x = order, y = angle, color = color.toRGB(), marker = 'o', linestyle = 'None', capsize=5, capthick=1) for order, angle, color in zip(orders, angles, colors)]

    # Define linear regression function
    def linear_regression(x, a): return a * x

    # Calculate regression for each color
    for selected_color in list(Color)[1:]:
        color_data = np.array([[order, angle] for order, angle, color in zip(orders, angles, colors) if color == selected_color])
        param, covariance = opt.curve_fit(linear_regression, color_data[:, 0], color_data[:, 1])
        param_error = np.sqrt(np.diag(covariance))
        residuals = color_data[:, 1] - linear_regression(color_data[:, 0], *param)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((color_data[:, 1] - np.mean(color_data[:, 1]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        # plot.plot(color_data[:, 0], linear_regression(color_data[:, 0], *param), label = f("s", param[0], param_error[0]) + " " + r2(r_squared), color = selected_color.toRGB())
        plot.plot(color_data[:, 0], linear_regression(color_data[:, 0], *param), label = "", color = selected_color.toRGB())
        # plot.legend()
        # Print color, slope, slope error, and r²
        print(f"{selected_color.name}:\t{param[0]:.3f} +/- {param_error[0]:.3f}\t{r_squared*100:.4f}%")

        
    # Finalize plot
    # plot.legend()

def plot_wavelengths(plot: plt, orders: list[int], angles: list[float], colors: list[Color]):

    # Calculate wavelengths
    ordered_pairs = calculate_wavelengths(orders, angles, colors)

    # Set up plot
    plot.get_yaxis().set_visible(False)

    # Plot lines by color λ = D * s
    [plot.axvline(x = λ, color = color.toRGB(), label = f("λ", λ[0], dλ[0], "nm")) for λ, dλ, color in ordered_pairs]
    [plot.errorbar(x = λ, y = 0, xerr=dλ, color = color.toRGB(), marker = None, linestyle = 'None', capsize=5, capthick=1) for λ, dλ, color in ordered_pairs]
    [plot.axvline(x = λ + dλ, color = color.toRGB(), linestyle='dashed') for λ, dλ, color in ordered_pairs]
    [plot.axvline(x = λ - dλ, color = color.toRGB(), linestyle='dashed') for λ, dλ, color in ordered_pairs]

    # Finalize plot
    plot.legend()

def plot_rydberg_constant(plot: plt, n: list[float], λ: list[float], dλ: list[float]) -> None:

    # Modify data
    n = np.array([1 / (_n**2) for _n in n])
    dλ = np.array([_λ / _dλ**2 for _λ, _dλ in zip(dλ, λ)])
    λ = np.array([1 / _λ for _λ in λ])

    # Plot points
    colors = [Color.RGB(1/_λ) for _λ in λ]
    [plot.errorbar(n, λ, yerr=dλ, marker = 'o', linestyle = 'None', capsize=5, capthick=1, color=color) for n, λ, dλ, color in zip(n, λ, dλ, colors)]

    # Define linear regression function
    def linear_regression(x, a, b): return a * x + b

    # Calculate regression
    params, covariance = opt.curve_fit(linear_regression, n, λ, sigma=dλ)
    param_errors = np.sqrt(np.diag(covariance))
    residuals = λ - linear_regression(n, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((λ - np.mean(λ))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Initialize plot
    plot.title.set_text("1/λ vs 1/n²")
    plot.set_xlabel("1/n²")
    plot.set_ylabel("1/λ (nm^-1)")
    plot.set_facecolor('black')

    # Plot regression
    plot.plot(n, linear_regression(n, *params), label = r2(r_squared), color='white')

    # Add statistics to legend
    handles, labels = plt.gca().get_legend_handles_labels()
    patches = [
        mpatches.Patch(color='none', label='Slope Label'), 
        mpatches.Patch(color='none', label='Intercept Label'),
        mpatches.Patch(color='none', label='Rydberg Constant Label 1'),
        mpatches.Patch(color='none', label='Rydberg Constant Label 2')
    ]
    [handles.append(patch) for patch in patches]
    labels.append(f("s", params[0]*1000, param_errors[0]*1000, "pm^-1"))
    labels.append(f("b", params[1]*1000, param_errors[1]*1000, "pm^-1"))
    labels.append(f"Ry_s = {-params[0]*1000:.3f} +/- {param_errors[0]*1000:.3f}pm^-1")
    labels.append(f"Ry_b = {params[1]*4000:.3f} +/- {param_errors[1]*4000:.3f}pm^-1")
    
    # Finalize plot
    plot.legend(handles, labels)

def plot_emission_spectrum(plot: plt, λ: np.ndarray, intensity: np.ndarray, sigma=lambda x: x) -> None:
    
    # Initialize plot
    plot.set_ylim(0, 1.1)
    plot.set_ylabel("sqrt(Relative Intensity)")
    plot.set_facecolor('black')

    # Normalize intensity
    intensity /= max(intensity)

    # Remove negative intensities
    λ = λ[np.where(intensity >= 0)]
    intensity = intensity[np.where(intensity >= 0)]

    # Plot data
    [plot.axvline(_λ, color=Color.RGB(_λ), alpha=constrain(sigma(I), 0, 1)) for _λ, I in zip(λ, intensity)]



def temp():
    # Create figure with 4 subplots in a 2x2 grid
    fig, ax4 = plt.subplots(1, 1)
    fig.patch.set_facecolor((10.6/255, 12.9/255, 17.3/255))
    fig.subplots_adjust(hspace=0.35)

    # Import manual spectrometer data
    data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0] - data[0][0]  # Set first angle to 0
    diffraction_angles = np.sin(diffraction_angles * np.pi / 180)  # Convert to sin(angle)
    diffraction_orders = data[3]  # Get diffraction order
    diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # Calculate wavelengths
    ordered_pairs = calculate_wavelengths(diffraction_orders, diffraction_angles, diffraction_colors)

    # Find the wavelengths of the Balmer series
    n = [6, 5, 4, 3]
    λ = np.array([i[0][0] for i in ordered_pairs])
    dλ = np.array([i[1][0] for i in ordered_pairs])

    # Plot 1/λ vs 1/n²
    plot_rydberg_constant(ax4, n, λ, dλ)
    plt.show()




# Collections of data
def manual_spectroscopy():

    # Create figure with 4 subplots in a 2x2 grid
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    fig.patch.set_facecolor((10.6/255, 12.9/255, 17.3/255))
    fig.subplots_adjust(hspace=0.35)


    # Import manual spectrometer data
    data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0] + data[1]/60  # Add arc seconds to degrees
    diffraction_angles = diffraction_angles - diffraction_angles[0]  # Set first angle to 0
    diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # Plot the measured diffraction pattern
    ax1.title.set_text("Manually Measured Atomic Spectrum of Hydrogen")
    plot_diffraction_spectrum(ax1, diffraction_angles, diffraction_colors)


    # Import theoretical spectrometer data
    data = np.loadtxt("./atomic_spectra/H_theory.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0]
    diffraction_colors = [Color(int(i)) for i in data[1]]

    # Plot the theoretical diffraction pattern
    ax2.title.set_text("Theoretical Atomic Spectrum of Hydrogen")
    plot_diffraction_spectrum(ax2, diffraction_angles, diffraction_colors)


    # Import manual spectrometer data
    data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0] - data[0][0]  # Set first angle to 0
    diffraction_angles = np.sin(diffraction_angles * np.pi / 180)  # Convert to sin(angle)
    diffraction_orders = data[3]  # Get diffraction order
    diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # Plot the sin(angle) vs diffraction order
    ax3.title.set_text("sin(Angle) vs Diffraction Order")
    ax3.set_xlabel("Diffraction Order")
    ax3.set_ylabel("sin(Angle)")
    ax3.set_facecolor('black')
    plot_colors(ax3, diffraction_orders, diffraction_angles, diffraction_colors)


    # # Import manual spectrometer data
    # data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # # Modify data to proper sets
    # diffraction_angles = data[0] - data[0][0]  # Set first angle to 0
    # diffraction_angles = np.sin(diffraction_angles * np.pi / 180)  # Convert to sin(angle)
    # diffraction_orders = data[3]  # Get diffraction order
    # diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # # Plot the wavelengths of the hydrogen spectrum
    # ax4.title.set_text("Wavelengths of Hydrogen Spectrum")
    # ax4.set_xlabel("Wavelengths (nm)")
    # ax4.set_facecolor('black')
    # plot_wavelengths(ax4, diffraction_orders, diffraction_angles, diffraction_colors)


    # Import manual spectrometer data
    data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0] - data[0][0]  # Set first angle to 0
    diffraction_angles = np.sin(diffraction_angles * np.pi / 180)  # Convert to sin(angle)
    diffraction_orders = data[3]  # Get diffraction order
    diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # Calculate wavelengths
    ordered_pairs = calculate_wavelengths(diffraction_orders, diffraction_angles, diffraction_colors)

    # Find the wavelengths of the Balmer series
    n = [6, 5, 4, 3]
    λ = np.array([i[0][0] for i in ordered_pairs])
    dλ = np.array([i[1][0] for i in ordered_pairs])

    # Plot 1/λ vs 1/n²
    plot_rydberg_constant(ax4, n, λ, dλ)


    # Show the plots
    plt.show()

def machine_spectroscopy():

    # Create figure with 2 subplots
    fig, (ax1) = plt.subplots(1,1)
    fig.subplots_adjust(hspace=0.35)



    # # Import machine spectrometer data
    # high_data = np.loadtxt('./atomic_spectra/H_high_off.txt', skiprows=1)
    # low_data = np.loadtxt('./atomic_spectra/H_low_on.txt', skiprows=1)

    # # Plot data around a certain λ
    # EXPECTED = [656.279, 486.135, 434.0472, 410.1734, 397.0075, 388.9064]
    # λ3, dλ3 = calculate_around_region(low_data, EXPECTED[0], dλ=10)
    # λ4, dλ4 = calculate_around_region(high_data, EXPECTED[1], dλ=10)
    # λ5, dλ5 = calculate_around_region(high_data, EXPECTED[2], dλ=9)
    # λ6, dλ6 = calculate_around_region(high_data, EXPECTED[3], dλ=7)
    # λ7, dλ7 = calculate_around_region(high_data, EXPECTED[4], dλ=5)
    # λ8, dλ8 = calculate_around_region(high_data, EXPECTED[5], dλ=3)

    # # Plot 1/λ vs 1/n²
    # plot_rydberg_constant(ax1, [3, 4, 5, 6, 7, 8], [λ3, λ4, λ5, λ6, λ7, λ8], [dλ3, dλ4, dλ5, dλ6, dλ7, dλ8])



    # Import machine spectrometer data
    data = np.loadtxt('./atomic_spectra/H_high_on.txt', skiprows=1)

    # Split data into wavelength and intensity
    wavelengths = data[:,0]
    intensities = data[:,1]

    # Plot the emission spectrum
    ax1.title.set_text("Emission Spectrum of Hydrogen")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_xlim(380, 780)
    plot_emission_spectrum(ax1, wavelengths, intensities, sigma=lambda x: np.sqrt(x))



    # # Import machine spectrometer data
    # data = np.loadtxt('./atomic_spectra/He.txt', skiprows=1)

    # # Split data into wavelength and intensity
    # wavelengths = data[:,0]
    # intensities = data[:,1]

    # # Plot the emission spectrum
    # ax1.title.set_text("Emission Spectrum of Helium")
    # ax1.set_xlabel("Wavelength (nm)")
    # ax1.set_xlim(380, 780)
    # plot_emission_spectrum(ax1, wavelengths, intensities, sigma=lambda x: np.sqrt(x))



    # # Import machine spectrometer data
    # data = np.loadtxt('./atomic_spectra/Ne_med.txt', skiprows=1)

    # # Split data into wavelength and intensity
    # wavelengths = data[:,0]
    # intensities = data[:,1]

    # # Plot the emission spectrum
    # ax1.title.set_text("Emission Spectrum of Neon")
    # ax1.set_xlabel("Wavelength (nm)")
    # ax1.set_xlim(380, 780)
    # plot_emission_spectrum(ax1, wavelengths, intensities)



    plt.show()


if __name__ == "__main__":
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#181818'
    # plt.rcParams['text.color'] = 'white'
    # # set tickmarks and labels to white
    # plt.rcParams['xtick.color'] = 'white'
    # plt.rcParams['ytick.color'] = 'white'
    # # set axes names to white
    # plt.rcParams['axes.labelcolor'] = 'white'
    # # set legend color to white
    # plt.rcParams['legend.facecolor'] = (10.6/255, 12.9/255, 17.3/255)

    fig, ax1 = plt.subplots(1, 1)    

    # Import manual spectrometer data
    data = np.loadtxt("./atomic_spectra/H_manual.txt", unpack=True)

    # Modify data to proper sets
    diffraction_angles = data[0] + data[1]/60  # Add arc seconds to degrees
    diffraction_angles = diffraction_angles - diffraction_angles[0]  # Set first angle to 0
    diffraction_colors = [Color(int(i)) for i in data[2]]  # Convert color index to Color enum

    # Plot the measured diffraction pattern
    ax1.title.set_text("Manually Measured Atomic Spectrum of Hydrogen")
    plot_diffraction_spectrum(ax1, diffraction_angles, diffraction_colors)
    plt.show()
    

    
    # manual_spectroscopy()
    # machine_spectroscopy()
    pass
