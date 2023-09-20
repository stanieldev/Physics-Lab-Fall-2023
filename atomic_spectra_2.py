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

<<<<<<< HEAD
<<<<<<< HEAD
# Iterate through the data
for i in range(30):
    iterate_one_step(data)
=======
=======
>>>>>>> parent of acae494 (Update atomic_spectra_2.py)
# Plot high saturation data
plt.figure()
plt.plot(high_on[:,0], high_on[:,1], label='on')
plt.plot(high_off[:,0], high_off[:,1], label='off')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (arb. units)')
plt.title('High Saturation')
plt.legend()
plt.show()

# # Plot medium saturation data
# plt.figure()
# plt.plot(med_on[:,0], med_on[:,1], label='on')
# plt.plot(med_off[:,0], med_off[:,1], label='off')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity (arb. units)')
# plt.title('Medium Saturation')
# plt.legend()
# plt.show()

# # Plot low saturation data
# plt.figure()
# plt.plot(low_on[:,0], low_on[:,1], label='on')
# plt.plot(low_off[:,0], low_off[:,1], label='off')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity (arb. units)')
# plt.title('Low Saturation')
# plt.legend()
# plt.show()



'''High confidence 
λ = ~410nm 
λ = ~434nm 
λ = ~487nm 
λ = ~651nm (Interferes too much with 656nm) 
λ = ~656nm 
λ = ~775nm 
λ = ~843nm 

Low confidence 
λ = ~357nm 
λ = ~580nm 
λ = ~614nm 
λ = ~926nm
'''


def plot_around_region(data, wavelength):

    # Plot data around a certain wavelength
    plot_data = data[(data[:,0] > wavelength - 10) & (data[:,0] < wavelength + 10)]
    plt.figure()
    plt.plot(plot_data[:,0], plot_data[:,1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Wavelength region around ' + str(wavelength) + 'nm')

    # Fit gaussian to data
    popt, pcov = curve_fit(gaussian_fit, plot_data[:,0], plot_data[:,1], p0=[1, wavelength, 1])

    # Print fit parameters with uncertainties
    # print('a = ' + str(popt[0]) + ' +/- ' + str(np.sqrt(pcov[0,0])))
    print('λ = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1,1])))
    print('σ = ' + str(popt[2]) + ' +/- ' + str(np.sqrt(pcov[2,2])))

    # Plot gaussian fit
    plt.plot(plot_data[:,0], gaussian_fit(plot_data[:,0], *popt), label='Gaussian fit')
    plt.legend()

    # Calculate r^2
    residuals = plot_data[:,1] - gaussian_fit(plot_data[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('r^2 = ' + str(r_squared))

    # Show plot
    plt.show()

def double_plot_around_region(data, wavelength1, wavelength2):

    # Plot data around a certain wavelength
    plot_data = data[(data[:,0] > wavelength1 - 20) & (data[:,0] < wavelength2 + 20)]
    plt.figure()
    plt.plot(plot_data[:,0], plot_data[:,1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Wavelength region around ' + str(wavelength1) + 'nm and ' + str(wavelength2) + 'nm')

    # Fit double gaussian to data
    popt, pcov = curve_fit(double_gaussian_fit, plot_data[:,0], plot_data[:,1], p0=[1, wavelength1, 1, 1, wavelength2, 1])

    # Print fit parameters with uncertainties
    print('λ1 = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1,1])))
    print('σ1 = ' + str(popt[2]) + ' +/- ' + str(np.sqrt(pcov[2,2])))
    print('λ2 = ' + str(popt[4]) + ' +/- ' + str(np.sqrt(pcov[4,4])))
    print('σ2 = ' + str(popt[5]) + ' +/- ' + str(np.sqrt(pcov[5,5])))


    # Plot gaussian fit
    plt.plot(plot_data[:,0], double_gaussian_fit(plot_data[:,0], *popt), label='Gaussian fit')
    plt.legend()

    # Calculate r^2
    residuals = plot_data[:,1] - double_gaussian_fit(plot_data[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('r^2 = ' + str(r_squared))

    # Show plot
    plt.show()


<<<<<<< HEAD
double_plot_around_region(high_on, 651, 656)
>>>>>>> parent of acae494 (Update atomic_spectra_2.py)
=======
double_plot_around_region(high_on, 651, 656)
>>>>>>> parent of acae494 (Update atomic_spectra_2.py)
