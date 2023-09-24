import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Gaussian fit function
def gaussian(x, A, μ, σ):
    return A*np.exp(-(x-μ)**2/(2*σ**2))

# Load data
high_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_high_saturation.txt', skiprows=1)
high_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_high_saturation.txt', skiprows=1)
med_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_med_saturation.txt', skiprows=1)
med_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_med_saturation.txt', skiprows=1)
low_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_low_saturation.txt', skiprows=1)
low_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_low_saturation.txt', skiprows=1)

# Plot high saturation data
plt.figure()
plt.plot(high_on[:,0], high_on[:,1], label='on')
plt.plot(high_off[:,0], high_off[:,1], label='off')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (arb. units)')
plt.title('High Saturation')
plt.legend()
plt.show()









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



def calc_with_interval(data, wavelength, interval):
    
    # Fit gaussian to data
    plot_data = data[(data[:,0] > wavelength - interval) & (data[:,0] < wavelength + interval)]
    popt, pcov = curve_fit(gaussian, plot_data[:,0], plot_data[:,1], p0=[1, wavelength, 1])

    # Calculate r^2
    residuals = plot_data[:,1] - gaussian(plot_data[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Return popt, pcov, and interval
    return [popt, pcov, r_squared, interval]



def plot_around_region(data, wavelength):

    # Create figure
    plt.figure()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Wavelength region around ' + str(wavelength) + 'nm')

    # Fit gaussian to data

    # Possible candidates
    candidates = []
    for i in range(1, 20):
        candidates.append(calc_with_interval(data, wavelength, i))
    
    # Find best r squared value
    best_r_squared = 0
    best_candidate = None
    for candidate in candidates:
        if candidate[2] > best_r_squared:
            best_r_squared = candidate[2]
            best_candidate = candidate
    
    # Unpack best candidate
    popt, pcov, r_squared, interval = best_candidate

    # Plot data around a certain wavelength
    plot_data = data[(data[:,0] > wavelength - interval) & (data[:,0] < wavelength + interval)]
    plt.plot(plot_data[:,0], plot_data[:,1])

    # Print fit parameters with uncertainties
    # print('a = ' + str(popt[0]) + ' +/- ' + str(np.sqrt(pcov[0,0])))
    print('λ = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1,1])))
    # print('σ = ' + str(popt[2]) + ' +/- ' + str(np.sqrt(pcov[2,2])))
    print('r^2 = ' + str(r_squared))

    # Plot gaussian fit
    plt.plot(plot_data[:,0], gaussian(plot_data[:,0], *popt), label='Gaussian fit')
    plt.legend()

    # Show plot
    plt.show()



plot_around_region(med_on, 486)
plot_around_region(low_on, 656)