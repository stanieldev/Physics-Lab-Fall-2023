import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt






def gaussian_fit(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def double_gaussian_fit(x, a1, b1, c1, a2, b2, c2):
    return a1*np.exp(-(x-b1)**2/(2*c1**2)) + a2*np.exp(-(x-b2)**2/(2*c2**2))




# Load data
high_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_high_saturation.txt', skiprows=1)
high_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_high_saturation.txt', skiprows=1)
med_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_med_saturation.txt', skiprows=1)
med_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_med_saturation.txt', skiprows=1)
low_on = np.loadtxt('./atomic_spectra/hydrogen_lights_on_low_saturation.txt', skiprows=1)
low_off = np.loadtxt('./atomic_spectra/hydrogen_lights_off_low_saturation.txt', skiprows=1)

# Create a function that finds the r^2 value from a wavelength and window size
def find_r_squared(data, wavelength, window_size):
    plot_data = data[(data[:,0] > wavelength - window_size) & (data[:,0] < wavelength + window_size)]
    popt, pcov = curve_fit(gaussian_fit, plot_data[:,0], plot_data[:,1], p0=[1, wavelength, 1])
    residuals = plot_data[:,1] - gaussian_fit(plot_data[:,0], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# For each wavelength in the input data, find the r^2 value and plot it against the wavelength
plt.figure()
def plot_r_squared(data, window_size):
    wavelengths = np.unique(data[:,0])
    r_squareds = []
    for wavelength in wavelengths:
        try:
            r2 = find_r_squared(data, wavelength, window_size)
            if r2 < 0:
                raise Exception
            r_squareds.append(r2)
        except:
            r_squareds.append(0)
    plt.plot(wavelengths, r_squareds, label = 'Window size = ' + str(window_size) + 'nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('r^2')
    
    


    
for window_size in [5, 10, 15, 20]:
    plot_r_squared(high_on, window_size)
    plot_r_squared(high_off, window_size)
    plot_r_squared(med_on, window_size)
    plot_r_squared(med_off, window_size)
    plot_r_squared(low_on, window_size)
    plot_r_squared(low_off, window_size)
    print("Finished window size " + str(window_size) + "nm")

plt.legend()
plt.show()












# # Plot high saturation data
# plt.figure()
# plt.plot(high_on[:,0], high_on[:,1], label='on')
# plt.plot(high_off[:,0], high_off[:,1], label='off')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity (arb. units)')
# plt.title('High Saturation')
# plt.legend()
# plt.show()

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





# def plot_around_region(data, wavelength):

#     # Plot data around a certain wavelength
#     plot_data = data[(data[:,0] > wavelength - 10) & (data[:,0] < wavelength + 10)]
#     plt.figure()
#     plt.plot(plot_data[:,0], plot_data[:,1])
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Intensity (arb. units)')
#     plt.title('Wavelength region around ' + str(wavelength) + 'nm')

#     # Fit gaussian to data
#     popt, pcov = curve_fit(gaussian_fit, plot_data[:,0], plot_data[:,1], p0=[1, wavelength, 1])

#     # Print fit parameters with uncertainties
#     # print('a = ' + str(popt[0]) + ' +/- ' + str(np.sqrt(pcov[0,0])))
#     print('λ = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1,1])))
#     print('σ = ' + str(popt[2]) + ' +/- ' + str(np.sqrt(pcov[2,2])))

#     # Plot gaussian fit
#     plt.plot(plot_data[:,0], gaussian_fit(plot_data[:,0], *popt), label='Gaussian fit')
#     plt.legend()

#     # Calculate r^2
#     residuals = plot_data[:,1] - gaussian_fit(plot_data[:,0], *popt)
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
#     r_squared = 1 - (ss_res / ss_tot)
#     print('r^2 = ' + str(r_squared))

#     # Show plot
#     plt.show()

# def double_plot_around_region(data, wavelength1, wavelength2):

#     # Plot data around a certain wavelength
#     plot_data = data[(data[:,0] > wavelength1 - 20) & (data[:,0] < wavelength2 + 20)]
#     plt.figure()
#     plt.plot(plot_data[:,0], plot_data[:,1])
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Intensity (arb. units)')
#     plt.title('Wavelength region around ' + str(wavelength1) + 'nm and ' + str(wavelength2) + 'nm')

#     # Fit double gaussian to data
#     popt, pcov = curve_fit(double_gaussian_fit, plot_data[:,0], plot_data[:,1], p0=[1, wavelength1, 1, 1, wavelength2, 1])

#     # Print fit parameters with uncertainties
#     print('λ1 = ' + str(popt[1]) + ' +/- ' + str(np.sqrt(pcov[1,1])))
#     print('σ1 = ' + str(popt[2]) + ' +/- ' + str(np.sqrt(pcov[2,2])))
#     print('λ2 = ' + str(popt[4]) + ' +/- ' + str(np.sqrt(pcov[4,4])))
#     print('σ2 = ' + str(popt[5]) + ' +/- ' + str(np.sqrt(pcov[5,5])))


#     # Plot gaussian fit
#     plt.plot(plot_data[:,0], double_gaussian_fit(plot_data[:,0], *popt), label='Gaussian fit')
#     plt.legend()

#     # Calculate r^2
#     residuals = plot_data[:,1] - double_gaussian_fit(plot_data[:,0], *popt)
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((plot_data[:,1]-np.mean(plot_data[:,1]))**2)
#     r_squared = 1 - (ss_res / ss_tot)
#     print('r^2 = ' + str(r_squared))

#     # Show plot
#     plt.show()


# double_plot_around_region(high_on, 651, 656)