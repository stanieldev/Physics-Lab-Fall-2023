import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Week 1 Constants
C = 0.822787  # Prelab
dC = 6.81e-4  # Prelab
R_0 = 4.76    # Ohms
dR_0 = 0.40   # Ohms
T_0 = 296.6   # K
dT_0 = 0.1    # K


# Plotting done in Week 1's Prelab
def week_1_prelab():

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

# Data compiling done in Week 1's Lab
def week_1_compile_data():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_1_raw.txt', skiprows=1)

    # Split the data into three arrays
    V = data[:,0]
    I = data[:,1]
    P = data[:,2]


    # Find the mean and standard error of voltage
    def voltage(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _V = V
        _dV = np.full(V.shape, 0.1)
        return _V, _dV
    V, dV = voltage(V)

    # Find the mean and standard error of current
    def current(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _I = I
        _dI = np.full(I.shape, 0.1)
        return _I, _dI
    I, dI = current(I)

    # Find the mean and standard error of power
    def power(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _P = P
        _dP = np.full(P.shape, 0.001)
        return _P, _dP
    P, dP = power(P)


    # Compile resistance data
    def resistance(V: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return V / I, (V / I) * np.sqrt((dV/V)**2 + (dI/I)**2)
    R, dR = resistance(V, I)

    # Compile temperature data
    def temperature(R: np.ndarray, dR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return T_0 * (R/R_0)**C, T_0 * (R/R_0)**C * np.sqrt((np.log(R/R_0) * dC) ** 2 + (C * (dR/R)) ** 2)
    T, dT = temperature(R, dR)

    # Compile log temperature data
    def log_temperature(R: np.ndarray, dR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.log(T_0 / R_0 ** C) + C * np.log(R), np.sqrt((np.log(R/R_0) * dC) ** 2 + (C * (dR/R)) ** 2)
    log_T, log_dT = log_temperature(R, dR)

    # Compile log power data
    def log_power(P: np.ndarray, dP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.log(P), dP / P
    log_P, log_dP = log_power(P, dP)


    # Write to file
    data = np.column_stack((log_T, log_dT, log_P, log_dP))
    np.savetxt('thermal_radiation/week_1_log_compiled.txt', data, header='log(T) dlog(T) log(P) dlog(P)')
    data = np.column_stack((T, dT, P, dP))
    np.savetxt('thermal_radiation/week_1_lin_compiled.txt', data, header='T dT P dP')

# Plot generation done in Week 1's Lab
def week_1_plot_log_power_vs_log_temperature():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_1_log_compiled.txt', skiprows=1)

    # Split the data into four arrays
    log_T = data[:,0]
    log_dT = data[:,1]
    log_P = data[:,2]
    log_dP = data[:,3]

    # Calculate a linear regression
    def linear_regression(x, a, b): return a * x + b
    params, cov = curve_fit(linear_regression, log_T, log_P)
    s, b = params
    ds, db = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

    # s ~ 4, b = -ln(k)
    # print s and b
    print(f"s +/- ds = {s:.6f} +/- {ds:.6f}")
    print(f"b +/- db = {b:.6f} +/- {db:.6f}")

    # Calculate r^2
    r_squared = 1 - (np.sum((log_P - linear_regression(log_T, s, b))**2) / np.sum((log_P - np.mean(log_P))**2))
    print(f"r^2 = {r_squared*100:.4f}%")

    # Plot the data
    plt.title('Thermal Radiation')
    plt.xlabel('ln(T) [ln K]')
    plt.ylabel('ln(P) [ln mV]')
    plt.errorbar(log_T, log_P, xerr=log_dT, yerr=log_dP, fmt='o')
    plt.plot(log_T, linear_regression(log_T, s, b))
    plt.show()

def week_1_plot_power_vs_temperature():
    
        # Load the data
        data = np.loadtxt('thermal_radiation/week_1_lin_compiled.txt', skiprows=1)
    
        # Split the data into four arrays
        T = data[:,0]
        dT = data[:,1]
        P = data[:,2]
        dP = data[:,3]
    
        # Plot the data
        plt.errorbar(T, P, xerr=dT, yerr=dP, fmt='o')
        plt.xlabel('T [K]')
        plt.ylabel('P [mV]')
        plt.title('Thermal Radiation')
        plt.show()

def week_1_plot_radiated_power_vs_electrical_power():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_1_raw.txt', skiprows=1)

    # Split the data into three arrays
    electrical_power = data[:,0] * data[:,1]
    radiated_power = data[:,2]

    # Calculate a linear regression
    def linear_regression(x, a): return a * x
    params, cov = curve_fit(linear_regression, electrical_power, radiated_power)
    s = params[0]
    ds = np.sqrt(cov[0,0])
    print(f"s +/- ds = {s:.6f} +/- {ds:.6f}")

    # Calculate r^2
    r_squared = 1 - (np.sum((radiated_power - linear_regression(electrical_power, s))**2) / np.sum((radiated_power - np.mean(radiated_power))**2))
    print(f"r^2 = {r_squared*100:.4f}%")

    # Plot the data
    plt.plot(electrical_power, radiated_power, 'o')
    plt.plot(electrical_power, linear_regression(electrical_power, s))
    plt.xlabel('Electrical Power [W]')
    plt.ylabel('Radiated Power [mV]')
    plt.title('Thermal Radiation')
    plt.show()



# Week 2 Constants
C = 0.822787  # Prelab
dC = 6.81e-4  # Prelab



# Data compiling done in Week 2's Lab
def compile_first_graph_data():
    
    # Load the data
    data = np.loadtxt('thermal_radiation/week_2_raw.txt', skiprows=1, encoding='utf-8-sig')

    # Split the data into 4 arrays
    λ = data[:,0]
    V, dV = data[:,1], np.full(data[:,1].shape, 0.1)
    I, dI = data[:,2], np.full(data[:,2].shape, 0.1)
    P, dP = data[:,3], np.full(data[:,3].shape, 0.001)

    # Calculate resistance
    def resistance(V: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return V / I, (V / I) * np.sqrt((dV/V)**2 + (dI/I)**2)
    R, dR = resistance(V, I)

    # Calculate temperature
    def temperature(R: np.ndarray, dR: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return T_0 * (R/R_0)**C, T_0 * (R/R_0)**C * np.sqrt((np.log(R/R_0) * dC) ** 2 + (C * (dR/R)) ** 2)
    T, dT = temperature(R, dR)

    # Calculate inverse temperature
    def inverse_temperature(T: np.ndarray, dT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 1/T, dT / T**2
    inv_T, inv_dT = inverse_temperature(T, dT)
    
    # Calculate log power
    def log_power(P: np.ndarray, dP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.log(P), dP / P
    log_P, log_dP = log_power(P, dP)

    # Save data to file for later use
    data = np.column_stack((λ, inv_T, inv_dT, log_P, log_dP))
    np.savetxt('thermal_radiation/week_2_compiled.txt', data, header='λ(nm) 1/T d(1/T) log(P) dlog(P)', encoding='utf-8-sig')

def compile_second_graph_data():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_2_compiled.txt', skiprows=1, encoding='utf-8-sig')

    # Split the data into 5 arrays
    λ = data[:,0]
    inv_T, inv_dT = data[:,1], data[:,2]
    log_P, inv_dP = data[:,3], data[:,4]

    # Calculate the ODR fit
    def calculate_slope(_λ: int):
        def linear_regression(x, a, b): return a * x + b
        params, cov = curve_fit(linear_regression, inv_T[λ == _λ], log_P[λ == _λ])
        return params[0], np.sqrt(cov[0,0])

    # Find each slope
    s600nm, ds600nm = calculate_slope(600)
    s800nm, ds800nm = calculate_slope(800)
    s1200nm, ds1200nm = calculate_slope(1200)
    s1600nm, ds1600nm = calculate_slope(1600)
    s1800nm, ds1800nm = calculate_slope(1800)
    s2400nm, ds2400nm = calculate_slope(2400)

    # Output to file as s, ds pairs
    inv_l = [1/600, 1/800, 1/1200, 1/1600, 1/1800, 1/2400]
    s = [s600nm, s800nm, s1200nm, s1600nm, s1800nm, s2400nm]
    ds = [ds600nm, ds800nm, ds1200nm, ds1600nm, ds1800nm, ds2400nm]
    data = np.column_stack((inv_l, s, ds))
    np.savetxt('thermal_radiation/week_2_slopes.txt', data, header='Inverse Wavelength[nm^-1] Slope dSlope', encoding='utf-8-sig')




# Plotting done in Week 2's Lab
def plot_first_graph():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_2_compiled.txt', skiprows=1, encoding='utf-8-sig')

    # Split the data into 5 arrays
    λ = data[:,0]
    inv_T, inv_dT = data[:,1], data[:,2]
    log_P, inv_dP = data[:,3], data[:,4]

    # Create plot
    plt.xlabel('1/T [1/K]')
    plt.ylabel('ln(P) [ln mV]')
    plt.title('log(Intensity) vs 1/Temperature')
    
    # Plot the data
    plt.errorbar(inv_T[λ == 600], log_P[λ == 600], yerr=inv_dP[λ == 600], fmt='o', color='red')
    plt.errorbar(inv_T[λ == 800], log_P[λ == 800], yerr=inv_dP[λ == 800], fmt='o', color='green')
    plt.errorbar(inv_T[λ == 1200], log_P[λ == 1200], yerr=inv_dP[λ == 1200], fmt='o', color='blue')
    plt.errorbar(inv_T[λ == 1600], log_P[λ == 1600], yerr=inv_dP[λ == 1600], fmt='o', color='yellow')
    plt.errorbar(inv_T[λ == 1800], log_P[λ == 1800], yerr=inv_dP[λ == 1800], fmt='o', color='magenta')
    plt.errorbar(inv_T[λ == 2400], log_P[λ == 2400], yerr=inv_dP[λ == 2400], fmt='o', color='cyan')

    # Plot linear regressions
    def plot_linear_regression(_λ: int, color: str):
        def linear_regression(x, a, b): return a * x + b
        params, cov = curve_fit(linear_regression, inv_T[λ == _λ], log_P[λ == _λ])
        plt.plot(inv_T[λ == _λ], linear_regression(inv_T[λ == _λ], params[0], params[1]), label=f"λ = {_λ} nm", color=color)
    plot_linear_regression(600, color='red')
    plot_linear_regression(800, color='green')
    plot_linear_regression(1200, color='blue')
    plot_linear_regression(1600, color='yellow')
    plot_linear_regression(1800, color='magenta')
    plot_linear_regression(2400, color='cyan')

    # Show plot
    plt.legend()
    plt.show()

def plot_second_graph():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_2_slopes.txt', skiprows=1, encoding='utf-8-sig')

    # Split the data into three arrays
    inv_l, s, ds = data[:,0], data[:,1], data[:,2]

    # Create the plot
    plt.xlabel('1/λ [1/nm]')
    plt.ylabel('Slope [1/K]')
    plt.title('Thermal Radiation')

    # Plot the data
    plt.errorbar(inv_l, s, xerr=0, yerr=ds, fmt='o')

    # Calculate a linear regression
    def linear_regression(x, a): return a * x
    params, cov = curve_fit(linear_regression, inv_l, s)

    # Print the slope with uncertainty
    print(f"hc/k +/- d(hc/k) = {-params[0]:.3f}nm K +/- {np.sqrt(cov[0,0]):.3f}nm K")
    print("                 = 14389602.471nm K")

    # Plot the linear regression
    plt.plot(inv_l, linear_regression(inv_l, params[0]))

    # Show plot
    plt.show()


plot_first_graph()
plot_second_graph()