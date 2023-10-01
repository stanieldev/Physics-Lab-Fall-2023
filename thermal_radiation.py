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


# Constants
C = 0.822787  # Prelab
dC = 6.81e-4  # Prelab
T_0 = 300
R_0 = 5
A_s = 1
Ω = 1
# If A_s or Ω has an error, it can be quickly added to the code below
# Consider switching out linear regressions with orthogonal distance regressions


# Data compiling done in Week 1's Lab
def compile_data():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_1_raw.txt', skiprows=1)

    # Split the data into three arrays
    V = data[:,0]
    I = data[:,1]
    P = data[:,2]


    # Find the mean and standard error of voltage
    def voltage(V: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        V_bar = []   # mean
        dV_bar = []  # standard deviation / sqrt(n)
        return V_bar, dV_bar
    V, dV = voltage(V)

    # Find the mean and standard error of current
    def current(I: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        I_bar = []   # mean
        dI_bar = []  # standard deviation / sqrt(n)
        return I_bar, dI_bar
    I, dI = current(I)

    # Find the mean and standard error of power
    def power(P: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        P_bar = []   # mean
        dP_bar = []  # standard deviation / sqrt(n)
        return P_bar, dP_bar
    P, dP = power(P)


    # Compile resistance data
    def resistance(V: np.ndarray, I: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        return V / I, (V / I) * np.sqrt((dV/V)**2 + (dI/I)**2)
    R, dR = resistance(V, I)

    # Compile temperature data
    def temperature(R: np.ndarray, dR: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        return T_0 * (R/R_0)**C, T_0 * (R/R_0)**C * np.sqrt((np.log(R/R_0) * dC) ** 2 + (C * (dR/R)) ** 2)
    T, dT = temperature(R, dR)

    # Compile log temperature data
    def log_temperature(R: np.ndarray, dR: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        return np.log(T_0 / R_0 ** C) + C * np.log(R), np.sqrt((np.log(R/R_0) * dC) ** 2 + (C * (dR/R)) ** 2)
    log_T, log_dT = log_temperature(R, dR)

    # Compile log power data
    def log_power(P: np.ndarray, dP: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        return np.log(P), dP / P
    log_P, log_dP = log_power(P, dP)


    # Write to file
    data = np.column_stack((log_T, log_dT, log_P, log_dP))
    np.savetxt('thermal_radiation/week_1_log_compiled.txt', data, header='log(T) dlog(T) log(P) dlog(P)')
    data = np.column_stack((T, dT, P, dP))
    np.savetxt('thermal_radiation/week_1_lin_compiled.txt', data, header='T dT P dP')


# Plot generation done in Week 1's Lab
def plot_log_power_vs_log_temperature():

    # Load the data
    data = np.loadtxt('thermal_radiation/week_1_log_compiled.txt', skiprows=1)

    # Split the data into four arrays
    log_T = data[:,0]
    log_dT = data[:,1]
    log_P = data[:,2]
    log_dP = data[:,3]

    # Calculate a linear regression
    def linear_regression(x, b): return 4 * x + b
    b, cov = curve_fit(linear_regression, log_T, log_P)
    db = np.sqrt(cov[0,0])
    σ = (4 * np.pi) / (A_s * Ω) * (10 ** b)
    dσ = (4 * np.pi) / (A_s * Ω) * (10 ** b) * np.log(10) * db
    print(f"σ +/- δσ = {σ:.3e} +/- {dσ:.3e} W/K^4")

    # Plot the data
    plt.title('Thermal Radiation')
    plt.xlabel('ln(T)')
    plt.ylabel('ln(P)')
    plt.errorbar(log_T, log_P, xerr=log_dT, yerr=log_dP, fmt='o')
    plt.plot(log_T, linear_regression(log_T, b), label=f"b = {b[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}")
    plt.legend()
    plt.show()

def plot_power_vs_temperature():
    
        # Load the data
        data = np.loadtxt('thermal_radiation/week_1_lin_compiled.txt', skiprows=1)
    
        # Split the data into four arrays
        T = data[:,0]
        dT = data[:,1]
        P = data[:,2]
        dP = data[:,3]
    
        # Plot the data
        plt.errorbar(T, P, xerr=dT, yerr=dP, fmt='o')
        plt.xlabel('T (K)')
        plt.ylabel('P (W)')
        plt.title('Thermal Radiation')
        plt.show()

def plot_power_vs_temperature_4th():
    
        # Load the data
        data = np.loadtxt('thermal_radiation/week_1_lin_compiled.txt', skiprows=1)
    
        # Split the data into four arrays
        T = data[:,0]
        dT = data[:,1]
        P = data[:,2]
        dP = data[:,3]

        # Rause the temperature to the 4th power
        dT = 4 * T**3 * dT
        T = T**4

        # Subtract T_0^4 from all temperatures
        T = T - T_0**4

        # Calculate a linear regression
        def linear_regression(x, a): return a * x
        s, cov = curve_fit(linear_regression, T, P)
        ds = np.sqrt(cov[0,0])
        σ = (4 * np.pi) / (A_s * Ω) * s
        dσ = (4 * np.pi) / (A_s * Ω) * ds
        print(f"σ +/- δσ = {σ:.3e} +/- {dσ:.3e} W/K^4")
    
        # Plot the data
        plt.title('Thermal Radiation')
        plt.xlabel('T^4 - T_0^4 (K)')
        plt.ylabel('P (W)')
        plt.errorbar(T, P, xerr=dT, yerr=dP, fmt='o')
        plt.plot(T, linear_regression(T, s), label=f"s = {s[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f}")
        plt.legend()
        plt.show()

