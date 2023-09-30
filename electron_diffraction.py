import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 66   # mm
T = 1.5  # mm
L = 130  # mm
dL = 2   # mm
dV = 100 # V
d11 = 0.123  # nm
d10 = 0.213  # nm
K = 1.227  # V^1/2 nm


# raw -> compiled -> corrected -> diffraction wavelength 


# Raw to compiled data
def raw_data_to_compiled_data() -> None:
    # Import data
    path = "./electron_diffraction/raw_data.txt"
    data = np.loadtxt(path, unpack=True)

    # Format new data
    kV = data[0]
    SRI = data[1]
    SRM = data[2]
    LRI = data[3]
    LRM = data[4]

    # Calculate the mean and standard deviation
    SRI_mean = []
    SRM_mean = []
    LRI_mean = []
    LRM_mean = []
    SRI_err = []
    SRM_err = []
    LRI_err = []
    LRM_err = []
    for i in range(0, len(kV)//3):
        SRI_subset = [SRI[3*i], SRI[3*i+1], SRI[3*i+2]]
        SRM_subset = [SRM[3*i], SRM[3*i+1], SRM[3*i+2]]
        LRI_subset = [LRI[3*i], LRI[3*i+1], LRI[3*i+2]]
        LRM_subset = [LRM[3*i], LRM[3*i+1], LRM[3*i+2]]
        SRI_mean.append(np.mean(SRI_subset))
        SRM_mean.append(np.mean(SRM_subset))
        LRI_mean.append(np.mean(LRI_subset))
        LRM_mean.append(np.mean(LRM_subset))
        SRI_err.append(np.std(SRI_subset)/np.sqrt(2))  # Accounting for population vs. sample standard deviation
        SRM_err.append(np.std(SRM_subset)/np.sqrt(2))
        LRI_err.append(np.std(LRI_subset)/np.sqrt(2))
        LRM_err.append(np.std(LRM_subset)/np.sqrt(2))

    # Write to file
    kV = kV[0::3]
    SRI_mean = np.array(SRI_mean)
    SRM_mean = np.array(SRM_mean)
    LRI_mean = np.array(LRI_mean)
    LRM_mean = np.array(LRM_mean)
    SRI_err = np.array(SRI_err)
    SRM_err = np.array(SRM_err)
    LRI_err = np.array(LRI_err)
    LRM_err = np.array(LRM_err)
    data = np.array([kV, SRI_mean, SRI_err, SRM_mean, SRM_err, LRI_mean, LRI_err, LRM_mean, LRM_err])
    data = np.transpose(data)
    np.savetxt("./electron_diffraction/compiled_data.txt", data, delimiter=" ", header="kV SRI(mm) SRI_err(mm) SRM(mm) SRM_err(mm) LRI(mm) LRI_err(mm) LRM(mm) LRM_err(mm)")

# Compiled to D' corrected data
def compiled_data_to_corrected_data() -> None:
    # Import data
    path = "./electron_diffraction/compiled_data.txt"
    data = np.loadtxt(path, unpack=True)

    # Unpacking data
    V = data[0]*1000  # kV -> V
    SRI, SRI_err = data[1], data[2]
    SRM, SRM_err = data[3], data[4]
    LRI, LRI_err = data[5], data[6]
    LRM, LRM_err = data[7], data[8]

    # Modify data to account for D'
    def d_prime_correction(d: np.ndarray, d_err: np.ndarray) -> np.ndarray:
        
        # Calculate D'
        D_PRIME = (d * L) / (L - T - R + np.sqrt(R**2 - (d/2)**2))

        # Calculate error in D'
        D_PRIME_BY_D_D = (D_PRIME / d) + (D_PRIME)**2 / (4 * L * np.sqrt(R**2 - (d/2)**2))
        D_PRIME_BY_D_L = (D_PRIME / L) + (D_PRIME)**2 / (d * L)

        # Calculate error in D'
        D_PRIME_ERR = np.sqrt((D_PRIME_BY_D_D * d_err)**2 + (D_PRIME_BY_D_L * dL)**2)

        # Return D', D'_err
        return D_PRIME, D_PRIME_ERR

    # Calculate D' for each data set
    SRI, SRI_err = d_prime_correction(SRI, SRI_err)
    SRM, SRM_err = d_prime_correction(SRM, SRM_err)
    LRI, LRI_err = d_prime_correction(LRI, LRI_err)
    LRM, LRM_err = d_prime_correction(LRM, LRM_err)

    # Write to file
    data = np.array([V/1000, SRI, SRI_err, SRM, SRM_err, LRI, LRI_err, LRM, LRM_err])
    data = np.transpose(data)
    np.savetxt("./electron_diffraction/corrected_data.txt", data, delimiter=" ", header="kV SRI(mm) SRI_err(mm) SRM(mm) SRM_err(mm) LRI(mm) LRI_err(mm) LRM(mm) LRM_err(mm)")

# Corrected to wavelengths data
def corrected_data_to_wavelengths_data() -> None:
    # Import data
    path = "./electron_diffraction/corrected_data.txt"
    data = np.loadtxt(path, unpack=True)

    # Unpacking data
    V = data[0]*1000  # kV -> V
    SRI, SRI_err = data[1], data[2]
    SRM, SRM_err = data[3], data[4]
    LRI, LRI_err = data[5], data[6]
    LRM, LRM_err = data[7], data[8]

    # Calculate debroglie wavelengths and error
    λDEB = K / V**0.5
    λDEB_err = λDEB/(2*V) * dV

    # Calculate diffraction wavelengths and error (D' corrected)
    def diffraction_wavelength(D_IJ: float, d: np.ndarray, d_err: np.ndarray) -> np.ndarray:
        λ = D_IJ * d / (2 * L)
        λ_err = λ * np.sqrt((d_err/d)**2 + (dL/L)**2)
        return λ, λ_err
    
    # Calculate diffraction wavelengths and error
    λSRI, λSRI_err = diffraction_wavelength(d10, SRI, SRI_err)
    λSRM, λSRM_err = diffraction_wavelength(d10, SRM, SRM_err)
    λLRI, λLRI_err = diffraction_wavelength(d11, LRI, LRI_err)
    λLRM, λLRM_err = diffraction_wavelength(d11, LRM, LRM_err)

    # Write to file
    data = np.array([V/1000, λDEB, λDEB_err, λSRI, λSRI_err, λSRM, λSRM_err, λLRI, λLRI_err, λLRM, λLRM_err])
    data = np.transpose(data)
    np.savetxt("./electron_diffraction/wavelengths_data.txt", data, delimiter=" ", header="kV DEB(nm) DEB_err(nm) SRI(nm) SRI_err(nm) SRM(nm) SRM_err(nm) LRI(nm) LRI_err(nm) LRM(nm) LRM_err(nm)")



# Load data
path = "./electron_diffraction/wavelengths_data.txt"
data = np.loadtxt(path, unpack=True)

# Unpacking data
kV = data[0]
λDEB, λDEB_err = data[1], data[2]
λSRI, λSRI_err = data[3], data[4]
λSRM, λSRM_err = data[5], data[6]
λLRI, λLRI_err = data[7], data[8]
λLRM, λLRM_err = data[9], data[10]

# Plot Wavelength vs Voltage
plt.title("Electron Diffraction")
plt.xlabel("Voltage (kV)")
plt.ylabel("Wavelength (nm)")
plt.errorbar(kV, λSRI, yerr=λSRI_err, fmt=".", label="SRI")
plt.errorbar(kV, λSRM, yerr=λSRM_err, fmt=".", label="SRM")
plt.errorbar(kV, λLRI, yerr=λLRI_err, fmt=".", label="LRI")
plt.errorbar(kV, λLRM, yerr=λLRM_err, fmt=".", label="LRM")
x = np.linspace(2, 5, 300)
plt.plot(x, K / (1000*x)**0.5, label="DeBroglie")
plt.legend()
plt.show()

# Plot 1/λ² vs Voltage
plt.title("Electron Diffraction")
plt.xlabel("Voltage (kV)")
plt.ylabel("1/λ² (nm⁻²)")
plt.errorbar(kV, 1/λSRI**2, yerr=(2*λSRI_err/λSRI**3), fmt=".", label="SRI")
plt.errorbar(kV, 1/λSRM**2, yerr=(2*λSRM_err/λSRM**3), fmt=".", label="SRM")
plt.errorbar(kV, 1/λLRI**2, yerr=(2*λLRI_err/λLRI**3), fmt=".", label="LRI")
plt.errorbar(kV, 1/λLRM**2, yerr=(2*λLRM_err/λLRM**3), fmt=".", label="LRM")
x = np.linspace(2, 5, 300)
plt.plot(x, (K / (1000*x)**0.5)**(-2), label="DeBroglie")
plt.legend()
plt.show()

# Plot the bragg wavelengths vs the DeBroglie wavelength
plt.title("Electron Diffraction")
plt.xlabel("DeBroglie Wavelength (nm)")
plt.ylabel("Bragg Wavelength (nm)")
plt.errorbar(λDEB, λSRI, xerr=λDEB_err, yerr=λSRI_err, fmt=".", label="SRI")
plt.errorbar(λDEB, λSRM, xerr=λDEB_err, yerr=λSRM_err, fmt=".", label="SRM")
plt.errorbar(λDEB, λLRI, xerr=λDEB_err, yerr=λLRI_err, fmt=".", label="LRI")
plt.errorbar(λDEB, λLRM, xerr=λDEB_err, yerr=λLRM_err, fmt=".", label="LRM")
plt.plot(λDEB, λDEB, label="DeBroglie")

# Keep the plot square
plt.gca().set_aspect('equal', adjustable='box')

plt.legend()
plt.show()
