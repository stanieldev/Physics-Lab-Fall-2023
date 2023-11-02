# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit



# Constants
η = 1.85e-5     # Pa s
ρ_air = 1.2     # kg/m^3
ρ_oil = 872     # kg/m^3
A = 0.07776e-6  # m
g = 9.81        # m/s^2
e = 1.602176634e-19 # C

# Experimental parameters
d, δd = 5.91e-3, 0.05e-3  # m
M, δM = 2.04, 0.05        # Unitless

# Measurement uncertainties
δy = 0.1   # mm
δt = 0.5   # s
δV = 1   # V



# Helper functions
def linear_cutoff(x: float, a, b, s):
    if x <= a - 1/s: return 0
    if a - 1/s < x and x < a: return s*(x-a) + 1
    if a <= x and x <= b: return 1
    if b < x and x < b + 1/s: return -s*(x-b) + 1
    if x >= b + 1/s: return 0
    raise Exception("x is out of bounds")

def linear_cutoff_function(x, a, b, s):
    return [linear_cutoff(i, a, b, s) for i in x]



# Statistics functions
def gaussian(x, A, μ, σ):
    return A*np.exp(-(x - μ)**2/(2*σ**2))

def find_statistics(data, weight_function=lambda x: 1):

    # Create a weighted list
    weights = weight_function(data)

    # Calculate a weighted average
    μ = np.sum(data * weights) / np.sum(weights)

    # Calculate a weighted standard deviation
    σ = np.sqrt(np.sum(weights * (data - μ)**2) / np.sum(weights))

    # Calculate weighted sum of squared errors
    error = np.sum(weights * (data - gaussian(data, 1, μ, σ))**2)

    # Return statistics
    return μ, σ, error

def find_gaussian(data, min_x, max_x, divisions=100, weight_dropoff=0.2):

    # Create a np array of x values using min_x, max_x, and divisions
    midpoints = np.linspace(min_x, max_x, divisions)

    # Create a few empty lists to store the best gaussian
    best_params = None
    best_error = float('inf')

    # Loop through all initial x values to find the best gaussian
    for midpoint in midpoints:

        # Use find_statistics to find μ, σ
        μ, σ, error = find_statistics(data, weight_function=lambda x: gaussian(x, 1, midpoint, weight_dropoff))

        # If the error is better than the best error, save the parameters
        if error < best_error:
            best_params = (μ, σ)
            best_error = error
    
    # Return the best parameters
    return best_params, best_error



# Measurement functions
def v_δv(Δy: float, Δt: float):
    v = (Δy/Δt)/M  # M is magnifying power
    δv = v * np.sqrt( (δy/Δy)**2 + (δt/Δt)**2 + (δM/M)**2 )
    return v, δv

def r_δr(v_G: float, δv_G: float):
    K = 9*η/(2*(ρ_oil - ρ_air)*g)
    r = np.sqrt(K * v_G)
    δr = (K/(2*r)) * δv_G
    return r, δr

def q_δq(r, δr, v_G, δv_G, v_V, δv_V, V):
    q = (6 * np.pi * η * d * (v_G + v_V) * r) / V
    δq = q * np.sqrt( (δd/d)**2 + (δV/V)**2 + ((δv_G+δv_V)/(v_G+v_V))**2 + (δr/r)**2 )
    return q, δq

def qc_δqc(Q, δQ, r, δr):
    qc = Q / (1 + A/r)**1.5
    δqc = qc * np.sqrt( (δQ/Q)**2 + (1.5*A*δr/(r*(r-A)))**2 )
    return qc, δqc

def n_δn(q, δq):
    n = q / 1.602e-19
    δn = n * (δq/q)
    return n, δn



# Data compile functions [Run once]
def compile_v():

    # Load data
    data = np.loadtxt("oil_drop/data_raw.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    V = data[:, 0]  # V
    Δy_V, Δt_V = data[:, 1], data[:, 2]  # mm, s
    Δy_G, Δt_G = data[:, 3], data[:, 4]  # mm, s

    # Calculate v,δv (mm/s)
    v_V, δv_V = v_δv(Δy_V, Δt_V)
    v_G, δv_G = v_δv(Δy_G, Δt_G)

    # Save data to file
    data = np.transpose([V, v_V, δv_V, v_G, δv_G])
    head = "V(V) | v_V(mm/s) | δv_V(mm/s) | v_G(mm/s) | δv_G(mm/s)"
    np.savetxt("oil_drop/data_velocities.txt", data, header=head, encoding='utf-8-sig')

def compile_r():

    # Load data
    data = np.loadtxt("oil_drop/data_velocities.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    v_G = data[:, 3] * 1e-3   # Convert velocity mm/s -> m/s
    δv_G = data[:, 4] * 1e-3  # Convert velocity mm/s -> m/s

    # Calculate r,δr (m)
    r, δr = r_δr(v_G, δv_G)

    # Save data to file
    data = np.transpose([r, δr])
    head = "r(m) | δr(m)"
    np.savetxt("oil_drop/data_radii.txt", data, header=head, encoding='utf-8-sig')

def compile_q():

    # Load data
    vel_data = np.loadtxt("oil_drop/data_velocities.txt", skiprows=1, encoding='utf-8-sig')
    rad_data = np.loadtxt("oil_drop/data_radii.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    r = rad_data[:, 0]
    δr = rad_data[:, 1]
    V = vel_data[:, 0]
    v_V = vel_data[:, 1] * 1e-3   # Convert velocity mm/s -> m/s
    δv_V = vel_data[:, 2] * 1e-3  # Convert velocity mm/s -> m/s
    v_G = vel_data[:, 3] * 1e-3   # Convert velocity mm/s -> m/s
    δv_G = vel_data[:, 4] * 1e-3  # Convert velocity mm/s -> m/s

    # Calculate q,δq (C)
    q, δq = q_δq(r, δr, v_G, δv_G, v_V, δv_V, V)

    # Save data to file
    data = np.transpose([q, δq])
    head = "q(C) | δq(C)"
    np.savetxt("oil_drop/data_charges.txt", data, header=head, encoding='utf-8-sig')

def compile_qc():

    # Load data
    rad_data = np.loadtxt("oil_drop/data_radii.txt", skiprows=1, encoding='utf-8-sig')
    chr_data = np.loadtxt("oil_drop/data_charges.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    q = chr_data[:, 0]
    δq = chr_data[:, 1]
    r = rad_data[:, 0]
    δr = rad_data[:, 1]
    
    # Calculate qc,δqc (C)
    qc, δqc = qc_δqc(q, δq, r, δr)

    # Save data to file
    data = np.transpose([qc, δqc])
    head = "qc(C) | δqc(C)"
    np.savetxt("oil_drop/data_charges_corrected.txt", data, header=head, encoding='utf-8-sig')

def compile_elementary():

    # Load data
    data = np.loadtxt("oil_drop/data_charges_corrected.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    q = data[:, 0]
    δq = data[:, 1]

    # Calculate n,δn (unitless)
    n, δn = n_δn(q, δq)

    # Save data to file
    data = np.transpose([n, δn])
    head = "n(unitless) | δn(unitless)"
    np.savetxt("oil_drop/data_elementary.txt", data, header=head, encoding='utf-8-sig')

def compile_all():
    compile_v()
    compile_r()
    compile_q()
    compile_qc()
    compile_elementary()



# Graphing functions
def plot_histogram(plt: plt, data, 
                   intermediate_tick_count=1, 
                   force_max_x=None,
                   force_min_x=None,
                   show_minor_ticks=False,
                   dy=0.0):

    # Determine min-max q
    min_q = force_min_x if isinstance(force_min_x, (int, float)) else 0
    max_q = force_max_x if isinstance(force_max_x, (int, float)) else (max(data) if len(data.shape) == 1 else max(data[:, 0]))

    # Better plot formatting
    Δq = 1/intermediate_tick_count
    Δx = -(1/intermediate_tick_count)/2

    # Creating useful lists
    upper = np.floor((max_q - Δx)//Δq) + 2
    histogram_list = [Δx + i*Δq for i in range(int(upper))]
    x_minor_ticks = [Δx + i*Δq for i in range(int(upper))][1:]
    x_minor_labels = [f"+{np.mod(2*i-1, 2*intermediate_tick_count)}/{2*intermediate_tick_count}" for i in range(int(upper))][1:]
    x_major_ticks = [i for i in range(int(max_q)+1)]
    x_major_labels = [f"{i}e-19" for i in range(int(max_q)+1)]
    quantity_labels = ((histogram_list[i] + Δq/2, int(plt.gca().patches[i].get_height())) for i in range(1, len(histogram_list)-1))

    # Set up plot
    plt.title(f"Charge Distribution")
    plt.xlabel(f"Q [Δq=1/{intermediate_tick_count}]")
    plt.ylabel('Quantity of Drops')

    # Plot histogram with thin black borders on each bar
    plt.hist(data, bins=histogram_list, edgecolor='black', linewidth=0.5)

    # Plot formatted ticks and grid
    plt.xticks(x_minor_ticks, x_minor_labels, minor=True, fontsize=8) if show_minor_ticks else None
    plt.xticks(x_major_ticks, x_major_labels)
    plt.gca().grid(which='minor', alpha=0.0)
    plt.gca().grid(which='major', alpha=1.0)
    [plt.gca().annotate(f"{int(y)}", xy=(x, y+dy), xytext=(x, y+1+dy), ha='center', va='center') for x, y in quantity_labels]
    plt.gca().axes.get_yaxis().set_ticks([])

    # Show plot
    plt.xlim(min_q + Δx, max_q - Δx)





# Experimental charge data
# data = np.loadtxt("oil_drop/data_charges_corrected.txt", skiprows=1, encoding='utf-8-sig')[:,0]
# data = data * 1e19  # Scale q to e-19 C
data = np.loadtxt("oil_drop/compiled_data.txt", skiprows=1, encoding='utf-8-sig')

# Plot histogram
plot_histogram(plt, data, 
               intermediate_tick_count=8,
               force_min_x=0,
               force_max_x=9,
               show_minor_ticks=False, 
               dy=-0.8)

# Find gaussian fits
# μ, σ, error = find_statistics(data, weight_function=lambda x: linear_cutoff_function(x, 0.5, 1.5, 2.0))
(μ1, σ1), error1 = find_gaussian(data, min_x=1.4, max_x=1.8, divisions=100, weight_dropoff=0.50/1)
(μ2, σ2), error2 = find_gaussian(data, min_x=3.2, max_x=3.6, divisions=100, weight_dropoff=0.50/2)
(μ3, σ3), error3 = find_gaussian(data, min_x=4.6, max_x=5.0, divisions=100, weight_dropoff=0.50/3)

# Plot gaussian fit
SPACE = np.linspace(0, 20, 1000)
plt.plot(SPACE, gaussian(SPACE, 213, μ1, σ1))
plt.plot(SPACE, gaussian(SPACE, 78, μ2, σ2))
plt.plot(SPACE, gaussian(SPACE, 34, μ3, σ3))

# Print statistics
print(f"μ1, σ1 = {μ1},{σ1}")
print(f"μ2, σ2 = {μ2},{σ2}")
print(f"μ3, σ3 = {μ3},{σ3}")

# Show plot
plt.show()