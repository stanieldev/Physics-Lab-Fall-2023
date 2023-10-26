import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Constants
η = 1.85e-5    # Pa s
ρ_air = 1.2    # kg/m^3
ρ_oil = 872    # kg/m^3
d = 5.91e-3    # m
δd = 0.05e-3   # m
A = 0.07776e-6 # m
g = 9.81        # m/s^2
e = 1.602176634e-19 # C


# Magnifying power
M = 2.04       # Unitless
δM = 0.05      # Unitless

# Measurement uncertainties
δy = 0.1   # mm
δt = 0.5   # s
δV = 1   # V



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



# Data compile functions
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








# Graphing functions
def plot_histogram(data, 
                   intermediate_tick_count=1, 
                   force_max_x=None,
                   force_min_x=None,
                   show_minor_ticks=False,
                   dy=0.0):

    # Determine max q
    max_q = max(data) if len(data.shape) == 1 else max(data[:, 0])
    if isinstance(force_max_x, (int, float)):
        max_q = force_max_x

    # Determine min q
    min_q = force_min_x if isinstance(force_min_x, (int, float)) else 0

    # Better plot formatting
    Δq = 1/intermediate_tick_count
    Δx = -(1/intermediate_tick_count)/2

    # Creating useful lists
    upper = np.floor((max_q - Δx)//Δq) + 2
    histogram_list = [Δx + i*Δq for i in range(int(upper))]
    x_minor_ticks = [Δx + i*Δq for i in range(int(upper))][1:]
    x_minor_labels = [f"+{np.mod(2*i-1, 2*intermediate_tick_count)}/{2*intermediate_tick_count}" for i in range(int(upper))][1:]
    x_major_ticks = [i for i in range(int(max_q)+1)]
    x_major_labels = [f"{i}" for i in range(int(max_q)+1)]
    quantity_labels = ((histogram_list[i] + Δq/2, int(plt.gca().patches[i].get_height())) for i in range(1, len(histogram_list)-1))

    # Set up plot
    plt.title(f"Charge Distribution")
    plt.xlabel(f"Q (e) [Δq=1/{intermediate_tick_count}]")
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
    plt.show()





# # Elementary charge data
# data = np.loadtxt("oil_drop/data_elementary.txt", skiprows=1, encoding='utf-8-sig')[:,0]
# plot_histogram(data, 
#                intermediate_tick_count=3,
#                force_min_x=0,
#                force_max_x=6,
#                show_minor_ticks=True, 
#                dy=-0.8)




# # Compiled data
# data = np.loadtxt("oil_drop/compiled_data.txt", skiprows=1, encoding='utf-8-sig')
# data /= 1.602176634  # Convert Q -> N*e
# plot_histogram(data, 
#                intermediate_tick_count=5,
#                force_min_x=0,
#                force_max_x=5,
#                show_minor_ticks=True,
#                dy=8)














# Load and modify data


# Define weighted mean and standard deviation function
def calculate_weighted_params(x_data, mu, s):

    # Calculate a weight proportional to the CDF of the z-score about mu
    p = 1 - erf((abs(x_data - mu)/s)/np.sqrt(2))
    p /= sum(p)  # Normalize p
    N = len(x_data)
    
    # Calculate weighted mean and standard deviation
    weighted_mean = sum(x_data * p)
    weighted_std = np.sqrt(N/(N-1)) * np.sqrt(sum(p * (x_data - weighted_mean)**2))

    # Return weighted mean and standard deviation
    return weighted_mean, weighted_std

def find_equilibrium_params(x_data, MU, plot=False):

    # Find equilibrium params
    SCOPE_SPACE = np.linspace(0.05, 1, 1000)[1:]
    mu_list = []
    cs_list = []
    for scope_standard_deviation in SCOPE_SPACE:
        _mu, _sigma = calculate_weighted_params(x_data, MU, scope_standard_deviation)
        mu_list.append(_mu)
        cs_list.append(_sigma)

    # Find zeros of d²mu_list and d²cs_list
    mu_zeros = np.where(np.diff(np.sign(np.gradient(np.gradient(mu_list)))))[0]
    cs_zeros = np.where(np.diff(np.sign(np.gradient(np.gradient(cs_list)))))[0]

    # Print zeros
    # print("mu zeros: s=", SCOPE_SPACE[mu_zeros])
    # print("cs zeros: s=", SCOPE_SPACE[cs_zeros])

    # Find a pair of zeros from mu_zeros, cs_zeros that minimizes the difference
    min_difference = 1e10
    min_mu, min_s = 0, 0
    for i in mu_zeros:
        for j in cs_zeros:
            if abs(SCOPE_SPACE[i] - SCOPE_SPACE[j]) < min_difference:
                min_difference = abs(SCOPE_SPACE[i] - SCOPE_SPACE[j])
                min_mu = SCOPE_SPACE[i]
                min_s = SCOPE_SPACE[j]

    # Print the 2 different minimizers
    μ1 = mu_list[np.where(SCOPE_SPACE == min_mu)[0][0]]
    σ1 = cs_list[np.where(SCOPE_SPACE == min_mu)[0][0]]
    s1 = SCOPE_SPACE[np.where(SCOPE_SPACE == min_mu)[0][0]]
    # print(f"{μ1=}, {σ1=}, {s1=}")
    μ2 = mu_list[np.where(SCOPE_SPACE == min_s)[0][0]]
    σ2 = cs_list[np.where(SCOPE_SPACE == min_s)[0][0]]
    s2 = SCOPE_SPACE[np.where(SCOPE_SPACE == min_s)[0][0]]
    # print(f"{μ2=}, {σ2=}, {s2=}")

    # If plot=True, plot mu_list and cs_list vs s_list
    if plot:

        # Create a figure with 2 subplots on top of each other who share the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Normalize mu_list and its derivative
        d_mu_list = np.gradient(mu_list)
        mu_list /= max(abs(np.array(mu_list)))
        d_mu_list /= max(abs(d_mu_list))

        # Plot mu_list vs s_list
        ax1.set_title("Weighted Mean vs. Scope Standard Deviation")
        ax1.plot(SCOPE_SPACE, mu_list, label="μ")
        ax1.plot(SCOPE_SPACE, d_mu_list, label="dμ/ds")
        ax1.vlines(min_s, min(min(mu_list), min(d_mu_list)), max(max(mu_list), max(d_mu_list)), colors='r', linestyles='dashed')
        ax1.vlines(min_mu, min(min(mu_list), min(d_mu_list)), max(max(mu_list), max(d_mu_list)), colors='b', linestyles='dashed')

        # Normalize mu_list and its derivative
        d_cs_list = np.gradient(cs_list)
        cs_list /= max(abs(np.array(cs_list)))
        d_cs_list /= max(abs(d_cs_list))

        # Plot cs_list vs s_list
        ax2.set_title("Weighted Standard Deviation vs. Scope Standard Deviation")
        ax2.plot(SCOPE_SPACE, cs_list, label="σ")
        ax2.plot(SCOPE_SPACE, d_cs_list, label="dσ/ds")
        ax2.vlines(min_s, min(min(cs_list), min(d_cs_list)), max(max(cs_list), max(d_cs_list)), colors='r', linestyles='dashed')
        ax2.vlines(min_mu, min(min(cs_list), min(d_cs_list)), max(max(cs_list), max(d_cs_list)), colors='b', linestyles='dashed')

        # Show plot
        plt.legend()
        plt.show()

    # Return the 2 different minimizers
    return (μ1, σ1, s1), (μ2, σ2, s2)



data = np.loadtxt("oil_drop/compiled_data.txt", skiprows=1, encoding='utf-8-sig')
data /= 1.602176634  # Convert Q -> N*e

# Find equilibrium params
G1, G2 = find_equilibrium_params(data, 2, plot=True)
print(f"MU=2: {G1, G2}")
