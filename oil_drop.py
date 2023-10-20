import numpy as np

M = 2.04       # Unitless
δM = 0.05      # Unitless
η = 1.85e-5    # Pa s
ρ_air = 1.2    # kg/m^3
ρ_oil = 850    # kg/m^3
d = 5.91e-3    # m
δd = 0.05e-3   # m
A = 0.07776e-6 # m
g = 9.8        # m/s^2




def q(r: float, v_g: float, v_u: float, V: float):
    return (6 * np.pi * η * d * (v_g + v_u) * r) / V

def q_c(q: float, r: float):
    return q / (1 + A/r)**1.5


# Data compile functions
def compile_v():

    # Load data
    data = np.loadtxt("oil_drop/data_raw.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    V = data[:, 0]
    Δy_g, Δt_g = data[:, 1], data[:, 2]
    Δy_u, Δt_u = data[:, 3], data[:, 4]

    # Define v,δv function
    def v_δv(Δy: float, Δt: float):
        δy = 0  # [TODO : Add uncertainty]
        δt = 0
        _v = (Δy/Δt)/M
        _dv = _v * np.sqrt( (δy/Δy)**2 + (δt/Δt)**2 + (δM/M)**2 )
        return _v, _dv

    # Calculate v,δv (mm/s)
    v_g, δv_g = v_δv(Δy_g, Δt_g)
    v_u, δv_u = v_δv(Δy_u, Δt_u)

    # Save data to file
    np.savetxt("oil_drop/data_velocities.txt", np.transpose([V, v_g, δv_g, v_u, δv_u]), header="V (V), v_g (mm/s), δv_g (mm/s), v_u (mm/s), δv_u (mm/s)", fmt="%f", encoding='utf-8-sig')

def compile_r():

    # Load data
    data = np.loadtxt("oil_drop/data_velocities.txt", skiprows=1, encoding='utf-8-sig')

    # Break data into columns
    v_g = data[:, 1]

    # Define r function
    def r(v_g: float):
        return np.sqrt( 9*η*v_g/(2*(ρ_oil - ρ_air)*g) )
    
    # Define dr function [TODO: Finish this]
    def dr():
        return np.zeros(len(r_g))

    # Calculate r
    r_g = r(v_g * 1e-3)  # Convert velocity mm/s -> m/s
    
    # Calculate dr
    dr_g = dr()

    # Save data to file
    np.savetxt("oil_drop/data_radii.txt", np.transpose([data[:, 0], r_g]), header="V (V), r (m)", fmt="%f", encoding='utf-8-sig')

def compile_q():

    # Load data
    vel_data = np.loadtxt("oil_drop/data_velocities.txt", skiprows=1, encoding='utf-8-sig')
    rad_data = np.loadtxt("oil_drop/data_radii.txt", skiprows=1, encoding='utf-8-sig')

    # Calculate q
    r = rad_data[:, 1]
    V = vel_data[:, 0]
    v_g = vel_data[:, 1]
    v_u = vel_data[:, 2]
    Q = q(r, v_g * 1e-3, v_u * 1e-3, V)  # Convert to m/s

    # Save data to file
    np.savetxt("oil_drop/data_charges.txt", np.transpose([V, Q]), header="V (V), Q (m^3/s)", fmt="%f")

def compile_q_c():

    # Load data
    rad_data = np.loadtxt("oil_drop/data_radii.txt", skiprows=1, encoding='utf-8-sig')
    chr_data = np.loadtxt("oil_drop/data_charges.txt", skiprows=1, encoding='utf-8-sig')

    # Calculate q_c
    Q = chr_data[:, 1]
    r = rad_data[:, 1]
    Q_c = q_c(Q, r)

    # Save data to file
    np.savetxt("oil_drop/data_charges_corrected.txt", np.transpose([data[:, 0], Q_c]), header="V (V), Q_c (m^3/s)", fmt="%f")


compile_v()