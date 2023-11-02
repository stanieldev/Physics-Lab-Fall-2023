# Create a special function for the gaussian
# def damped_gaussian(x, A, b, e, σ):
#     def decay(n): return np.exp(-b*e*(n-1))
#     def gauss(n): return np.exp(-(x - e*n)**2/(2*σ**2))
#     return A * sum([decay(i) * gauss(i) for i in range(1, 20)])


# # Statistical analysis functions
# def frequency_list(data, Ndx, maxval=2.5):

#     # Calculate dx
#     Δx = maxval/Ndx
#     histogram_list = [i*Δx for i in range(0, Ndx + 1)]
    
#     # Calculate frequency of each charge in each bin
#     frequency_list = []
#     for i in range(Ndx):
#         frequency_list.append(len(data[(data >= histogram_list[i])]) - len(data[(data >= histogram_list[i+1])]))
#     frequency_list.append(0)  # Add 0 to the end of the list for values >= maxval
#     frequency_list = np.array(frequency_list)

#     # Return frequency list and histogram list
#     return frequency_list, [i + Δx/2 for i in histogram_list]

# def find_gaussian(data, Ndx, maxval):

#     # Create a frequency list of the data
#     freq, hist = frequency_list(data, Ndx, maxval)

#     # Fit gaussian to data
#     popt, pcov = curve_fit(gaussian, hist, freq, p0=[1, 1, 1])

#     # Return gaussian parameters
#     return popt

# def find_damped_gaussian(data, Ndx, maxval):

#     # Create a frequency list of the data
#     freq, hist = frequency_list(data, Ndx, maxval=maxval)

#     # Fit gaussian to data
#     popt, pcov = curve_fit(damped_gaussian, hist, freq, p0=[max(freq), 0.6, 1.6, 0.25])

#     # Return gaussian parameters
#     return popt




# # Define weighted mean and standard deviation function
# def calculate_weighted_params(x_data, mu, s):

#     # Calculate a weight proportional to the CDF of the z-score about mu
#     p = 1 - erf((abs(x_data - mu)/s)/np.sqrt(2))
#     p /= sum(p)  # Normalize p
#     N = len(x_data)
    
#     # Calculate weighted mean and standard deviation
#     weighted_mean = sum(x_data * p)
#     weighted_std = np.sqrt(N/(N-1)) * np.sqrt(sum(p * (x_data - weighted_mean)**2))

#     # Return weighted mean and standard deviation
#     return weighted_mean, weighted_std

# def find_equilibrium_params(x_data, MU, plot=False):

#     # Find equilibrium params
#     SCOPE_SPACE = np.linspace(0.05, 1, 1000)[1:]
#     mu_list = []
#     cs_list = []
#     for scope_standard_deviation in SCOPE_SPACE:
#         _mu, _sigma = calculate_weighted_params(x_data, MU, scope_standard_deviation)
#         mu_list.append(_mu)
#         cs_list.append(_sigma)

#     # Find zeros of d²mu_list and d²cs_list
#     mu_zeros = np.where(np.diff(np.sign(np.gradient(np.gradient(mu_list)))))[0]
#     cs_zeros = np.where(np.diff(np.sign(np.gradient(np.gradient(cs_list)))))[0]

#     # Find a pair of zeros from mu_zeros, cs_zeros that minimizes the difference
#     min_difference = 1e10
#     min_mu, min_s = 0, 0
#     for i in mu_zeros:
#         for j in cs_zeros:
#             if abs(SCOPE_SPACE[i] - SCOPE_SPACE[j]) < min_difference:
#                 min_difference = abs(SCOPE_SPACE[i] - SCOPE_SPACE[j])
#                 min_mu = SCOPE_SPACE[i]
#                 min_s = SCOPE_SPACE[j]

#     # Print the 2 different minimizers
#     μ1 = mu_list[np.where(SCOPE_SPACE == min_mu)[0][0]]
#     σ1 = cs_list[np.where(SCOPE_SPACE == min_mu)[0][0]]
#     s1 = SCOPE_SPACE[np.where(SCOPE_SPACE == min_mu)[0][0]]
#     μ2 = mu_list[np.where(SCOPE_SPACE == min_s)[0][0]]
#     σ2 = cs_list[np.where(SCOPE_SPACE == min_s)[0][0]]
#     s2 = SCOPE_SPACE[np.where(SCOPE_SPACE == min_s)[0][0]]

#     # If plot=True, plot mu_list and cs_list vs s_list
#     if plot:

#         # Create a figure with 2 subplots on top of each other who share the x-axis
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

#         # Normalize mu_list and its derivative
#         d_mu_list = np.gradient(mu_list)
#         mu_list /= max(abs(np.array(mu_list)))
#         d_mu_list /= max(abs(d_mu_list))

#         # Plot mu_list vs s_list
#         ax1.set_title("Weighted Mean vs. Scope Standard Deviation")
#         ax1.plot(SCOPE_SPACE, mu_list, label="μ")
#         ax1.plot(SCOPE_SPACE, d_mu_list, label="dμ/ds")
#         ax1.vlines(min_s, min(min(mu_list), min(d_mu_list)), max(max(mu_list), max(d_mu_list)), colors='r', linestyles='dashed')
#         ax1.vlines(min_mu, min(min(mu_list), min(d_mu_list)), max(max(mu_list), max(d_mu_list)), colors='b', linestyles='dashed')

#         # Normalize mu_list and its derivative
#         d_cs_list = np.gradient(cs_list)
#         cs_list /= max(abs(np.array(cs_list)))
#         d_cs_list /= max(abs(d_cs_list))

#         # Plot cs_list vs s_list
#         ax2.set_title("Weighted Standard Deviation vs. Scope Standard Deviation")
#         ax2.plot(SCOPE_SPACE, cs_list, label="σ")
#         ax2.plot(SCOPE_SPACE, d_cs_list, label="dσ/ds")
#         ax2.vlines(min_s, min(min(cs_list), min(d_cs_list)), max(max(cs_list), max(d_cs_list)), colors='r', linestyles='dashed')
#         ax2.vlines(min_mu, min(min(cs_list), min(d_cs_list)), max(max(cs_list), max(d_cs_list)), colors='b', linestyles='dashed')

#         # Show plot
#         plt.legend()
#         plt.show()

#     # Return the 2 different minimizers
#     return (μ1, σ1, s1), (μ2, σ2, s2)




# # Find equilibrium params
# e_data = np.loadtxt("oil_drop/compiled_data.txt", skiprows=1, encoding='utf-8-sig')
# e_data /= 1.602176634  # Convert Q -> N*e

# # Plot histogram
# N=5
# plot_histogram(plt, e_data,
#                 intermediate_tick_count=N,
#                 force_min_x=0,
#                 force_max_x=10,
#                 show_minor_ticks=False,
#                 dy=8)

# # Frequency list
# f, d = frequency_list(e_data, N * 10, maxval=10)

# # Plot equilibrium params
# for k in range(1, 5):
#     X = np.linspace(0, 10, 1000)
#     G1, G2 = find_equilibrium_params(e_data, k, plot=False)

#     # Graph 1
#     MU = G1[0]
#     S = G1[1]
#     A = f[int(MU * N)]
#     plt.plot(X, A*np.exp(-(X - MU)**2/(2*S**2)), label="Gaussian 1", color='r')

#     # Graph 2
#     MU = G2[0]
#     S = G2[1]
#     A = f[int(MU * N)]
#     plt.plot(X, A*np.exp(-(X - MU)**2/(2*S**2)), label="Gaussian 2", color='b')

# # Show plot
# plt.show()