import random
import pandas as pd
from PIL import ImageTk, Image
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import lfilter
from scipy.interpolate import CubicSpline
import math
import copy
from matplotlib import ticker, cm
import itertools
import miepython
import time


def rayleigh(m,x):
    """
    Calculate the efficiencies for a small sphere.

    Based on equations 5.7 - 5.9 in Bohren and Huffman

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ratio = (m**2-1)/(m**2+2)
    qsca = 8/3*x**4*abs(ratio)**2
    qext = 4*x*ratio*(1+x**2/15*ratio*(m**4+27*m**2+38)/(2*m**2+3))
    qext = abs(qext.imag + qsca)
    qback = 4*x**4*abs(ratio)**2
    g = 0
    return qext, qsca, qback, g


def rayleigh_S1_S2(m,x,mu):
    """
    Calculate the scattering amplitude functions for small spheres.

    Based on equation 5.4 in Bohren and Huffman

    The amplitude functions are normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """

    a1 = (2*x**3)/3 * (m**2-1)/(m**2+2)*1j
    a1 += (2*x**5)/5 * (m**2-2)*(m**2-1)/(m**2+2)**2 *1j

    s1 = (3/2)*a1*np.ones_like(mu)
    s2 = (3/2)*a1*mu

    ## scale so integral over all angles is single scattering albedo
    qext, qsca, qback, g = rayleigh(m,x)

    factor = np.sqrt(np.pi*qext)*x
    return s1/factor, s2/factor


def rayleigh_unpolarized(m,x,mu):
    """
    Return the unpolarized scattered intensity for small spheres.

    This is the average value for randomly polarized incident light.
    The intensity is normalized so the integral of the unpolarized
    intensity over 4pi steradians is equal to the single scattering albedo.

    Args:
       m: the complex index of refraction of the sphere
       x: the size parameter
       mu: the cos(theta) of each direction desired

    Returns
       The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = rayleigh_S1_S2(m,x,mu)
    return (abs(s1)**2+abs(s2)**2)/2


def data_for_cylinder_along_z(center_x, center_y, loc_radius, height_z):
    loc_z = np.linspace(-height_z/2, height_z/2, 50)
    loc_theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(loc_theta, loc_z)
    x_grid = loc_radius * np.cos(theta_grid) + center_x
    y_grid = loc_radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def light_path_iteration():
    phi = np.random.uniform(0, 2 * math.pi)
    theta_cos = np.random.uniform(-1, 1)
    theta = np.arccos(theta_cos)
    theta = theta * (np.sin(theta) ** 2)
    loc_x = step * np.sin(theta) * np.cos(phi)
    loc_y = step * np.sin(theta) * np.sin(phi)
    loc_z = step * np.cos(theta)
    loc_direction_vector = np.array([loc_x, loc_y, loc_z])
    return loc_direction_vector


def random_vector():
    # number_of_randoms = 10**6
    loc_phi = np.random.uniform(0, 2 * math.pi)
    # loc_theta_cos = np.random.uniform(-1, 1)
    # loc_theta = np.arccos(loc_theta_cos)
    loc_theta_index = np.random.randint(len(theta_array_homo))
    loc_theta = theta_array_homo[loc_theta_index]
    loc_x = step * np.sin(loc_theta) * np.cos(loc_phi)
    loc_y = step * np.sin(loc_theta) * np.sin(loc_phi)
    loc_z = step * np.cos(loc_theta)
    loc_direction_vector = np.array([loc_x, loc_y, loc_z])
    return loc_direction_vector


def rayleigh_scattering_random(dir_in_x, dir_in_y, dir_in_z):
    loc_theta_0 = np.arccos(dir_in_z/np.sqrt(dir_in_x**2 + dir_in_y**2 + dir_in_z**2))
    loc_phi_0 = np.sign(dir_in_y) * np.arccos(dir_in_x/np.sqrt(dir_in_x**2 + dir_in_y**2))
    loc_phi = np.random.uniform(0, 2 * math.pi)
    loc_theta_index = np.random.randint(len(theta_array_sin_sqr))
    loc_theta = theta_array_sin_sqr[loc_theta_index]
    loc_x = step * np.sin(loc_theta) * np.cos(loc_phi)
    loc_y = step * np.sin(loc_theta) * np.sin(loc_phi)
    loc_z = step * np.cos(loc_theta)
    loc_rot_y = np.array([loc_x * np.cos(loc_theta_0) + loc_z * np.sin(loc_theta_0), loc_y, loc_z * np.cos(loc_theta_0)
                          - loc_x * np.sin(loc_theta_0)])
    loc_rot_z = np.array([loc_rot_y[0] * np.cos(loc_phi_0) - loc_rot_y[1] * np.sin(loc_phi_0), loc_rot_y[0] *
                          np.sin(loc_phi_0) + loc_rot_y[1] * np.cos(loc_phi_0), loc_rot_y[2]])
    loc_direction_vector = loc_rot_z
    #print(loc_direction_vector)
    return loc_direction_vector


def sin_distribution(num_of_sins, random_max):
    # Actually a homogeneous distribution
    sin_array = np.array([0])
    for sin_iter in range(1, num_of_sins):
        loc_theta = sin_iter * math.pi / num_of_sins
        sin_array = np.append(sin_array, np.ones(int(np.sin(loc_theta) * random_max)) * loc_theta)
    return sin_array


def sin_squared_distribution(num_of_sins, random_max):
    # sin(theta)**2 distribution in surface
    sin_array = np.array([0])
    for sin_iter in range(1, num_of_sins):
        loc_theta = sin_iter * math.pi / num_of_sins
        sin_array = np.append(sin_array, np.ones(int((np.sin(loc_theta)**3) * random_max * 1000)) * loc_theta)
    return sin_array


height = 1*10**4     # microns, micrometers everywhere
radius = 1.5*10**4
collection_sphere_radius = 8*10**4
R = collection_sphere_radius
step = 1           # microns
chance_of_interaction_per_micron = 0.05   # percent
n_photons = 5000
num_of_angles = 10 ** 2
random_iter = 10 ** 3
x_arr = np.zeros(n_photons)
y_arr = np.zeros(n_photons)
z_arr = np.zeros(n_photons)
x_last_arr = np.zeros(n_photons)
y_last_arr = np.zeros(n_photons)
z_last_arr = np.zeros(n_photons)
rand_range = 1000
start_x = 10**4
start_y = 0
start_z = 0
Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, radius, height)
start_time = time.time()
inside = 1
iteration = 0
theta_r = math.pi / num_of_angles
theta_array_homo = sin_distribution(num_of_angles, random_iter)
theta_array_sin_sqr = sin_squared_distribution(num_of_angles, random_iter)
print("--- %s seconds ---" % (time.time() - start_time))
print("Lowest theta = ", theta_r)
print("Num of lowest theta = ", int(random_iter * np.sin(theta_r)))
print("Num of lowest theta for sin^3 = ", int(random_iter * 1000 * (np.sin(theta_r)**3)))
print("Length of theta homo array = ", len(theta_array_homo))
print("Length of theta sqr array = ", len(theta_array_sin_sqr))
"""for lol in range(4):
    print("Theta (in Pi) = ", theta_r/math.pi)
    print("High for randint = ", int(1000000 * np.sin(theta_r)))
    rand_temp_print = np.random.randint(int(1000000 * np.sin(theta_r))) / 1000000
    print(rand_temp_print)
    theta_r += math.pi / 4"""
# Matrix method
"""position_all = np.zeros((n_photons, 3))
position_all[:, 0] += 10000
vector_all = np.random.rand(n_photons, 3) - 0.5
vector_norm = 1 / np.sum(np.square(vector_all), axis=1)
vector_norm = np.reshape(vector_norm, (n_photons, 1))
vector_all = vector_all * vector_norm
radius_sqr_arr = np.ones(n_photons) * (radius**2)
while iteration < 10 ** 6:
    iteration += 1
    rand_array = np.random.rand(n_photons)
    inside_arr = np.heaviside(position_all[:, 2] + height / 2, 1) - np.heaviside(position_all[:, 2] - height / 2, 1) * \
                 np.heaviside(radius_sqr_arr - (np.square(position_all[:, 0]) + np.square(position_all[:, 1])), 1)
    interaction_array = np.heaviside((chance_of_interaction_per_micron / 100) - rand_array, 1) * inside_arr
    non_interaction_array = 1 - interaction_array
    vector_all[:, 0] = vector_all[:, 0] * non_interaction_array + (np.random.rand(n_photons) - 0.5) * interaction_array
    vector_all[:, 1] = vector_all[:, 1] * non_interaction_array + (np.random.rand(n_photons) - 0.5) * interaction_array
    vector_all[:, 2] = vector_all[:, 2] * non_interaction_array + (np.random.rand(n_photons) - 0.5) * interaction_array
    vector_norm = 1 / np.sum(np.square(vector_all), axis=1)
    vector_norm = np.reshape(vector_norm, (n_photons, 1))
    vector_all = vector_all * vector_norm
    inside_arr = np.reshape(inside_arr, (n_photons, 1))
    position_all = position_all + vector_all * inside_arr
    #inside = max(inside_arr)"""
iteration_array = np.zeros(n_photons)
for i in range(n_photons):
    x = start_x           # + random.randrange(-rand_range*10, rand_range*10)
    y = start_y           # + random.randrange(-rand_range*10, rand_range*10)
    z = start_z           # + random.randrange(-rand_range*20, rand_range*20)
    #x_dir = random.randrange(-rand_range, rand_range)
    #y_dir = random.randrange(-rand_range, rand_range)
    #z_dir = random.randrange(-rand_range, rand_range)
    #v_dir_coeff = math.sqrt(x_dir**2 + y_dir**2 + z_dir**2) / step
    #direction_vector = np.array([x_dir/v_dir_coeff, y_dir/v_dir_coeff, z_dir/v_dir_coeff])
    direction_vector = random_vector()
    inside = 1
    if z >= height/2 or z <= -height/2 or (x**2 + y**2 >= radius**2):
        inside = 0
        break
    iteration = 0
    while inside == 1 and iteration < 10**6:
        iteration += 1
        if z >= height/2 or z <= -height/2 or (x**2 + y**2 >= radius**2):
            inside = 0
            break
        # check_z = np.heaviside(z + height/2, 1) - np.heaviside(z - height/2, 1)
        # check_radius = np.heaviside(radius**2 - (x**2 + y**2), 1)
        # inside = check_radius * check_z
        #heavy = np.heaviside([z + height/2, z - height/2, radius**2 - (x**2 + y**2)], 1)
        #inside = (heavy[0] - heavy[1]) * heavy[2]
        i_seed = random.randrange(int(1000/chance_of_interaction_per_micron))
        if i_seed <= chance_of_interaction_per_micron*1000:
            direction_vector = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])
            #direction_vector = random_vector()
        # temp_direction_vector = light_path_iteration()
        # direction_vector = direction_vector * np.heaviside(i_seed - chance_of_interaction_per_micron * 1000, 1) + \
        #                    temp_direction_vector * np.heaviside(chance_of_interaction_per_micron * 1000 - i_seed, 0)
        x = x + direction_vector[0]
        y = y + direction_vector[1]
        z = z + direction_vector[2]
    if iteration >= 10 ** 6:
        print("Too many iterations!")
        print("Photon number = ", i)
        break
    iteration_array[i] = iteration
    x0 = x
    y0 = y
    z0 = z
    x_last_arr[i] = x
    y_last_arr[i] = y
    z_last_arr[i] = z
    r_loc = math.sqrt(x0**2 + y0**2 + z0**2)
    # x = x0 + direction_vector[0] * n
    # y = y0 + direction_vector[1] * n
    # z = z0 + direction_vector[2] * n
    # x**2 + y**2 + z**2 = collection_sphere_radius**2
    # x0**2 + y0**2 + z0**2 + 2n*(x0*direction_vector[0] + ) + n**2(direction_vector[0]**2 + ) = collection_sphere_radius**2
    a = direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2
    b = 2 * (x0 * direction_vector[0] + y0 * direction_vector[1] + z0 * direction_vector[2])
    c = r_loc**2 - R**2   # should be < 0
    #print(a)
    if (b**2 - 4*a*c) < 0:
        print("ERROR! In the equation for intersection with collection - sqrt less than zero!")
        break
    if a > 0:
        n = (-(b) + math.sqrt(b**2 - 4*a*c))/(2*a)
    elif a < 0:
        print("ERROR! The direction vector is complex!")
        break
    else:
        print("ERROR! The direction vector is zero!")
        break
    x_arr[i] = x0 + direction_vector[0] * n
    y_arr[i] = y0 + direction_vector[1] * n
    z_arr[i] = z0 + direction_vector[2] * n
# Matrix method
"""x0 = position_all[:, 0]
y0 = position_all[:, 1]
z0 = position_all[:, 2]
r_square_arr = np.square(x0) + np.square(y0) + np.square(z0)
R_sqr_arr = np.ones(n_photons) * (R**2)
# x = x0 + direction_vector[0] * n
# y = y0 + direction_vector[1] * n
# z = z0 + direction_vector[2] * n
# x**2 + y**2 + z**2 = collection_sphere_radius**2
# x0**2 + y0**2 + z0**2 + 2n*(x0*direction_vector[0] + ) + n**2(direction_vector[0]**2 + ) = collection_sphere_radius**2
a = np.square(vector_all[:, 0]) + np.square(vector_all[:, 1]) + np.square(vector_all[:, 2])
b = 2 * (x0 * vector_all[:, 0] + y0 * vector_all[:, 1] + z0 * vector_all[:, 2])
c = r_square_arr - R_sqr_arr   # should be < 0
#print(a)
n = (-(b) + np.sqrt(b**2 - 4*a*c))/(2*a)
x_arr = x0 + vector_all[:, 0] * n
y_arr = y0 + vector_all[:, 1] * n
z_arr = z0 + vector_all[:, 2] * n"""
print("--- %s seconds ---" % (time.time() - start_time))
print("Max number of iterations in model = ", max(iteration_array))
#print(x_arr)
#print(y_arr)
# x_ind = np.flatnonzero(x_arr)
# y_ind = np.flatnonzero(y_arr)
# z_ind = np.flatnonzero(z_arr)
# x_coord = x_arr[x_ind]
# y_coord = y_arr[y_ind]
# z_coord = z_arr[z_ind]
x_coord = x_arr
y_coord = y_arr
z_coord = z_arr
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_coord, y_coord, z_coord)
ax.scatter(start_x, start_y, start_z)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
plt.show()
plt.close()
#collected_photons = pd.DataFrame([x_coord, y_coord, z_coord], columns=["x_v", "y_v", "z_v"])
#print(collected_photons)
"""_______________________________Collecting photons into apperture and rotating__________________________________"""
num_angles = 36
counts_zone_phi = np.zeros(num_angles)
counts_zone_theta = np.zeros(num_angles)
h = (10**8)/(2*math.pi*R)
a0 = 1
b0 = 0
c0 = 0
d = R - h
print(d)
angles = np.asarray(range(num_angles)) * 360 / num_angles
for phi_d in range(num_angles):
    phi_rad = phi_d * 2 * math.pi / num_angles
    a = a0 * math.cos(phi_rad)              # - b0 * math.sin(theta_r)
    b = a0 * math.sin(phi_rad)              # + b0 * math.cos(theta_r)
    #a = a0 * math.cos(phi_rad)
    #c = - a0 * math.sin(phi_rad)
    counts = 0
    for k in range(len(x_coord)):
        if a * x_coord[k] + b * y_coord[k] >= d:
            counts += 1
    counts_zone_phi[phi_d] = counts
plt.plot(angles, counts_zone_phi, label="Horizontal")
for theta_d in range(num_angles):
    theta_rad = theta_d * 2 * math.pi / num_angles
    a = a0 * math.cos(theta_rad)              # - b0 * math.sin(theta_r)
    c = - a0 * math.sin(theta_rad)              # + b0 * math.cos(theta_r)
    counts = 0
    for k in range(len(x_coord)):
        if a * x_coord[k] + c * z_coord[k] >= d:
            counts += 1
    counts_zone_theta[theta_d] = counts
plt.plot(angles, counts_zone_theta, label="Vertical")
print("--- %s seconds ---" % (time.time() - start_time))
plt.xlabel("Angle, degrees")
plt.ylabel("Counts")
plt.legend()
plt.show()
plt.close()
