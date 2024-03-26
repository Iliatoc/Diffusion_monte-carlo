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

step = 1

def random_point_on_unit_sphere():
    """
    Generate a random point on a unit sphere using Marsaglia's method.
    Returns:
        Tuple (x, y, z): Coordinates of the random point.
    """
    while True:
        # Generate three random numbers in the range [-1, 1]
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        
        # Check if the point lies within the unit sphere
        if x**2 + y**2 + z**2 <= 1:
            break
    
    # Normalize the point to lie on the unit sphere
    length = np.sqrt(x**2 + y**2 + z**2)
    x /= length
    y /= length
    z /= length

    return np.array([x, y, z])

def random_directions(n):
    vectors = []
    for _ in range(n):
        vectors.append(random_point_on_unit_sphere())
    return np.array(vectors)

def set_start_positions(n, starting_vector):
    return np.tile(starting_vector, (n, 1))

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

def rayleigh_scattering_random(directions, theta_array_sin_sqr):
    loc_theta_0 = np.arccos(directions[:, 2] / np.linalg.norm(directions, axis=1))
    loc_phi_0 = np.sign(directions[:, 1]) * np.arccos(directions[:, 0] / np.sqrt(directions[:, 0]**2 + directions[:, 1]**2))
    loc_phi = np.random.uniform(0, 2 * math.pi, len(directions))
    loc_theta_index = np.random.randint(len(theta_array_sin_sqr), size=len(directions))
    loc_theta = theta_array_sin_sqr[loc_theta_index]
    loc_x = step * np.sin(loc_theta) * np.cos(loc_phi)
    loc_y = step * np.sin(loc_theta) * np.sin(loc_phi)
    loc_z = step * np.cos(loc_theta)
    loc_rot_y = np.array([loc_x * np.cos(loc_theta_0) + loc_z * np.sin(loc_theta_0),
                          loc_y,
                          loc_z * np.cos(loc_theta_0) - loc_x * np.sin(loc_theta_0)]).T
    loc_rot_z = np.array([loc_rot_y[:, 0] * np.cos(loc_phi_0) - loc_rot_y[:, 1] * np.sin(loc_phi_0),
                          loc_rot_y[:, 0] * np.sin(loc_phi_0) + loc_rot_y[:, 1] * np.cos(loc_phi_0),
                          loc_rot_y[:, 2]]).T
    loc_direction_vector = loc_rot_z
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

def inside_cylinder(positions, h, r):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    return (x**2 + y**2 <= r**2) & (-h/2 <= z) & (z <= h/2)

def photon_interactions(photon_num, chance_per_micron = 0.5):
    i_seed = np.random.rand(photon_num,1)
    return np.where(i_seed<=chance_per_micron, True, False)
    
    
def iter_step(mask,inter_mask, positions, directions):
    '''i_seed = random.randrange(int(1000/chance_of_interaction_per_micron))
    if i_seed <= chance_of_interaction_per_micron*1000:
        direction_vector = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])'''
    
    return np.where(mask[:, np.newaxis], positions + directions, positions)
def main():
    print("hello")
    theta_array_homo = sin_distribution(10 ** 2, 10 ** 3)
    theta_array_sin_sqr = sin_squared_distribution(10 ** 2, 10 ** 3)

    height, radius, collection_sphere_radius = 1*10**4, 1.5*10**4, 8*10**4     # microns, micrometers everywhere
    chance_of_interaction_per_micron = 0.05   # percent
    R = collection_sphere_radius
    n_photons = 5000
    photon_directions = random_directions(n_photons)
    photon_locs = set_start_positions(n_photons, np.array([10**4, 0, 0]))

    start_time = time.time()

    max_iterations = 10**4

    for _ in range(max_iterations):
        interaction_mask = photon_interactions(n_photons)
        cylinder_mask = inside_cylinder(photon_locs, height, radius)
        if np.all(inside_cylinder) == False:
            break
        photon_directions = np.where(interaction_mask & cylinder_mask[:, np.newaxis], rayleigh_scattering_random(photon_directions, theta_array_sin_sqr), photon_directions)
        photon_locs = np.where(cylinder_mask[:, np.newaxis], photon_locs + photon_directions, photon_locs)

    r_locs = np.linalg.norm(photon_locs, axis=1)
    a = np.sum(photon_directions**2, axis=1)
    b = 2*np.sum(photon_locs*photon_directions,axis=1)
    c = r_locs**2 - R**2
    n = ((-(b) + np.sqrt(b**2 - 4*a*c))/(2*a))[:, np.newaxis]
    results_arr = photon_locs + photon_directions*n

    print("--- %s seconds ---" % (time.time() - start_time))
if __name__=="__main__":
    main()