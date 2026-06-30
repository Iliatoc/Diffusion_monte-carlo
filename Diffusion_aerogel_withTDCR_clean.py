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
#import miepython
import time


class Photon:
    def __init__(self, number, fin_x, fin_y, fin_z, batch, detector_n):
        self.number = number
        self.fin_x = fin_x
        self.fin_y = fin_y
        self.fin_z = fin_z
        #self.dir_x = dir_x
        #self.dir_y = dir_y
        #self.dir_z = dir_z
        self.batch = batch
        self.detector_n = detector_n
        #self.time = time
    """def transform_surf(self):
        # new_class = my_class
        new_class = Surface(str(self.number), transform_into_str(self.indices), transform_into_str(self.x_scale),
                            transform_into_str(self.y_scale), transform_into_str(self.z_scale), transform_into_str(self.omega),
                            transform_into_str(self.theta), transform_into_str(self.phi), transform_into_str(self.x_shift),
                            transform_into_str(self.y_shift), transform_into_str(self.z_shift))
        return new_class"""


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


def circle_points(loc_r, loc_step):
    loc_coord = np.array([[0, 0]])
    for loc_i in range(-loc_r, loc_r, loc_step):
        for loc_k in range(-loc_r, loc_r, loc_step):
            if loc_i**2 + loc_k**2 <= loc_r**2:
                loc_temp = np.array([[loc_i, loc_k]])
                loc_coord = np.append(loc_coord, loc_temp, axis=0)
    return loc_coord


def random_vector(loc_theta_array, loc_theta_index):
    # number_of_randoms = 10**6
    loc_phi = np.random.uniform(0, 2 * math.pi)
    # loc_theta_cos = np.random.uniform(-1, 1)
    # loc_theta = np.arccos(loc_theta_cos)
    loc_theta = loc_theta_array[loc_theta_index]
    loc_x = step * np.sin(loc_theta) * np.cos(loc_phi)
    loc_y = step * np.sin(loc_theta) * np.sin(loc_phi)
    loc_z = step * np.cos(loc_theta)
    loc_direction_vector = np.array([loc_x, loc_y, loc_z])
    return loc_direction_vector


def rayleigh_scattering_random(dir_in_x, dir_in_y, dir_in_z):
    loc_theta_0 = np.arccos(dir_in_z/np.sqrt(dir_in_x**2 + dir_in_y**2 + dir_in_z**2))
    loc_phi_0 = np.sign(dir_in_y) * np.arccos(dir_in_x/np.sqrt(dir_in_x**2 + dir_in_y**2))
    loc_phi = np.random.uniform(0, 2 * math.pi)
    loc_theta_index = np.random.randint(len(theta_array_cos_sqr_plus1))
    loc_theta = theta_array_cos_sqr_plus1[loc_theta_index]
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
    loc_sin_array = np.array([0])
    for sin_iter in range(1, num_of_sins - 1):
        loc_theta = sin_iter * math.pi / num_of_sins
        loc_sin_array = np.append(loc_sin_array, np.ones(int(np.sin(loc_theta) * random_max)) * loc_theta)
    loc_sin_array = np.append(loc_sin_array, math.pi)
    return loc_sin_array


def cos_squared_plus_one_distribution(num_of_angl, random_max):
    # sin(theta)**2 distribution in surface
    loc_ray_array = np.array([0])
    for ray_iter in range(1, num_of_angl - 1):
        loc_theta = ray_iter * math.pi / num_of_angl
        loc_ray_array = np.append(loc_ray_array, np.ones(int((np.sin(loc_theta) * ((np.cos(loc_theta)**2) + 1)) *
                                                             random_max * 1000)) * loc_theta)
    loc_ray_array = np.append(loc_ray_array, math.pi)
    return loc_ray_array


def cumulative_probability_distance():
    loc_num_steps = int(np.sqrt((radius * 2) ** 2 + height ** 2))
    temp_step_0 = chance_of_interaction_per_micron / 100
    temp_sum_prob = 0
    temp_step_arr = np.zeros(loc_num_steps)
    temp_sum_prob_arr = np.zeros(loc_num_steps)
    range_of_steps = np.arange(loc_num_steps)
    for loc_i in range_of_steps:
        temp_step = (1 - temp_sum_prob) * temp_step_0
        temp_step_arr[loc_i] = temp_step
        temp_sum_prob = temp_sum_prob + temp_step
        temp_sum_prob_arr[loc_i] = temp_sum_prob
    loc_multiple_min = 10
    loc_min_number = loc_multiple_min / temp_step
    print("Lowest number of cumulative prob slots = ", loc_multiple_min)
    print("First step of the cumulative prob = ", loc_min_number * temp_step_0)
    if int(temp_step_0 * loc_min_number) * loc_num_steps > 10**8:
        print("POWER! UNLIMITED POWER!!! The array for the random interaction may be too big")
    loc_interaction_array = np.ones(int(temp_step_0 * loc_min_number)) * step
    for loc_ii in range_of_steps:
        loc_interaction_array = np.append(loc_interaction_array, np.ones(int(temp_step_arr[loc_ii] * loc_min_number)) * loc_ii)
    # print(temp_step)
    # print(temp_sum_prob)
    loc_interaction_array = np.append(loc_interaction_array, np.ones(int(len(loc_interaction_array) * (1 - temp_sum_prob))) * (loc_num_steps+1))
    plt.plot(range_of_steps, temp_step_arr)
    plt.plot(range_of_steps, temp_sum_prob_arr)
    plt.grid()
    plt.yscale("log")
    #plt.show()
    plt.close()
    return loc_interaction_array


number_of_photons_per_decay = 50
ph_yield_step = 10
#ph_per_decay_array = np.arange(10, 50 + ph_yield_step, ph_yield_step)
#ph_per_decay_array = np.array([10, 20, 30, 40, 50, 60, 100, 150, 200, 300, 400, 500])
ph_per_decay_array = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20])
#ph_per_decay_array = np.arange(10, 260, 10)
ph_per_decay_array_core = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20])
num_steps = 1
laser_r = int(1.25 * 10**4)
laser_step = 100
laser_array = circle_points(laser_r, laser_step)
laser_array_length, trashy = laser_array.shape
print("Laser array shape = ", laser_array_length)
#print("Laser array = ", laser_array)
# aerogel dimensions
height = 3 * 10**4     # microns, micrometers everywhere
radius = int(1.25 * 10**4)
n_air = 1                # index of refraction
n_aerogel = 1.15            # index of refraction
angle_of_total_reflection = np.arcsin(n_air / n_aerogel)
step = 1                   # microns
#chance_of_interaction_array = [0.0001, 0.005, 0.01, 0.02, 0.025, 0.03]
#chance_of_interaction_array = [0.04, 0.05]
chance_of_interaction_per_micron = 0.0265   # percent
#chance_of_interaction_per_micron = 0.0001
start_array_step = 10**3    # step for the source of light
# cylinder_start_array_x = np.zeros(int(height * radius * radius * 4 / (start_array_step**2)))
# cylinder_start_array_y = np.zeros(int(height * radius * radius * 4 / (start_array_step**2)))
# cylinder_start_array_z = np.zeros(int(height * radius * radius * 4 / (start_array_step**2)))
cylinder_start_array_x = np.array([0])
cylinder_start_array_y = np.array([0])
cylinder_start_array_z = np.array([0])
radius_start = radius
"""for c_s_i_v in range(int(height / start_array_step)):
    loc_csi_z_coord = c_s_i_v * start_array_step - (height / 2) + (height / (start_array_step * 2))
    cylinder_start_array_z = np.append(cylinder_start_array_z, loc_csi_z_coord)
for c_s_i_hx in range(int(radius_start * 2 / start_array_step)):
    for c_s_i_hy in range(int(radius_start * 2 / start_array_step)):
        #loc_index_csi = c_s_i_v * int(radius_start * radius_start * 4 / (start_array_step**2)) + c_s_i_hx * int(radius_start * 2 / start_array_step) + c_s_i_hy
        #loc_csi_z_coord = c_s_i_v * start_array_step - (height / 2) + (height / (start_array_step * 2))
        loc_csi_x_coord = c_s_i_hx * start_array_step - radius_start + (radius_start / start_array_step)
        loc_csi_y_coord = c_s_i_hy * start_array_step - radius_start + (radius_start / start_array_step)
        if loc_csi_x_coord**2 + loc_csi_y_coord**2 <= radius_start**2:
            cylinder_start_array_x = np.append(cylinder_start_array_x, loc_csi_x_coord)
            cylinder_start_array_y = np.append(cylinder_start_array_y, loc_csi_y_coord)
            #cylinder_start_array_z = np.append(cylinder_start_array_z, loc_csi_z_coord)
print("Cylinder source x starting values: ", cylinder_start_array_x)"""
# collection of photons param
collection_sphere_radius = 6 * 10**4
apperture_radius = 3 * 10**4
R = collection_sphere_radius
source_step = 4 * 10**3
for grand_i in range(len(ph_per_decay_array_core)):
    number_of_photons_per_decay = ph_per_decay_array[grand_i]
    start_x = 0  #+ (source_step * ii)
    start_y = 0
    start_z = 0
    # start_z = height/2 - 1
    new_start_x = start_x
    new_start_y = start_y
    new_start_z = start_z
    num_of_zeroes = 0
    #chance_of_interaction_per_micron = chance_of_interaction_array[grand_i]   # percent
    n_photons = 500000
    leftover_photons = n_photons
    temp_list_batches = []
    while leftover_photons > 0:
        temp_this_decay = np.random.poisson(ph_per_decay_array_core[grand_i])
        temp_list_batches.append(temp_this_decay)
        leftover_photons = leftover_photons - temp_this_decay
    #num_of_decays = int(n_photons/number_of_photons_per_decay)
    num_of_decays = int(len(temp_list_batches))
    temp_batch_array = np.asarray(temp_list_batches)
    num_angles = 3                     # for collection
    num_of_angles = 10 ** 2             # for random theta arrays
    random_iter = 10 ** 3
    x_arr = np.zeros(n_photons)
    y_arr = np.zeros(n_photons)
    z_arr = np.zeros(n_photons)
    x_last_arr = np.zeros(n_photons)
    y_last_arr = np.zeros(n_photons)
    z_last_arr = np.zeros(n_photons)
    r_h = np.zeros((num_angles, 3))
    r_v = np.zeros((num_angles, 3))
    rand_range = 1000
    start_x = 0
    start_y = 0
    #start_z = 0
    start_z = height/2 - 1
    #Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, radius, height)
    start_time = time.time()
    inside = 1
    iteration = 0
    theta_r = math.pi / num_of_angles
    theta_array_homo = sin_distribution(num_of_angles, random_iter)
    theta_array_cos_sqr_plus1 = cos_squared_plus_one_distribution(num_of_angles, random_iter)
    #interaction_prob_arr = cumulative_probability_distance()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Lowest theta = ", theta_r)
    print("Num of lowest theta = ", int(random_iter * np.sin(theta_r)))
    print("Num of lowest theta for Rayleigh = ", int(random_iter * (np.sin(theta_r) * ((np.cos(theta_r)**2) + 1))))
    print("Length of theta homo array = ", len(theta_array_homo))
    print("Length of theta Rayleigh array = ", len(theta_array_cos_sqr_plus1))
    #print("Length of interaction array = ", len(interaction_prob_arr))
    iteration_array = np.zeros(n_photons)
    fig, ax = plt.subplots(2, 2)
    photon_list = []
    """_______________________start of the loop responsible for each photon packet (decay event)_____________________"""
    for ii in range(num_steps):
        batch_counter = 0   # inside the batch
        batch_array = np.zeros((num_steps, num_angles, num_of_decays+1))
        used_batch_photons = 0
        for i in range(n_photons):
            if used_batch_photons == temp_batch_array[batch_counter]:
                #my_index_source_xy = np.random.randint(len(cylinder_start_array_x))
                #my_index_source_z = np.random.randint(len(cylinder_start_array_z))
                #new_start_x = start_x + cylinder_start_array_x[my_index_source_xy]          # + random.randrange(-rand_range*10, rand_range*10)
                #new_start_y = start_y + cylinder_start_array_y[my_index_source_xy]          # + random.randrange(-rand_range*10, rand_range*10)
                #new_start_z = start_z + cylinder_start_array_z[my_index_source_z]          # + random.randrange(-rand_range*20, rand_range*20)
                batch_counter += 1
                laser_rand_ind = np.random.randint(laser_array_length)
                new_start_x = start_x + laser_array[laser_rand_ind][0]
                new_start_y = start_y + laser_array[laser_rand_ind][1]
                new_start_z = np.random.randint(int(height/100))*100 - int(height/2)
                #print("Batch number = ", batch_counter)
                #print("Photon number = ", i+1)
                used_batch_photons = 0
            used_batch_photons += 1
            x = new_start_x
            y = new_start_y
            z = new_start_z
            #x_dir = random.randrange(-rand_range, rand_range)
            #y_dir = random.randrange(-rand_range, rand_range)
            #z_dir = random.randrange(-rand_range, rand_range)
            #v_dir_coeff = math.sqrt(x_dir**2 + y_dir**2 + z_dir**2) / step
            #direction_vector = np.array([x_dir/v_dir_coeff, y_dir/v_dir_coeff, z_dir/v_dir_coeff])
            random_vect_ind = np.random.randint(len(theta_array_homo))
            direction_vector = random_vector(theta_array_homo, random_vect_ind)
            # This time we set the starting direction as the laser falling on the sample
            #direction_vector = np.array([0.0000000001, 0.0000000001, -0.9999999998])
            #direction_vector = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])
            #direction_vector = rayleigh_scattering_random(0.001, 0, 1)
            inside = 1
            if z >= height/2 or z <= -height/2 or (x**2 + y**2 >= radius**2):
                inside = 0
                continue
            iteration = 0
            while inside == 1 and iteration < 10**5:
                iteration += 1
                #if z >= height/2 or z <= -height/2 or (x**2 + y**2 >= radius**2):
                #    inside = 0
                #    break
                # check_z = np.heaviside(z + height/2, 1) - np.heaviside(z - height/2, 1)
                # check_radius = np.heaviside(radius**2 - (x**2 + y**2), 1)
                # inside = check_radius * check_z
                #heavy = np.heaviside([z + height/2, z - height/2, radius**2 - (x**2 + y**2)], 1)
                #inside = (heavy[0] - heavy[1]) * heavy[2]
                #i_seed = random.randrange(int(1000/chance_of_interaction_per_micron))
                #if i_seed <= chance_of_interaction_per_micron*1000:
                #    direction_vector = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])
                #    #direction_vector = random_vector()
                # Pre-made, home-assembled array for the geometric probability
                #rand_interaction = np.random.randint(len(interaction_prob_arr))
                #n_steps = interaction_prob_arr[rand_interaction]
                n_steps = np.random.geometric(p=chance_of_interaction_per_micron/100)
                virtual_x = x + direction_vector[0] * n_steps
                virtual_y = y + direction_vector[1] * n_steps
                virtual_z = z + direction_vector[2] * n_steps
                theta_dir = np.arccos(direction_vector[2] / np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2 + direction_vector[2] ** 2))
                reflect_iter = 0
                leftover_iter = n_steps
                """____________________________Checking if the photon leaves the aerogel or reflects back__________________"""
                if virtual_z > height/2 or virtual_z < -height/2 or virtual_x**2 + virtual_y**2 > radius**2:
                    a = b = c = a1 = b1 = c1 = a2 = b2 = c2 = 0
                    while inside == 1 and reflect_iter < 100 and leftover_iter > 0:
                        if direction_vector[0]**2 < 0.0000000001 and direction_vector[1]**2 < 0.0000000001:          #  outside
                            inside = 0
                            break
                        reflect_iter += 1
                        # a*x**2 + b*x + c = 0
                        a = direction_vector[0] ** 2 + direction_vector[1] ** 2
                        b = 2 * (direction_vector[0] * x + direction_vector[1] * y)
                        c = x ** 2 + y ** 2 - radius ** 2
                        if (b ** 2 - 4 * a * c) < 0:
                            print("ERROR! In the equation for intersection with collection - sqrt less than zero! (Outside aerogel)")
                            continue
                        if a > 0:
                            t_cylinder = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                        else:
                            print(direction_vector)
                            print("ERROR! (reflection) a = ", a)
                            continue
                        if direction_vector[2] > 0:
                            t_plane = ((height / 2) - z) / direction_vector[2]
                        elif direction_vector[2] < 0:
                            t_plane = - ((height / 2) + z) / direction_vector[2]
                        else:
                            t_plane = 10**9
                        if max([leftover_iter - t_plane, leftover_iter - t_cylinder]) <= 0:     # interaction is inside after all
                            x = x + direction_vector[0] * leftover_iter
                            y = y + direction_vector[1] * leftover_iter
                            z = z + direction_vector[2] * leftover_iter
                            break
                        temp_theta = np.arccos(direction_vector[2] / np.sqrt(direction_vector[0] ** 2 +
                                                                              direction_vector[1] ** 2 + direction_vector[2] ** 2))
                        temp_phi = np.sign(direction_vector[1]) * np.arccos(direction_vector[0] /
                                                                            np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
                        if t_cylinder > t_plane:
                            """_________________Photon collides with a horizontal plane of the cylinder_____________________"""
                            leftover_iter = leftover_iter - t_plane
                            intersection_x = x + direction_vector[0] * t_plane
                            intersection_y = y + direction_vector[1] * t_plane
                            intersection_z = z + direction_vector[2] * t_plane
                            if temp_theta < angle_of_total_reflection:                     #  refracted outside
                                shit_theta = temp_theta
                                temp_theta = np.arcsin(np.sin(temp_theta) * n_aerogel / n_air)
                                x = intersection_x
                                y = intersection_y
                                z = intersection_z
                                shit_direction = direction_vector
                                direction_vector[0] = step * np.sin(temp_theta) * np.cos(temp_phi)
                                direction_vector[1] = step * np.sin(temp_theta) * np.sin(temp_phi)
                                direction_vector[2] = step * np.cos(temp_theta)
                                inside = 0
                                break
                            elif math.pi - temp_theta < angle_of_total_reflection:
                                reverse_theta = math.pi - temp_theta
                                reverse_theta = np.arcsin(np.sin(reverse_theta) * n_aerogel / n_air)
                                temp_theta = math.pi - reverse_theta
                                x = intersection_x
                                y = intersection_y
                                z = intersection_z
                                direction_vector[0] = step * np.sin(temp_theta) * np.cos(temp_phi)
                                direction_vector[1] = step * np.sin(temp_theta) * np.sin(temp_phi)
                                direction_vector[2] = step * np.cos(temp_theta)
                                inside = 0
                                break
                            else:                                                          #  reflected inside
                                direction_vector[2] = - direction_vector[2]
                                x = intersection_x
                                y = intersection_y
                                z = intersection_z
                        elif t_cylinder < t_plane:
                            """___________________Photon collides with the curved part of the cylinder______________"""
                            leftover_iter = leftover_iter - t_cylinder
                            intersection_x = x + direction_vector[0] * t_cylinder
                            intersection_y = y + direction_vector[1] * t_cylinder
                            intersection_z = z + direction_vector[2] * t_cylinder
                            # plane and line equation for angle of intersection:
                            a1, b1, c1 = direction_vector[0], direction_vector[1], direction_vector[2]
                            a2, b2, c2 = intersection_x / radius, intersection_y / radius, 0
                            if a1**2 == a2**2 and b1**2 == b2**2 and c1**2 == c2**2:            #  outside
                                """Check again, something does not add up, but should not trigger often anyway"""
                                inside = 0
                                break
                            angle_intersection_under_sin = (a1*a2 + b1*b2 + c1*c2)/(np.sqrt(a1**2 + b1**2 + c1**2) * np.sqrt(a2**2 + b2**2 + c2**2))
                            if angle_intersection_under_sin > 1 and angle_intersection_under_sin < 1.01:
                                angle_intersection_under_sin = 1
                            angle_intersection = np.arcsin(angle_intersection_under_sin)
                            angle_intersection = (math.pi / 2) - (np.sign(angle_intersection) * angle_intersection)
                            if angle_intersection == 0:
                                inside = 0
                                break
                            plane_theta = np.arccos(c2 / np.sqrt(a2 ** 2 + b2 ** 2 + c2 ** 2))
                            plane_phi = np.sign(b2) * np.arccos(a2 / np.sqrt(a2 ** 2 + b2 ** 2))
                            virtual_rot_z = np.array([direction_vector[0] * np.cos(-plane_phi) - direction_vector[1] * np.sin(-plane_phi),
                                                      direction_vector[0] * np.sin(-plane_phi) + direction_vector[1] * np.cos(-plane_phi),
                                                      direction_vector[2]])
                            virtual_rot_y = np.array([virtual_rot_z[0] * np.cos(-plane_theta) + virtual_rot_z[2] * np.sin(-plane_theta),
                                                      virtual_rot_z[1],
                                                      virtual_rot_z[2] * np.cos(-plane_theta) - virtual_rot_z[0] * np.sin(-plane_theta)])
                            virtual_plane_dir = virtual_rot_y
                            virtual_phi_under_cos = virtual_plane_dir[0] / np.sqrt(virtual_plane_dir[0]**2 + virtual_plane_dir[1]**2)
                            if np.sqrt(virtual_plane_dir[0]**2 + virtual_plane_dir[1]**2) == 0:
                                inside = 0
                                break
                            if virtual_phi_under_cos > 1 and virtual_phi_under_cos < 1.01:
                                virtual_phi_under_cos = 1
                            virtual_phi = np.sign(virtual_plane_dir[1]) * np.arccos(virtual_phi_under_cos)
                            if angle_intersection < angle_of_total_reflection:                  #  refracted outside
                                refracted_theta = np.arcsin(np.sin(angle_intersection) * n_aerogel / n_air)
                                refracted_phi = virtual_phi                          # could be anything since it doesn't change
                                refr_x = step * np.sin(refracted_theta) * np.cos(refracted_phi)
                                refr_y = step * np.sin(refracted_theta) * np.sin(refracted_phi)
                                refr_z = step * np.cos(refracted_theta)
                                rot_y = np.array([refr_x * np.cos(plane_theta) + refr_z * np.sin(plane_theta), refr_y,
                                                  refr_z * np.cos(plane_theta) - refr_x * np.sin(plane_theta)])
                                rot_z = np.array([rot_y[0] * np.cos(plane_phi) - rot_y[1] * np.sin(plane_phi), rot_y[0] *
                                                  np.sin(plane_phi) + rot_y[1] * np.cos(plane_phi), rot_y[2]])
                                direction_vector = rot_z
                                x = intersection_x
                                y = intersection_y
                                z = intersection_z
                                inside = 0
                                break
                            else:                                                               #  reflected inside
                                reflected_theta = math.pi - angle_intersection
                                reflected_phi = virtual_phi                          # could be anything since it doesn't change
                                refl_x = step * np.sin(reflected_theta) * np.cos(reflected_phi)
                                refl_y = step * np.sin(reflected_theta) * np.sin(reflected_phi)
                                refl_z = step * np.cos(reflected_theta)
                                rot_y = np.array([refl_x * np.cos(plane_theta) + refl_z * np.sin(plane_theta), refl_y,
                                                  refl_z * np.cos(plane_theta) - refl_x * np.sin(plane_theta)])
                                rot_z = np.array([rot_y[0] * np.cos(plane_phi) - rot_y[1] * np.sin(plane_phi), rot_y[0] *
                                                  np.sin(plane_phi) + rot_y[1] * np.cos(plane_phi), rot_y[2]])
                                direction_vector = rot_z
                                x = intersection_x
                                y = intersection_y
                                z = intersection_z
                        else:
                            inside = 0
                            break
                else:
                    x = virtual_x
                    y = virtual_y
                    z = virtual_z
                #inside = 0
                if inside == 0:
                    break
                direction_vector_new = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])
                if direction_vector_new[0]**2 + direction_vector_new[1]**2 + direction_vector_new[2]**2 == 0:
                    print("ERROR! The direction vector is zero! Retry! (Rayleigh)")
                    iter_dir = 0
                    while direction_vector_new[0]**2 + direction_vector_new[1]**2 + direction_vector_new[2]**2 == 0 and iter_dir <= 10**2:
                        iter_dir += 1
                        direction_vector_new = rayleigh_scattering_random(direction_vector[0], direction_vector[1], direction_vector[2])
                    if iter_dir >= 10**2:
                        print("Something wrong with Rayleigh")
                        break
                direction_vector = direction_vector_new
                # temp_direction_vector = light_path_iteration()
                # direction_vector = direction_vector * np.heaviside(i_seed - chance_of_interaction_per_micron * 1000, 1) + \
                #                    temp_direction_vector * np.heaviside(chance_of_interaction_per_micron * 1000 - i_seed, 0)
                # x = x + direction_vector[0] * n_steps
                # y = y + direction_vector[1] * n_steps
                # z = z + direction_vector[2] * n_steps
            if iteration >= 10 ** 5:
                print("Too many iterations!")
                print("Photon number = ", i)
                continue
            iteration_array[i] = iteration
            #x_t = old_x + direction_vector[0] * t
            #y_t = old_y + direction_vector[1] * t
            #z_t = old_z + direction_vector[2] * t
            #t = (x_t - old_x) / direction_vector[0]
            #x**2 + y**2 = radius**2
            #old_x**2 + old_y**2 + 2*t*(direction_vector[0]* + direction_vector[1]) + t**2 * (direction_vector[0]**2 + direction_vector[1]**2)
            ##############
            """a = direction_vector[0]**2 + direction_vector[1]**2
            b = 2 * (direction_vector[0] * old_x + direction_vector[1] * old_y)
            c = old_x**2 + old_y**2 - radius**2
            if (b**2 - 4*a*c) < 0:
                print("ERROR! In the equation for intersection with collection - sqrt less than zero! (Outside aerogel)")
                continue
            if a > 0:
                t_cylinder = (-(b) + math.sqrt(b**2 - 4*a*c))/(2*a)
            else:
                print("ERROR! The direction vector is zero!")
                continue
            if direction_vector[2] > 0:
                t_plane = ((height / 2) - old_z) / direction_vector[2]
            elif direction_vector[2] < 0:
                t_plane = - ((height / 2) + old_z) / direction_vector[2]
            elif direction_vector[2] == 0:
                t_plane = 10**9
            if t_cylinder > t_plane and not (direction_vector[0] == 0 and direction_vector[1] == 0):
                temp_theta = np.arccos(direction_vector[2] / (np.sqrt(direction_vector[0]**2 + direction_vector[1]**2) + direction_vector[2]**2))
                temp_phi = np.sign(direction_vector[1]) * np.arccos(direction_vector[0] / np.sqrt(direction_vector[0]**2 + direction_vector[1]**2))
                temp_theta = np.arcsin(np.sin(temp_theta) * n_aerogel / n_air)"""
            ###########
            x0 = x
            y0 = y
            z0 = z
            x_last_arr[i] = x
            y_last_arr[i] = y
            z_last_arr[i] = z
            r_loc = math.sqrt(x0**2 + y0**2 + z0**2)
            num_of_zeroes += 1 + (1 * bool(inside))
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
                print("ERROR! In the equation for intersection with collection - sqrt less than zero! (Outside sphere)")
                continue
            if a > 0:
                n = (-(b) + math.sqrt(b**2 - 4*a*c))/(2*a)
            elif a < 0:
                print("ERROR! The direction vector is complex!")
                continue
            else:
                print("ERROR! The direction vector is zero! (Sphere interpolation)")
                continue
            x_arr[i] = x0 + direction_vector[0] * n
            y_arr[i] = y0 + direction_vector[1] * n
            z_arr[i] = z0 + direction_vector[2] * n
            batch_n = int(batch_counter)
            photon_list.append(Photon(i, x_arr[i], y_arr[i], z_arr[i], batch_n, 0))
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
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(x_coord, y_coord, z_coord)
        # #ax.scatter(start_x, start_y, start_z)
        # ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
        # plt.show()
        # plt.close()
        #collected_photons = pd.DataFrame([x_coord, y_coord, z_coord], columns=["x_v", "y_v", "z_v"])
        #print(collected_photons)
        """_______________________________Collecting photons into apperture and rotating__________________________________"""
        print("Number of photons in/on the sphere = ", len(x_coord))
        print("Number of photons inside the sphere = ", len(x_coord) - num_of_zeroes)
        all_coord = np.asarray((x_coord, y_coord, z_coord))
        all_coord = np.transpose(all_coord)
        file_name = "Fin_tallaerogel-poisson_all_photons_TDCR_Center_beam_particles_" + str(int(ph_per_decay_array_core[grand_i])) + \
                  "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-prob-per-micron_Reflection.txt"
        np.savetxt(file_name, all_coord)
        file_name_batches = "Fin_tallaerogel-poisson_batches_TDCR_Center_beam_particles_" + str(int(ph_per_decay_array_core[grand_i])) + \
                            "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-prob-per-micron_Reflection.txt"
        np.savetxt(file_name_batches, temp_batch_array)
        """with open(file_name, "r") as read_file:
            temp_all_coord = np.loadtxt(read_file)
        x_coord, y_coord, z_coord = np.hsplit(temp_all_coord, [1, 2])"""
        counts_zone_phi = np.zeros(num_angles)
        counts_zone_theta = np.zeros(num_angles)
        x_angle = np.arcsin(apperture_radius / collection_sphere_radius)
        d = collection_sphere_radius * np.cos(x_angle)
        # h = (10**8) / (2 * math.pi * R)
        a0 = 1
        b0 = 0
        c0 = 0
        # d = R - h
        print(d)
        print("Number of batches = ", batch_counter)
        angles = np.asarray(range(num_angles)) * 360 / num_angles
        current_decay_number = 0
        count_array = np.zeros(num_angles * num_of_decays)
        list_ph_n_A = []
        list_ph_n_B = []
        list_ph_n_C = []
        for phi_d in range(num_angles):
            print("phi_d = ", phi_d)
            phi_rad = phi_d * 2 * math.pi / num_angles
            a = a0 * math.cos(phi_rad)              # - b0 * math.sin(theta_r)
            b = a0 * math.sin(phi_rad)              # + b0 * math.cos(theta_r)
            #a = a0 * math.cos(phi_rad)
            #c = - a0 * math.sin(phi_rad)
            counts = 0
            ca_counter = 0
            temp_used_batch_photons = 0
            temp_batch_counter = 0
            for k in range(len(x_coord)):
                if temp_used_batch_photons == temp_batch_array[temp_batch_counter]:
                    temp_used_batch_photons = 0
                    temp_batch_counter += 1
                temp_used_batch_photons += 1
                if a * x_coord[k] + b * y_coord[k] >= d:
                    counts += 1
                    batch_array[ii, phi_d, int(temp_batch_counter)] += 1
                """current_decay_number += 1
                if current_decay_number == number_of_photons_per_decay:
                    current_decay_number = 0
                    count_array[ca_counter + int(len(x_coord)*phi_d / number_of_photons_per_decay)] = counts
                    counts = 0
                    ca_counter += 1
            counts = sum(count_array[int(len(x_coord)*phi_d / number_of_photons_per_decay):int(len(x_coord)*(phi_d+1) / number_of_photons_per_decay)])"""
            counts_zone_phi[phi_d] = counts
            r_h[phi_d, ii] = counts
        skip_x = start_x / 1000  # in millimeters
        label_plot0 = f"Center + {skip_x:5.1f} mm"
        ax[0, 0].plot(angles, counts_zone_phi, label=label_plot0)
        ax[0, 0].set_title("Horizontal")
        for theta_d in range(num_angles):
            theta_rad = theta_d * 2 * math.pi / num_angles
            a = a0 * math.cos(theta_rad)              # - b0 * math.sin(theta_r)
            c = - a0 * math.sin(theta_rad)              # + b0 * math.cos(theta_r)
            counts = 0
            for k in range(len(x_coord)):
                if a * x_coord[k] + c * z_coord[k] >= d:
                    counts += 1
            counts_zone_theta[theta_d] = counts
            r_v[theta_d, ii] = counts
        ax[0, 1].plot(angles, counts_zone_theta, label=label_plot0)
        ax[0, 1].set_title("Vertical")
        label_plot1_0 = label_plot0 + " Horizontal"
        label_plot1_1 = label_plot0 + " Vertical"
        ax[1, 0].plot(angles, counts_zone_phi, label=label_plot1_0)
        ax[1, 0].plot(angles, counts_zone_theta, label=label_plot1_1)
        ax[1, 0].set_title("Horizontal and vertical")
        ax[1, 1].plot(angles, counts_zone_theta - counts_zone_phi, label=label_plot0)
        ax[1, 1].set_title("Vertical - horizontal")
        print("--- %s seconds ---" % (time.time() - start_time))
        ax[0, 0].set_xlabel("Angle, degrees")
        ax[0, 0].set_ylabel("Counts")
        ax[0, 0].legend()
        ax[0, 0].grid(True)
        ax[0, 1].set_xlabel("Angle, degrees")
        ax[0, 1].set_ylabel("Counts")
        ax[0, 1].legend()
        ax[0, 1].grid(True)
        ax[1, 0].set_xlabel("Angle, degrees")
        ax[1, 0].set_ylabel("Counts")
        ax[1, 0].legend()
        ax[1, 0].grid(True)
        ax[1, 1].set_xlabel("Angle, degrees")
        ax[1, 1].set_ylabel("Counts")
        ax[1, 1].legend()
        ax[1, 1].grid(True)
    #plt.show()
    plt.close()
    angles = math.pi * angles / 180
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    r1 = r_h[:, 0]
    r2 = r_h[:, 1]
    r3 = r_h[:, 2]
    ax[0].plot(angles, r1, marker="o", linestyle="dashed", label="center")
    ax[0].plot(angles, r2, marker="s", linestyle="dashed", label="4 mm from center")
    ax[0].plot(angles, r3, marker="^", linestyle="dashed", label="8 mm from center")
    ax[0].set_title("Horizontal")
    ax[0].legend()
    save_out = np.asarray((angles, r1, r2, r3))
    save_out = np.transpose(save_out)
    r1h = r1
    file_name_h = "Fin_tallaerogel-poisson_horizontal_collection_TDCR_" + str(int(apperture_radius / 10**3)) + "mm_" + str(int(ph_per_decay_array_core[grand_i])) + \
                  "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-prob-per-micron_Reflection.txt"
    np.savetxt(file_name_h, save_out)
    r1 = r_v[:, 0]
    r2 = r_v[:, 1]
    r3 = r_v[:, 2]
    ax[1].plot(angles, r1, marker="o", linestyle="dashed", label="center")
    ax[1].plot(angles, r2, marker="s", linestyle="dashed", label="4 mm from center")
    ax[1].plot(angles, r3, marker="^", linestyle="dashed", label="8 mm from center")
    ax[1].set_title("Vertical")
    save_out = np.asarray((angles, r1, r2, r3))
    save_out = np.transpose(save_out)
    file_name_v = "Fin_tallaerogel-poisson_vertical_collection_TDCR_" + str(int(apperture_radius / 10**3)) + "mm_" + str(int(ph_per_decay_array_core[grand_i])) + \
                  "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-ppm_Reflection.txt"
    np.savetxt(file_name_v, save_out)
    r1v = r1
    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax[0].grid(True)
    ax[1].grid(True)
    #ax.set_title("Angular distribution of light from New YAG:Ce - uncalcinated, 520nm", va='bottom')
    #plt.show()
    plt.close()
    """_________________________________________Coincidence counter____________________________________________"""
    decay_step = 0.512
    coincidence_window = 40
    coincidence_window_2 = 400
    steps_coincidence = int(coincidence_window_2 / decay_step)
    decay_df = pd.read_excel("Radiolum_decay_512ps_YAG.xlsx")
    decay_prob = np.asarray(decay_df["Probability"])
    #detector_A = r1h[0]
    #detector_B = r1h[int(len(r1h) / 3)]
    #detector_C = r1h[2 * int(len(r1h) / 3)]
    for ik_counter in range(num_steps):
        d_AB = 0
        d_BC = 0
        d_CA = 0
        triple_c = 0
        num_of_missed_batches = 0
        num_of_empty_batches = 0
        #count_array_a, count_array_b, count_array_c = np.split(count_array, 3)
        for k_counter in range(num_of_decays):
            timing_temp_a = 0
            timing_temp_b = 0
            timing_temp_c = 0
            #tot_counts = batch_array[ik_counter, 0, k_counter] + batch_array[ik_counter, 1, k_counter] + batch_array[ik_counter, 2, k_counter]
            tot_counts = sum(batch_array[ik_counter, :, k_counter])
            if tot_counts != 0:
                timing_temp = np.random.choice(int(len(decay_prob)), int(tot_counts), p=decay_prob)
            else:
                num_of_empty_batches += 1
                continue
            timing_temp_a, timing_temp_b, timing_temp_c = np.split(timing_temp, [int(batch_array[ik_counter, 0, k_counter]), int(batch_array[ik_counter, 1, k_counter] + batch_array[ik_counter, 0, k_counter])])
            #___________________________Single detector______________________________
            if batch_array[ik_counter, 0, k_counter] + batch_array[ik_counter, 1, k_counter] == 0 or batch_array[ik_counter, 1, k_counter] + batch_array[ik_counter, 2, k_counter] == 0 or batch_array[ik_counter, 0, k_counter] + batch_array[ik_counter, 2, k_counter] == 0:
                num_of_missed_batches += 1
                continue
            #___________________________Two detectors________________________________
            if batch_array[ik_counter, 0, k_counter] == 0:
                timing_mins = np.array([max(decay_prob)+1, min(timing_temp_b), min(timing_temp_c)])
                timing_supermin = min(timing_mins)
                timing_mins = timing_mins - timing_supermin
                if np.absolute(timing_mins[1] - timing_mins[2]) < coincidence_window_2:
                    d_BC += 1
                continue
            if batch_array[ik_counter, 1, k_counter] == 0:
                timing_mins = np.array([min(timing_temp_a), max(decay_prob)+1, min(timing_temp_c)])
                timing_supermin = min(timing_mins)
                timing_mins = timing_mins - timing_supermin
                if np.absolute(timing_mins[0] - timing_mins[2]) < coincidence_window_2:
                    d_CA += 1
                continue
            if batch_array[ik_counter, 2, k_counter] == 0:
                timing_mins = np.array([min(timing_temp_a), min(timing_temp_b), max(decay_prob)+1])
                timing_supermin = min(timing_mins)
                timing_mins = timing_mins - timing_supermin
                if np.absolute(timing_mins[0] - timing_mins[1]) < coincidence_window_2:
                    d_AB += 1
                continue
            #___________________________Three detectors______________________________
            timing_mins = np.array([min(timing_temp_a), min(timing_temp_b), min(timing_temp_c)])
            timing_supermin = min(timing_mins)
            timing_mins = timing_mins - timing_supermin
            if max(timing_mins) < coincidence_window_2:
                triple_c += 1
            if np.absolute(timing_mins[0] - timing_mins[1]) < coincidence_window_2:
                d_AB += 1
            if np.absolute(timing_mins[1] - timing_mins[2]) < coincidence_window_2:
                d_BC += 1
            if np.absolute(timing_mins[0] - timing_mins[2]) < coincidence_window_2:
                d_CA += 1
            if np.absolute(timing_mins[0] - timing_mins[1]) > coincidence_window_2 and np.absolute(timing_mins[1] - timing_mins[2]) > coincidence_window_2 and np.absolute(timing_mins[0] - timing_mins[2]) > coincidence_window_2:
                num_of_missed_batches += 1
        double_coincidence = d_AB + d_BC + d_CA - 2 * triple_c
        print("D = ", double_coincidence)
        print("AB = ", d_AB)
        print("Number of decays without photons arriving to detectors = ", num_of_empty_batches)
        print("Number of decays with only single detections = ", num_of_missed_batches)
        #out_multi = number_of_photons_per_decay / n_photons
        if double_coincidence > 0:
            fin_out = np.array([double_coincidence / num_of_decays, d_AB / num_of_decays, d_BC / num_of_decays, d_CA / num_of_decays,
                                triple_c / num_of_decays, triple_c / double_coincidence])
            names_out = ["D", "AB", "BC", "CA", "triple_c", "T/D"]
        else:
            fin_out = np.array([double_coincidence / num_of_decays, d_AB / num_of_decays, d_BC / num_of_decays, d_CA / num_of_decays,
                                triple_c / num_of_decays])
            names_out = ["D", "AB", "BC", "CA", "triple_c"]
        index = np.arange(len(fin_out)) + 0.3
        bar_width = 0.4
        out_bar_cont = plt.bar(names_out, fin_out)
        plt.ylabel("Coincidences")
        plt.ylim(0, 1)
        plt.bar_label(out_bar_cont, fmt="{:.2f}")
        #plt.yticks(values * value_increment, ['%d' % val for val in values])
        #plt.xticks(fin_out)
        #plt.title('Final values')
        #plt.show()
        file_name_d = "Fin_tallaerogel-poisson_double-coincidence_TDCR_" + str(int(apperture_radius / 10**3)) + "mm_" + str(int(ph_per_decay_array_core[grand_i])) + \
                      "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-prob-per-micron_Reflection_{ik_counter*4}mm.txt"
        file_name_d_fig = "Fin_tallaerogel-poisson_double-coincidence_TDCR_" + str(int(apperture_radius / 10 ** 3)) + "mm_" + str(int(ph_per_decay_array_core[grand_i])) + \
                          "ph-per-decay_" + f"{chance_of_interaction_per_micron:.3f}percent-prob-per-micron_Reflection_{ik_counter * 4}mm.png"
        plt.savefig(file_name_d_fig, dpi=200)
        plt.close()
        np.savetxt(file_name_d, fin_out)

