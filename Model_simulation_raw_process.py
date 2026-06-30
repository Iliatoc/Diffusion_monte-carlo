from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import itertools
#from os import listdir
#from os.path import isfile, join
from matplotlib import pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import axes3d
#from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.signal import lfilter
import math
import copy
import pandas as pd
#from io import StringIO


def get_vert_horiz(loc_coord, loc_num_angles, loc_number_of_photons_per_decay, batch_num):
    all_coord = loc_coord
    x_coord = all_coord[:, 0]
    y_coord = all_coord[:, 1]
    z_coord = all_coord[:, 2]
    n_photons = len(x_coord)
    #loc_num_decays = int(n_photons / loc_number_of_photons_per_decay)
    #loc_num_angles = 36
    counts_zone_phi = np.zeros(loc_num_angles)
    counts_zone_theta = np.zeros(loc_num_angles)
    x_angle = np.arcsin(apperture_radius / collection_sphere_radius)
    d = collection_sphere_radius * np.cos(x_angle)
    # h = (10**8) / (2 * math.pi * R)
    a0 = 1
    b0 = 0
    c0 = 0
    # d = R - h
    print(d)
    #loc_number_of_photons_per_decay = 500
    num_of_decays = int(len(batch_num))
    angles = np.asarray(range(loc_num_angles)) * 360 / loc_num_angles
    current_decay_number = 0
    count_array = np.zeros(loc_num_angles * num_of_decays)
    list_ph_n_A = []
    list_ph_n_B = []
    list_ph_n_C = []
    r_h = np.zeros(loc_num_angles)
    r_v = np.zeros(loc_num_angles)
    batch_array = np.zeros((loc_num_angles, num_of_decays+1))
    for phi_d in range(loc_num_angles):
        print("phi_d = ", phi_d)
        phi_rad = phi_d * 2 * math.pi / loc_num_angles
        a = a0 * math.cos(phi_rad)  # - b0 * math.sin(theta_r)
        b = a0 * math.sin(phi_rad)  # + b0 * math.cos(theta_r)
        # a = a0 * math.cos(phi_rad)
        # c = - a0 * math.sin(phi_rad)
        counts = 0
        ca_counter = 0
        temp_used_batch_photons = 0
        temp_batch_counter = 0
        for k in range(len(x_coord)):
            if temp_used_batch_photons == batch_num[temp_batch_counter]:
                temp_used_batch_photons = 0
                temp_batch_counter += 1
            temp_used_batch_photons += 1
            if a * x_coord[k] + b * y_coord[k] >= d:
                counts += 1
                batch_array[phi_d, int(temp_batch_counter)] += 1
        counts_zone_phi[phi_d] = counts
        r_h[phi_d] = counts
    """for k in range(len(x_coord)):
        if y_coord[k]/x_coord[k] <= 2 and y_coord[k]/x_coord[k] > -2 and x_coord[k] > 0:
            batch_array[0, int(k / loc_number_of_photons_per_decay)] += 1
        if y_coord[k]/x_coord[k] > 2 and x_coord[k] > 0:
            batch_array[1, int(k / loc_number_of_photons_per_decay)] += 1
        if x_coord[k] <= 0 and y_coord[k] >= 0:
            batch_array[1, int(k / loc_number_of_photons_per_decay)] += 1
        if y_coord[k]/x_coord[k] <= -2 and x_coord[k] > 0:
            batch_array[2, int(k / loc_number_of_photons_per_decay)] += 1
        if x_coord[k] <= 0 and y_coord[k] < 0:
            batch_array[2, int(k / loc_number_of_photons_per_decay)] += 1"""
    for theta_d in range(loc_num_angles):
        theta_rad = theta_d * 2 * math.pi / loc_num_angles
        a = a0 * math.cos(theta_rad)  # - b0 * math.sin(theta_r)
        c = - a0 * math.sin(theta_rad)  # + b0 * math.cos(theta_r)
        counts = 0
        for k in range(len(x_coord)):
            if a * x_coord[k] + c * z_coord[k] >= d:
                counts += 1
        counts_zone_theta[theta_d] = counts
        r_v[theta_d] = counts
    return r_h, r_v, batch_array, n_photons


def coincidence_counter(batch_array, loc_photons_per_decay, loc_n_photons):
    """_________________________________________Coincidence counter____________________________________________"""
    decay_step = 0.512
    coincidence_window = 400
    coincidence_window_2 = 400
    steps_coincidence = int(coincidence_window / decay_step)
    decay_df = pd.read_excel("C:/Users/maipa/Desktop/PhD/Radiolum_decay_512ps_YAG.xlsx")
    decay_prob = np.asarray(decay_df["Probability"])
    loc_num_decays = len(batch_array[0, :])
    # detector_A = r1h[0]
    # detector_B = r1h[int(len(r1h) / 3)]
    # detector_C = r1h[2 * int(len(r1h) / 3)]
    d_AB = 0
    d_BC = 0
    d_CA = 0
    triple_c = 0
    num_of_missed_batches = 0
    num_of_empty_batches = 0
    # count_array_a, count_array_b, count_array_c = np.split(count_array, 3)
    loc_n_phe = 0
    for k_counter in range(loc_num_decays):
        timing_temp_a = 0
        timing_temp_b = 0
        timing_temp_c = 0
        # tot_counts = batch_array[ik_counter, 0, k_counter] + batch_array[ik_counter, 1, k_counter] + batch_array[ik_counter, 2, k_counter]
        tot_counts = sum(batch_array[:, k_counter])
        loc_n_phe += tot_counts
        if tot_counts != 0:
            timing_temp = np.random.choice(int(len(decay_prob)), int(tot_counts), p=decay_prob)
        else:
            num_of_empty_batches += 1
            continue
        timing_temp_a, timing_temp_b, timing_temp_c = np.split(timing_temp,
                                                               [int(batch_array[0, k_counter]), int(
                                                                   batch_array[1, k_counter] +
                                                                   batch_array[0, k_counter])])
        # ___________________________Single detector______________________________
        if batch_array[0, k_counter] + batch_array[1, k_counter] == 0 or batch_array[1, k_counter] + batch_array[2, k_counter] == 0 or batch_array[0, k_counter] + batch_array[2, k_counter] == 0:
            num_of_missed_batches += 1
            continue
        # ___________________________Two detectors________________________________
        if batch_array[0, k_counter] == 0:
            timing_mins = np.array([max(decay_prob) + 1, min(timing_temp_b), min(timing_temp_c)])
            timing_supermin = min(timing_mins)
            timing_mins = timing_mins - timing_supermin
            if np.absolute(timing_mins[1] - timing_mins[2]) < coincidence_window:
                d_BC += 1
            continue
        if batch_array[1, k_counter] == 0:
            timing_mins = np.array([min(timing_temp_a), max(decay_prob) + 1, min(timing_temp_c)])
            timing_supermin = min(timing_mins)
            timing_mins = timing_mins - timing_supermin
            if np.absolute(timing_mins[0] - timing_mins[2]) < coincidence_window:
                d_CA += 1
            continue
        if batch_array[2, k_counter] == 0:
            timing_mins = np.array([min(timing_temp_a), min(timing_temp_b), max(decay_prob) + 1])
            timing_supermin = min(timing_mins)
            timing_mins = timing_mins - timing_supermin
            if np.absolute(timing_mins[0] - timing_mins[1]) < coincidence_window:
                d_AB += 1
            continue
        # ___________________________Three detectors______________________________
        timing_mins = np.array([min(timing_temp_a), min(timing_temp_b), min(timing_temp_c)])
        timing_supermin = min(timing_mins)
        timing_mins = timing_mins - timing_supermin
        if max(timing_mins) < coincidence_window:
            triple_c += 1
        if np.absolute(timing_mins[0] - timing_mins[1]) < coincidence_window:
            d_AB += 1
        if np.absolute(timing_mins[1] - timing_mins[2]) < coincidence_window:
            d_BC += 1
        if np.absolute(timing_mins[0] - timing_mins[2]) < coincidence_window:
            d_CA += 1
        if np.absolute(timing_mins[0] - timing_mins[1]) > coincidence_window and np.absolute(
                timing_mins[1] - timing_mins[2]) > coincidence_window and np.absolute(
                timing_mins[0] - timing_mins[2]) > coincidence_window:
            num_of_missed_batches += 1
    double_coincidence = d_AB + d_BC + d_CA - 2 * triple_c
    print("D = ", double_coincidence)
    print("AB = ", d_AB)
    print("T = ", triple_c)
    print("Number of decays without photons arriving to detectors = ", num_of_empty_batches)
    print("Number of decays with only single detections = ", num_of_missed_batches)
    # out_multi = number_of_photons_per_decay / n_photons
    if double_coincidence > 0:
        fin_out = np.array(
            [double_coincidence / loc_num_decays, d_AB / loc_num_decays, d_BC / loc_num_decays, d_CA / loc_num_decays,
             triple_c / loc_num_decays, triple_c / double_coincidence])
        names_out = ["D", "AB", "BC", "CA", "triple_c", "T/D"]
    else:
        fin_out = np.array(
            [double_coincidence / loc_num_decays, d_AB / loc_num_decays, d_BC / loc_num_decays, d_CA / loc_num_decays,
             triple_c / loc_num_decays])
        names_out = ["D", "AB", "BC", "CA", "triple_c"]
    loc_n_phe = loc_n_phe / loc_num_decays
    return fin_out, loc_n_phe


def line_eq_zero(loc_x, loc_k):
    return loc_x * loc_k


def find_prob(loc_loc_points, loc_ref):
    loc_x_point = np.zeros(len(loc_loc_points))
    for loc_i in range(len(loc_loc_points)):
        loc_closest_ind = np.argmin(np.abs(loc_ref - loc_loc_points[loc_i]))
        loc_closest_dist = loc_ref[loc_closest_ind] - loc_loc_points[loc_i]
        if loc_closest_dist <= 0:
            loc_range = loc_ref[loc_closest_ind+1] - loc_ref[loc_closest_ind]
            loc_far_dist = loc_range + loc_closest_dist
            loc_x_point[loc_i] = - loc_ref[loc_closest_ind+1]*loc_closest_dist/loc_range + loc_ref[loc_closest_ind]*loc_far_dist/loc_range
        else:
            loc_range = loc_ref[loc_closest_ind] - loc_ref[loc_closest_ind + 1]
            loc_far_dist = loc_range - loc_closest_dist
            loc_x_point[loc_i] = loc_ref[loc_closest_ind + 1] * loc_closest_dist / loc_range + loc_ref[
                loc_closest_ind] * loc_far_dist / loc_range
    return loc_x_point


mypath = "C:/Users/maipa/Desktop/PhD/"
reference_path_Kr = "C:/Users/maipa/Desktop/PhD/Papers/Thesis/New_graphs/source_data/Kr-85_aerogel_5cm_abs_eff-dens_fromPENELOPE.txt"
reference_path_H3 = "C:/Users/maipa/Desktop/PhD/Papers/Thesis/New_graphs/source_data/H-3_aerogel_5cm_abs_eff-density_fromPENELOPE.txt"
root = Tk()
root.wait_visibility()
root.filenames = filedialog.askopenfilenames(initialdir=mypath, title="Select files")
root.mainloop()
collection_sphere_radius = 6 * 10 ** 4
apperture_radius = 3 * 10 ** 4
diffusion_list = [0.0001, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
#diffusion_list = [0.025, 0.03]
photons_list = [3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 40, 50, 60, 100, 150]
#photon_array_fin = np.arange(10, 260, 10)
photon_array_fin = np.array([3, 4, 5, 6])
fit_pos_x_1_arr = []
fit_pos_x_2_arr = []
hard_one_step_arr = []
hard_two_step_arr = []
num_angles = 3
#number_of_photons_per_decay = 500
plot_angles = np.arange(0, num_angles*10, 10) * 2 * 3.14 / 360
#fig, ax = plt.subplots(2)
double_coin = []
ab_coin = []
bc_coin = []
ac_coin = []
triple_coin = []
double_coin_c = []
ab_coin_c = []
bc_coin_c = []
ac_coin_c = []
triple_coin_c = []
n_phe = []
n_phe_c = []
for file_i in range(len(photon_array_fin)):
    with open(root.filenames[file_i+int(len(photons_list)*2)]) as temp_file:
        temp_data = np.loadtxt(temp_file)
    with open(root.filenames[file_i+int(len(photons_list)*2)+int(len(photon_array_fin))]) as batch_file:
        temp_batch_data = np.loadtxt(batch_file)
    temp_horiz, temp_vert, temp_batch, temp_photon_n = get_vert_horiz(temp_data, num_angles, photon_array_fin[file_i], temp_batch_data)
    coincidences, temp_n_phe = coincidence_counter(temp_batch, photon_array_fin[file_i], temp_photon_n)
    double_coin.append(coincidences[0])
    ab_coin.append(coincidences[1])
    bc_coin.append(coincidences[2])
    ac_coin.append(coincidences[3])
    triple_coin.append(coincidences[4])
    n_phe.append(temp_n_phe)
    print("Counts at detectors(r_h): ", temp_horiz)
    print("Counts at detector A (batch): ", sum(temp_batch[0, :]))
    print("Counts at detector B (batch): ", sum(temp_batch[1, :]))
    print("Counts at detector C (batch): ", sum(temp_batch[2, :]))
    #ax[0].scatter(photons_list[file_i], coincidences[0], marker="^", color="royalblue")
    #ax[1].scatter(photons_list[file_i], coincidences[4], marker="^", color="gold")
for file_i in range(int(len(photons_list))):
    with open(root.filenames[file_i]) as temp_file:
        temp_data = np.loadtxt(temp_file)
    with open(root.filenames[file_i+int(len(photons_list))]) as batch_file:
        temp_batch_data = np.loadtxt(batch_file)
    temp_horiz, temp_vert, temp_batch, temp_photon_n = get_vert_horiz(temp_data, num_angles, photons_list[file_i], temp_batch_data)
    coincidences, temp_n_phe = coincidence_counter(temp_batch, photons_list[file_i], temp_photon_n)
    double_coin_c.append(coincidences[0])
    ab_coin_c.append(coincidences[1])
    bc_coin_c.append(coincidences[2])
    ac_coin_c.append(coincidences[3])
    triple_coin_c.append(coincidences[4])
    n_phe_c.append(temp_n_phe)
    print("Counts at detectors(r_h): ", temp_horiz)
    print("Counts at detector A (batch): ", sum(temp_batch[0, :]))
    print("Counts at detector B (batch): ", sum(temp_batch[1, :]))
    print("Counts at detector C (batch): ", sum(temp_batch[2, :]))
    #ax[0].scatter(photons_list[file_i], coincidences[0], marker="^", color="royalblue")
    #ax[1].scatter(photons_list[file_i], coincidences[4], marker="^", color="gold")
#plt.fill_between(diffusion_list, upper_bound_1, lower_bound_1, alpha=0.1, color="tab:blue")
#plt.fill_between(diffusion_list, upper_bound_2, lower_bound_2, alpha=0.1, color="tab:orange")
#plt.xlim(0, 0.05)
#plt.ylim(0, 5)
plt.plot(photon_array_fin, double_coin, marker="^", label="Double coincidences", color="blue")
plt.plot(photon_array_fin, triple_coin, marker="^", label="Triple coincidences", color="orange")
plt.plot(photons_list, double_coin_c, alpha=0.3, marker="^", label="Double coincidences", color="blue")
plt.plot(photons_list, triple_coin_c, alpha=0.3, marker="^", label="Triple coincidences", color="orange")
#plt.xlabel("p, % probability of scattering per micrometer")
plt.xlabel("Number of generated photons per decay")
plt.ylabel("Number of counts")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
#ax[0].set_xlabel("p, % probability of scattering per micrometer")
#ax[0].set_ylabel("Double coincidences")
#ax[0].grid()
#ax[1].set_xlabel("p, % probability of scattering per micrometer")
#ax[1].set_ylabel("Triple coincidences")
#ax[1].grid()
#ax[0].legend()
#plt.savefig("Diffusion_fit_top-all_laser-w-4mm.png", dpi=300)
plt.show()
plt.close()
plt.plot(photon_array_fin, double_coin, marker="^", label="Double coincidences, TDCR", color="blue")
plt.plot(photon_array_fin, triple_coin, marker="^", label="Triple coincidences, TDCR", color="orange")
plt.plot(photons_list, double_coin_c, alpha=0.3, marker="^", label="Double coincidences, C-TDCR", color="blue")
plt.plot(photons_list, triple_coin_c, alpha=0.3, marker="^", label="Triple coincidences, C-TDCR", color="orange")
#plt.xlabel("p, % probability of scattering per micrometer")
plt.xlabel("Number of generated photons per decay")
plt.ylabel("Number of counts")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
plt.savefig("Diffusion_trend_brightness_compare_400ns.png", dpi=200)
plt.show()
plt.close()
t_over_d = np.asarray(triple_coin) / np.asarray(double_coin)
t_over_d_c = np.asarray(triple_coin_c) / np.asarray(double_coin_c)
photons_array = np.asarray(photons_list)
plt.plot(photon_array_fin, t_over_d, marker="^", label="T/D, TDCR", color="blue")
plt.plot(photons_array, t_over_d_c, marker="^", label="T/D, C-TDCR", color="orange")
plt.xlabel("Number of generated photons per decay")
plt.ylabel("Fraction in number of counts")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
plt.savefig("Diffusion_trend_ToverD_compare_400ns.png", dpi=200)
plt.show()
plt.close()
#n_phe_calc = - 3 * np.log(1 - (np.asarray(triple_coin)/np.asarray(ab_coin))) - 3 * np.log(1 - (np.asarray(triple_coin)/np.asarray(bc_coin))) \
#             - 3 * np.log(1 - (np.asarray(triple_coin)/np.asarray(ac_coin)))
n_phe_calc_c = - 3 * np.log(1 - (np.asarray(triple_coin_c)/np.asarray(ab_coin_c))) - 3 * np.log(1 - (np.asarray(triple_coin_c)/np.asarray(bc_coin_c))) \
               - 3 * np.log(1 - (np.asarray(triple_coin_c)/np.asarray(ac_coin_c)))
trendline_param, trash = curve_fit(line_eq_zero, photons_array[:len(photons_array)], n_phe_calc_c[:len(photons_array)])
trendline_points = line_eq_zero(photons_array[:len(photons_array)], *trendline_param)
trend_label = f"Trendline={trendline_param}"
#plt.plot(photon_array_fin, n_phe_calc, marker="^", label="n_phe, TDCR", color="blue")
plt.plot(photons_array, n_phe_calc_c, marker="^", label="n_phe, C-TDCR", color="orange")
plt.plot(photons_array[:len(photons_array)], trendline_points, label=trend_label)
plt.xlabel("Number of generated photons per decay")
plt.ylabel("Number of photons arriving at PMTs based on T/D calc. (n_phe)")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
plt.savefig("n_phe_basedon_ToverD_400ns.png", dpi=200)
plt.show()
plt.close()
coef_tdcr = np.average(n_phe / photon_array_fin)
coef_c_tdcr = np.average(n_phe_c / photons_array)
label_tdcr = f"TDCR photons, {coef_tdcr:f} n_phe per generated photon"
label_c_tdcr = f"C-TDCR photons, {coef_c_tdcr:f} n_phe per generated photon"
plt.plot(photon_array_fin, n_phe, marker="^", label=label_tdcr, color="blue")
plt.plot(photons_array, n_phe_c, marker="^", label=label_c_tdcr, color="orange")
plt.xlabel("Number of generated photons per decay")
plt.ylabel("Number of photons arriving at PMTs per decay")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
plt.savefig("Linear_nphe_tdcr-vs-ctdcr.png", dpi=200)
plt.show()
plt.close()
"""______________________________________________Fin calcs (interp)_______________________________________"""
with open(reference_path_H3) as temp_ref_file:
    ref_data = np.loadtxt(temp_ref_file)
n_phe_temp = photon_array_fin * trendline_param
energy_temp = n_phe_temp * 1000 / 0.1204
energy_ref = ref_data[:, 0]
prob_dens_ref = ref_data[:, 1]
prob_ref = np.zeros(len(energy_ref))
for ii in range(1, len(energy_ref)):
    prob_ref[ii] = (energy_ref[ii] - energy_ref[ii-1])*ref_data[ii, 1]
prob_ref[0] = energy_ref[0] * ref_data[0, 1]
prob_dens_interp = np.interp(energy_temp, energy_ref, prob_dens_ref)
prob_fin = np.zeros(len(energy_temp))
for ii in range(1, len(energy_temp)):
    prob_fin[ii] = (energy_temp[ii] - energy_temp[ii-1]) * prob_dens_interp[ii]
prob_fin[0] = energy_temp[0] * prob_dens_interp[0]
"""______________________________________________Fin calcs (detection and plot)_______________________________________"""
double_eff = np.sum(double_coin * prob_fin)
triple_tot = np.sum(triple_coin * prob_fin)
t_over_d_fin = triple_tot/double_eff
deposition_eff_label = f"Modelled points for calculations. \n Double coincidence eff={double_eff:.3f}, T/D={t_over_d_fin:.3f}"
plt.plot(energy_temp/1000, prob_fin, marker="^", label=deposition_eff_label, color="blue")
#plt.plot(energy_ref/1000, prob_ref, label="Deposited energy, PENELOPE", color="orange")
plt.xlabel("Energy of deposition, keV")
plt.ylabel("Probability")
plt.grid()
plt.legend()
#plt.rc('axes', labelsize=12)
plt.savefig("Fin_diffusion_efficiency_calc.png", dpi=200)
plt.show()
plt.close()

