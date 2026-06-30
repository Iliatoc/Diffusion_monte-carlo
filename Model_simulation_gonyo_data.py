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
#from scipy.optimize import curve_fit
from scipy.signal import lfilter
import math
import copy
import pandas as pd
#from io import StringIO


mypath = "C:/Users/maipa/Desktop/PhD/"
root = Tk()
root.wait_visibility()
root.filenames = filedialog.askopenfilenames(initialdir=mypath, title="Select files")
root.mainloop()
diffusion_list = [0.0001, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
#diffusion_list = [0.025, 0.03]
fit_pos_x_1_arr = []
fit_pos_x_2_arr = []
hard_one_step_arr = []
hard_two_step_arr = []
for file_i in range(3):
    one_step = np.zeros(len(diffusion_list))
    two_step = np.zeros(len(diffusion_list))
    file_counter = int(len(root.filenames) / 3)
    count_low = file_counter * file_i
    count_high = file_counter * (file_i + 1)
    diffusion_count = 0
    for i in range(count_low, count_high):
        with open(root.filenames[i]) as temp_file:
            temp_data = np.loadtxt(temp_file)
        one_step[diffusion_count] = temp_data[0, 2]/temp_data[0, 1]
        two_step[diffusion_count] = temp_data[0, 3]/temp_data[0, 1]
        diffusion_count += 1
    gonyo_path = "Gonyo/Rotation/YAG-Ce_aerogel/"
    with open(mypath + gonyo_path + "YAG-Ce_uncalcinated_center.txt") as temp_file:
        hard_data_0 = np.loadtxt(temp_file)
    with open(mypath + gonyo_path + "YAG-Ce_uncalcinated_center-4mm.txt") as temp_file:
        hard_data_4 = np.loadtxt(temp_file)
    with open(mypath + gonyo_path + "YAG-Ce_uncalcinated_center-8mm.txt") as temp_file:
        hard_data_8 = np.loadtxt(temp_file)
    hard_one_step = hard_data_4[0, 1] / hard_data_0[0, 1]
    hard_two_step = hard_data_8[0, 1] / hard_data_0[0, 1]
    y_1 = hard_one_step
    y_2 = hard_two_step
    for k in range(len(one_step)):
        if y_1 >= one_step[k]:
            k1 = k
        else:
            k1 = k
            break
    for k in range(len(two_step)):
        if y_2 >= two_step[k]:
            k2 = k
        else:
            k2 = k
            break
    if k1 == 0:
        k1 = 1
    if k2 == 0:
        k2 = 1
    x1_1 = diffusion_list[k1]
    x2_1 = diffusion_list[k1-1]

    x1_2 = diffusion_list[k2]
    x2_2 = diffusion_list[k2-1]

    y1_1 = one_step[k1]
    y2_1 = one_step[k1-1]

    y1_2 = two_step[k2]
    y2_2 = two_step[k2-1]

    delta_x1 = x1_1 - x2_1
    delta_y1 = y1_1 - y2_1
    delta_x2 = x1_2 - x2_2
    delta_y2 = y1_2 - y2_2
    #fit_pos_x_1 = 0.02688
    #fit_pos_x_2 = 0.02538
    fit_pos_x_1 = ((hard_one_step - y2_1) * delta_x1 / delta_y1) + x2_1
    fit_pos_x_2 = ((hard_two_step - y2_2) * delta_x2 / delta_y2) + x2_2
    fit_pos_x_1_arr.append(fit_pos_x_1)
    fit_pos_x_2_arr.append(fit_pos_x_2)
    hard_one_step_arr.append(hard_one_step)
    hard_two_step_arr.append(hard_two_step)
    if file_i == 0:
        plt.plot(diffusion_list, one_step, marker="^", color="tab:blue", label="4mm from center")
        plt.plot(diffusion_list, two_step, marker="o", color="tab:orange", label="8mm from center")
    elif file_i == 2:
        lower_bound_1 = one_step
        lower_bound_2 = two_step
    elif file_i == 1:
        upper_bound_1 = one_step
        upper_bound_2 = two_step
    else:
        print("ERROR! look again")
plt.plot(fit_pos_x_1_arr, hard_one_step_arr, linestyle="dashed", marker="^", color="red")
plt.plot(fit_pos_x_2_arr, hard_two_step_arr, linestyle="dashed", marker="o", color="green")
plt.fill_between(diffusion_list, upper_bound_1, lower_bound_1, alpha=0.1, color="tab:blue")
plt.fill_between(diffusion_list, upper_bound_2, lower_bound_2, alpha=0.1, color="tab:orange")
plt.xlim(0, 0.05)
plt.ylim(0, 5)
plt.xlabel("p, % probability of scattering per micrometer")
plt.ylabel("I_R, relative intensity")
plt.grid()
plt.legend()
plt.savefig("Diffusion_fit_top-all_laser-w-4mm.png", dpi=300)
plt.show()
plt.close()
