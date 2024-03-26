import torch
import numpy as np
import math
import time

step = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_point_on_unit_sphere():
    """
    Generate a random point on a unit sphere using Marsaglia's method.
    Returns:
        Tensor (x, y, z): Coordinates of the random point.
    """
    while True:
        # Generate three random numbers in the range [-1, 1]
        x = torch.rand(1, device=device) * 2 - 1
        y = torch.rand(1, device=device) * 2 - 1
        z = torch.rand(1, device=device) * 2 - 1
        
        # Check if the point lies within the unit sphere
        if x**2 + y**2 + z**2 <= 1:
            break
    
    # Normalize the point to lie on the unit sphere
    length = torch.sqrt(x**2 + y**2 + z**2)
    x /= length
    y /= length
    z /= length

    return torch.cat((x, y, z), dim=0)

def random_directions(n):
    vectors = []
    for _ in range(n):
        vectors.append(random_point_on_unit_sphere())
    return torch.stack(vectors)

def set_start_positions(n, starting_vector):
    return starting_vector.repeat(n, 1)

def data_for_cylinder_along_z(center_x, center_y, loc_radius, height_z):
    loc_z = torch.linspace(-height_z/2, height_z/2, 50, device=device)
    loc_theta = torch.linspace(0, 2*math.pi, 50, device=device)
    theta_grid, z_grid = torch.meshgrid(loc_theta, loc_z)
    x_grid = loc_radius * torch.cos(theta_grid) + center_x
    y_grid = loc_radius * torch.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def light_path_iteration(step):
    phi = torch.rand(1, device=device) * 2 * math.pi
    theta_cos = torch.rand(1, device=device) * 2 - 1
    theta = torch.acos(theta_cos)
    theta = theta * (torch.sin(theta) ** 2)
    loc_x = step * torch.sin(theta) * torch.cos(phi)
    loc_y = step * torch.sin(theta) * torch.sin(phi)
    loc_z = step * torch.cos(theta)
    loc_direction_vector = torch.cat((loc_x, loc_y, loc_z), dim=0)
    return loc_direction_vector

def rayleigh_scattering_random(directions, theta_array_sin_sqr, step):
    loc_theta_0 = torch.acos(directions[:, 2] / torch.linalg.norm(directions, dim=1))
    loc_phi_0 = torch.sign(directions[:, 1]) * torch.acos(directions[:, 0] / torch.sqrt(directions[:, 0]**2 + directions[:, 1]**2))
    loc_phi = torch.rand(len(directions), device=device) * 2 * math.pi
    loc_theta_index = torch.randint(len(theta_array_sin_sqr), size=(len(directions),), device=device)
    loc_theta = theta_array_sin_sqr[loc_theta_index]
    loc_x = step * torch.sin(loc_theta) * torch.cos(loc_phi)
    loc_y = step * torch.sin(loc_theta) * torch.sin(loc_phi)
    loc_z = step * torch.cos(loc_theta)
    loc_rot_y = torch.stack((loc_x * torch.cos(loc_theta_0) + loc_z * torch.sin(loc_theta_0),
                          loc_y,
                          loc_z * torch.cos(loc_theta_0) - loc_x * torch.sin(loc_theta_0)), dim=1)
    loc_rot_z = torch.stack((loc_rot_y[:, 0] * torch.cos(loc_phi_0) - loc_rot_y[:, 1] * torch.sin(loc_phi_0),
                          loc_rot_y[:, 0] * torch.sin(loc_phi_0) + loc_rot_y[:, 1] * torch.cos(loc_phi_0),
                          loc_rot_y[:, 2]), dim=1)
    loc_direction_vector = loc_rot_z
    return loc_direction_vector

def sin_distribution(num_of_sins, random_max):
    # Actually a homogeneous distribution
    sin_array = torch.tensor([0.], device=device)
    for sin_iter in range(1, num_of_sins):
        loc_theta = sin_iter * math.pi / num_of_sins
        sin_value = torch.sin(torch.tensor(loc_theta, device=device))  # Ensure loc_theta is a tensor
        sin_array = torch.cat((sin_array, torch.ones(int(sin_value * random_max), device=device) * loc_theta))
    return sin_array

def sin_squared_distribution(num_of_sins, random_max):
    # sin(theta)**2 distribution in surface
    sin_array = torch.tensor([0.], device=device)
    for sin_iter in range(1, num_of_sins):
        loc_theta = sin_iter * math.pi / num_of_sins
        sin_value = torch.sin(torch.tensor(loc_theta, device=device))  # Ensure loc_theta is a tensor
        sin_array = torch.cat((sin_array, torch.ones(int((sin_value**3) * random_max * 1000), device=device) * loc_theta))
    return sin_array

def inside_cylinder(positions, h, r):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    return (x**2 + y**2 <= r**2) & (-h/2 <= z) & (z <= h/2)

def photon_interactions(photon_num, chance_per_micron = 0.5):
    i_seed = torch.rand(photon_num, 1, device=device)
    return i_seed <= chance_per_micron

def iter_step(mask, inter_mask, positions, directions):
    return torch.where(mask.unsqueeze(1), positions + directions, positions)

def main():
    print("hello")
    theta_array_homo = sin_distribution(10 ** 2, 10 ** 3)
    theta_array_sin_sqr = sin_squared_distribution(10 ** 2, 10 ** 3)

    height, radius, collection_sphere_radius = 1*10**4, 1.5*10**4, 8*10**4     # microns, micrometers everywhere
    chance_of_interaction_per_micron = 0.05   # percent
    R = collection_sphere_radius
    n_photons = 5000
    photon_directions = random_directions(n_photons)
    photon_locs = set_start_positions(n_photons, torch.tensor([10**4, 0, 0], device=device))

    start_time = time.time()

    max_iterations = 10**6

    progress = 0
    percent_time = time.time()
    for i in range(max_iterations):
        interaction_mask = photon_interactions(n_photons)
        cylinder_mask = inside_cylinder(photon_locs, height, radius)

        #if torch.all(inside_cylinder) == False:
        #    break
        photon_directions = torch.where(interaction_mask & cylinder_mask.unsqueeze(1), 
                                 rayleigh_scattering_random(photon_directions, theta_array_sin_sqr, step), 
                                 photon_directions)
        photon_locs = torch.where(cylinder_mask.unsqueeze(1), photon_locs + photon_directions, photon_locs)
        if int((i/max_iterations)*100) > progress:
            print(f"ETA: --- {(100 - progress) * (time.time() - percent_time)} seconds ---")
            percent_time = time.time()
            progress = int((i/max_iterations)*100)
            print(f"progress: {progress}%")

    r_locs = torch.linalg.norm(photon_locs, dim=1)
    a = torch.sum(photon_directions**2, dim=1)
    b = 2 * torch.sum(photon_locs * photon_directions, dim=1)
    c = r_locs**2 - R**2
    n = ((-(b) + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)).unsqueeze(1)
    results_arr = photon_locs + photon_directions * n

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()