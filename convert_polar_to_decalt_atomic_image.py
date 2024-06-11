import numpy as np

polar_image = np.loadtxt("atomic_image_all.csv", delimiter=",")


x_min = -10
x_max = 10
x_step = 0.1
y_min = -10
y_max = 10
y_step = 0.1
r_max = 10

x_list = np.arange(x_min, x_max + 0.000001, x_step)
y_list = np.arange(y_min, y_max + 0.000001, y_step)
Y, X = np.meshgrid(y_list, x_list)
R = np.sqrt(X**2 + Y**2)
print(R.shape)

decalt_atomic_image = np.zeros_like(R)


def get_intensity_from_polar_angle(
    image, r, theta_deg, center_r=100, step_r=0.1, step_theta_deg=1
):
    theta_deg = theta_deg % 360
    image_search = np.zeros((image.shape[0], image.shape[1] + 2))
    image_search[:, :-2] = image
    image_search[:, -2] = image[:, 0]
    image_search[:, -1] = image[:, 1]
    r_index = center_r + int(r / step_r)
    theta_index = int(theta_deg / step_theta_deg)
    r_distance = (r / step_r) - int(r / step_r)
    theta_distance = (theta_deg / step_theta_deg) - int(theta_deg / step_theta_deg)
    image_intensity = 0
    image_intensity += (
        image_search[r_index, theta_index] * (1 - r_distance) * (1 - theta_distance)
    )
    image_intensity += (
        image_search[r_index + 1, theta_index] * (r_distance) * (1 - theta_distance)
    )
    image_intensity += (
        image_search[r_index, theta_index + 1] * (1 - r_distance) * (theta_distance)
    )
    image_intensity += (
        image_search[r_index + 1, theta_index + 1] * (r_distance) * (theta_distance)
    )
    return image_intensity


for i, x in enumerate(x_list):
    for j, y in enumerate(y_list):
        r = np.sqrt(x**2 + y**2)
        theta = np.rad2deg(np.arctan2(y, x))
        if r >= r_max:
            continue
        else:
            decalt_atomic_image[i, j] = get_intensity_from_polar_angle(
                polar_image, r, theta
            )

np.savetxt("atomic_image_2d_xy.csv", decalt_atomic_image.T, delimiter=",")
