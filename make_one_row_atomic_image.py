import numpy as np
import InputConfig
import TyHologram
import AtomicImageReconstruction

input_config = InputConfig.InputConfig("./", "configuration.json")
input_config.saveJson("configuration_temp.json")

hologram = TyHologram.TyHologram("./holo.csv", 10000, step_phi_deg=1, step_theta_deg=1)
HoloBase = hologram.Hologram.copy()
atomic_image_all = np.zeros((201, 360))
for i in range(360):
    print(i)
    holo_convert = hologram.rotate_hologram_euler_angle(0, i, 0, False)
    np_holo_convert = np.zeros(
        (1, holo_convert.Hologram.shape[0], holo_convert.Hologram.shape[1])
    )
    energys = np.zeros(1)
    energys[0] = 10000
    np_holo_convert[0] = holo_convert.Hologram
    atomic_image = AtomicImageReconstruction.AtomicImageReconstructionReturnDirect(
        "./configuration.json", np_holo_convert, energys
    )
    # print(atomic_image_all.shape)
    atomic_image_all[:, i] = atomic_image[0, 0, :]

print(atomic_image_all)
np.savetxt("atomic_image_all.csv", atomic_image_all, delimiter=",")
