import numpy as np

# Sobel
sob_1 = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]), dtype=float)

sob_2 = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]), dtype=float)

# Prewitt
pre_1 = np.array(([1, 0, -1], [1, 0, -1], [1, 0, -1]), dtype=float)
pre_2 = np.array(([1, 1, 1], [0, 0, 0], [-1, -1, -1]), dtype=float)

# Scharr
sch_1 = np.array(([3, 0, -3], [10, 0, -10], [3, 0, -3]), dtype=float)
sch_2 = np.array(([3, 10, 3], [0, 0, 0], [-3, -10, -3]), dtype=float)

# Kirsch
kir_1 = np.array(([5, 5, 5], [-3, 0, -3], [-3, -3, -3]), dtype=float)
kir_2 = np.array(([5, 5, -3], [5, 0, -3], [-3, -3, -3]), dtype=float)
kir_3 = np.array(([5, -3, -3], [5, 0, -3], [5, -3, -3]), dtype=float)
kir_4 = np.array(([-3, -3, -3], [5, 0, -3], [5, 5, -3]), dtype=float)
kir_5 = np.array(([-3, -3, -3], [-3, 0, -3], [5, 5, 5]), dtype=float)
kir_6 = np.array(([-3, -3, 5], [-3, 0, 5], [-3, 5, 5]), dtype=float)
kir_7 = np.array(([-3, -3, 5], [-3, 0, 5], [-3, -3, 5]), dtype=float)
kir_8 = np.array(([-3, 5, 5], [-3, 0, 5], [-3, -3, -3]), dtype=float)

# Turn them in to 3 x 3 x 3
sobels_1 = np.stack((sob_1, sob_1, sob_1))
sobels_2 = np.stack((sob_2, sob_2, sob_2))

prewitts_1 = np.stack((pre_1, pre_1, pre_1))
prewitts_2 = np.stack((pre_2, pre_2, pre_2))

scharr_1 = np.stack((sch_1, sch_1, sch_1))
scharr_2 = np.stack((sch_2, sch_2, sch_2))

kirsh_1 = np.stack((kir_1, kir_1, kir_1))
kirsh_2 = np.stack((kir_2, kir_2, kir_2))
kirsh_3 = np.stack((kir_3, kir_3, kir_3))
kirsh_4 = np.stack((kir_4, kir_4, kir_4))
kirsh_5 = np.stack((kir_5, kir_5, kir_5))
kirsh_6 = np.stack((kir_6, kir_6, kir_6))
kirsh_7 = np.stack((kir_7, kir_7, kir_7))
kirsh_8 = np.stack((kir_8, kir_8, kir_8))

# slam them all together into a 3 x 3 x 3 x 6 tensor

kernel_set = np.stack(
    (
        sobels_1,
        sobels_2,
        prewitts_1,
        prewitts_2,
        scharr_1,
        scharr_2,
        kirsh_1,
        kirsh_2,
        kirsh_3,
        kirsh_4,
        kirsh_5,
        kirsh_6,
        kirsh_7,
        kirsh_8,
    ),
    axis=3,
)
