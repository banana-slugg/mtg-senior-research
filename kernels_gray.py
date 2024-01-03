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

# combine into one tensor

kernel_set = np.stack(
    (
        sob_1,
        sob_2,
        pre_1,
        pre_2,
        sch_1,
        sch_2,
        kir_1,
        kir_2,
        kir_3,
        kir_4,
        kir_5,
        kir_6,
        kir_7,
        kir_8,
    ),
    axis=2,
)
