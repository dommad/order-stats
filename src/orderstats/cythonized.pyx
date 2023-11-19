
# Save this code in a file with a .pyx extension, e.g., mymodule.pyx
# Compile using: cythonize -i mymodule.pyx

# mymodule.pyx
import numpy as np
cimport numpy as np

def get_confidence_intervals(np.ndarray[double, ndim=2] data, int idx, double alpha):
    cdef Py_ssize_t n_samples = data.shape[0]

    # Use memoryviews for faster access
    cdef double[:] data_idx_view = data[:, idx]

    # Calculate the mean using NumPy
    cdef double master_mean = np.mean(data_idx_view)

    # Convert data_idx_view to a NumPy array before subtraction
    cdef np.ndarray[double, ndim=1] data_diff = np.array(data_idx_view) - master_mean

    # Calculate differences and sort
    cdef np.ndarray[double, ndim=1] diff = np.sort(data_diff)

    # Calculate confidence intervals
    cdef double ci_l = master_mean - diff[int(n_samples * alpha / 2)]
    cdef double ci_u = master_mean - diff[int(n_samples * (1 - alpha / 2))]

    return master_mean, ci_l, ci_u


from typing import List

def get_mask(np.ndarray[double, ndim=1] df_sorted_p_values, List[List[double]] critical_array):
    cdef Py_ssize_t i, j, len_sorted_p, len_critical
    len_sorted_p = len(df_sorted_p_values)
    len_critical = len(critical_array)

    # Use memoryviews for faster access
    cdef double[:] sorted_p_view = df_sorted_p_values

    # Use a pre-allocated list to store the results
    cdef list result = []

    # Temporary Python list for efficient array creation
    cdef list critical_value_list

    for i in range(len_critical):
        # Use a Python list for critical values
        critical_value_list = critical_array[i]

        # Create a boolean array directly within the loop
        mask = np.zeros(len_sorted_p, dtype=bool)

        # Cythonized loop to generate the mask
        for j in range(len_sorted_p):
            mask[j] = sorted_p_view[j] <= critical_value_list[j]

        result.append(mask)

    return result