import numpy as np
import os, sys
import ctypes, time, signal

## load the C library to get the projection mask
project_c = np.ctypeslib.load_library('project.so', os.path.dirname(__file__))

image_mask_c = project_c.get_image_mask

array_1d_int    = np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags='C_CONTIGUOUS')
array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
array_2d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')

image_mask_c.argtypes = [array_1d_double, array_1d_double, ctypes.c_int, ctypes.c_int,\
                         array_2d_double, array_2d_double, array_1d_double, ctypes.c_int, ctypes.c_int]
image_mask_c.restype  = array_1d_int

process_c    = project_c.process
process_c.argtypes = [ctypes.c_double, ctypes.c_int, array_1d_double, \
                      array_2d_double, array_2d_double, array_2d_double]

## and the spice furnish function for the library
furnish_c    = project_c.furnish
furnish_c.argtypes = [ctypes.c_char_p]

