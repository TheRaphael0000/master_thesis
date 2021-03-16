# window_size = 2*(30) + 1
# win = np.ones(window_size) / window_size
# derivative_kernel = np.array([-1, 1])
#
# smoothed = signal.convolve(original, win, mode="valid")
# dx = signal.convolve(smoothed, derivative_kernel, mode="valid")
# dx_smoothed = signal.convolve(dx, win, mode="valid")
