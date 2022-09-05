# map_range = lambda n, in_min, in_max, out_min, out_max: (n - in_min)/(in_max - in_min) * (out_max - out_min) + out_min
def map_range(n, in_min, in_max, out_min, out_max):
    new_value = (n - in_min)/(in_max - in_min) * (out_max - out_min) + out_min
    return float(f'{new_value:.2f}')
