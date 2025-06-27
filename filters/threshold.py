def compute_dynamic_threshold(min_val, avg_val, max_val, median):
    if avg_val == min_val:
        return min_val  # avoid division by zero; fallback threshold

    # Normalize the position of median between min and avg
    norm_median_pos = (median - min_val) / (avg_val - min_val)

    # Invert it: closer to min = higher weight to avg, closer to avg = higher weight to min
    weight_min = 1 - norm_median_pos
    weight_avg = norm_median_pos

    # Optional: normalize weights (just in case, though they should sum to 1)
    total_weight = weight_min + weight_avg
    weight_min /= total_weight
    weight_avg /= total_weight

    # Dynamic threshold
    threshold = (weight_min * min_val) + (weight_avg * avg_val)

    return threshold
