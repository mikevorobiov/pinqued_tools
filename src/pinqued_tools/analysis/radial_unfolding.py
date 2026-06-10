import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict
from scipy.integrate import simpson

def get_radial_profile_from_pivot(
    data_2d: np.ndarray,
    x_coords_mm: np.ndarray,
    y_coords_mm: np.ndarray,
    pivot_x_mm: float,
    pivot_y_mm: float,
    n_bins: int = 50,
    r_max_mm: Optional[float] = None,
    data_err_2d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts a radial profile from a 2D data array based on an external pivot point.
    Incorporates independent pixel uncertainties and estimates structural unfolding error.

    Args:
        data_2d (np.ndarray): The 2D data array (e.g., intensity map).
        x_coords_mm (np.ndarray): 1D array of physical coordinates (mm) for the columns (X-axis).
        y_coords_mm (np.ndarray): 1D array of physical coordinates (mm) for the rows (Y-axis).
        pivot_x_mm (float): X-coordinate (mm) of the center point for the radial profile.
        pivot_y_mm (float): Y-coordinate (mm) of the center point for the radial profile.
        n_bins (int): Number of radial bins to use for averaging the profile.
        r_max_mm (Optional[float]): Maximum radius (mm) for the bins. If None, it uses 
                                    the maximum distance found in the data.
        data_err_2d (Optional[np.ndarray]): 2D array of independent uncertainties 
                                            corresponding to data_2d.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: 
            - Radial distance centers (mm).
            - Mean signal intensity at each radial distance.
            - Dictionary containing uncertainty arrays:
                'propagated_err': The standard error propagated from data_err_2d.
                'unfolding_err': The standard error of the mean due to data spread within each bin.
                'total_err': The combined uncertainty added in quadrature.
    """
    
    # 1. Create 2D coordinate grids for every pixel center
    # Note: meshgrid's output order matters. X_grid corresponds to columns, Y_grid to rows.
    # The shape should match data_2d (rows x columns).
    X_grid, Y_grid = np.meshgrid(x_coords_mm, y_coords_mm)

    # 2. Calculate the Radial Distance (R) from the external pivot point
    # R is a 2D array with the same shape as data_2d
    R = np.sqrt((X_grid - pivot_x_mm)**2 + (Y_grid - pivot_y_mm)**2)

    # 3. Flatten the data and distance arrays for binning
    r_flat = R.flatten()
    data_flat = data_2d.flatten()

    # Determine bin boundaries
    if r_max_mm is None:
        r_max_mm = np.max(r_flat)
        
    # The bins are evenly spaced in radial distance (R)
    bins = np.linspace(0, r_max_mm, n_bins + 1)
    
    # Use numpy.histogram to compute the sum of the signal (data_flat)
    # and the count of points (data_flat multiplied by 0, 1, or 2) in each bin.
    
    # sum_of_signal_in_bins: sum(I) in each bin
    sum_of_signal_in_bins, _ = np.histogram(r_flat, bins=bins, weights=data_flat)
    sum_of_sq_signal_in_bins, _ = np.histogram(r_flat, bins=bins, weights=data_flat**2)
    
    # count_of_points_in_bins: N in each bin
    count_of_points_in_bins, _ = np.histogram(r_flat, bins=bins)

    if data_err_2d is not None:
        data_err_flat = data_err_2d.flatten()
        sum_of_variance_in_bins, _ = np.histogram(r_flat, bins=bins, weights=data_err_flat**2)
    else:
        sum_of_variance_in_bins = np.zeros_like(sum_of_signal_in_bins)

    # 4. Calculate the Mean Profile
    # Prevent division by zero if a bin has no points
    valid_bins = count_of_points_in_bins > 0
    valid_var_bins = count_of_points_in_bins > 1

    mean_profile = np.divide(
        sum_of_signal_in_bins, 
        count_of_points_in_bins, 
        out=np.zeros_like(sum_of_signal_in_bins, dtype=float), 
        where=valid_bins
    )

    # 5. Estimate Uncertainties
    prop_err = np.zeros_like(mean_profile)
    prop_err[valid_bins] = np.sqrt(sum_of_variance_in_bins[valid_bins]) / count_of_points_in_bins[valid_bins]

    unfold_err = np.zeros_like(mean_profile)
    sum_sq = sum_of_sq_signal_in_bins[valid_var_bins]
    sum_sig = sum_of_signal_in_bins[valid_var_bins]
    n_pts = count_of_points_in_bins[valid_var_bins]
    
    # Sample variance in each bin: (sum(x^2) - sum(x)^2 / N) / (N - 1)
    sample_variance = (sum_sq - (sum_sig**2) / n_pts) / (n_pts - 1)
    sample_variance = np.maximum(sample_variance, 0) # Prevent tiny negative values from precision errors
    
    # Standard error of the mean representing uncertainty from structural spread/asymmetry
    unfold_err[valid_var_bins] = np.sqrt(sample_variance / n_pts)

    errors = {
        'propagated_err': prop_err,
        'unfolding_err': unfold_err,
        'total_err': np.sqrt(prop_err**2 + unfold_err**2)
    }

    # Calculate bin centers for the R axis
    r_centers = (bins[:-1] + bins[1:]) / 2
    
    return r_centers, mean_profile, errors