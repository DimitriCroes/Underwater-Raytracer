"""
UNDERWATER RAY TRACER

Original MATLAB version by Dimitri Croes
Engineering Physics Student at Eindhoven, The Netherlands
2021

Paper used:
TU Delft, "Underwater propagation Ray acoustics," pp. 65–80.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# ============================================================
# DEFAULT SSP DATA
# ============================================================

# Depths (m) and sound speeds (m/s) from a DW mission.
DEFAULT_Z = np.array([
    0, 102, 204, 306, 408, 537, 667, 796, 926,
    1056, 1185, 1315, 1444, 1574, 1704, 1833,
    1963, 2092, 2222, 2352, 2481, 2611, 2740
]) / 3.281

DEFAULT_C = np.array([
    5082.99, 5084.57, 5086.16, 5087.75, 5089.33,
    5089.05, 5088.88, 5088.84, 5088.91, 5089.09,
    5089.40, 5089.82, 5090.36, 5091.02, 5091.80,
    5092.69, 5093.71, 5094.84, 5096.09, 5097.45,
    5098.94, 5100.54, 5102.26
]) / 3.281


# ============================================================
# SSP FUNCTIONS
# ============================================================

def create_sound_speed_profile(z, c, max_depth=2200):
    """
    Interpolate SSP using PCHIP interpolation.
    """

    zz = np.arange(0, max_depth + 1, 1)

    interp = PchipInterpolator(z, c, extrapolate=True)
    cc = interp(zz)

    return zz, cc


def calculate_gradient(cc, zz):
    """
    Calculate SSP gradient.
    """

    return np.diff(cc) / np.diff(zz)


# ============================================================
# RAY TRACING
# ============================================================

def trace_single_ray(
    theta_start,
    z0,
    xmax,
    dx,
    zz,
    cc,
    g
):
    """
    Trace a single acoustic ray.
    """

    x = [0]
    Z = [z0]
    angle = [theta_start]

    c0 = [cc[int(Z[0])]]
    G = g[int(Z[0])]

    R0 = [-c0[0] / (np.cos(np.radians(angle[0])) * G)]

    while x[-1] <= xmax:

        # ----------------------------------------------------
        # Step in x
        # ----------------------------------------------------

        x_new = x[-1] + dx

        # ----------------------------------------------------
        # New angle
        # ----------------------------------------------------

        angle_new = np.degrees(
            np.arcsin(
                ((x_new - x[-1]) / R0[-1]) +
                np.sin(np.radians(angle[-1]))
            )
        )

        # ----------------------------------------------------
        # New depth
        # ----------------------------------------------------

        Z_new = (
            R0[-1]
            * (
                np.cos(np.radians(angle_new))
                - np.cos(np.radians(angle[-1]))
            )
        ) + Z[-1]

        # ----------------------------------------------------
        # Surface reflection
        # ----------------------------------------------------

        if Z_new <= 0:

            Z_new = 0
            angle_new = -angle[-1]

            c_new = cc[0]
            G = g[0]

        # ----------------------------------------------------
        # Bottom reflection
        # ----------------------------------------------------

        elif Z_new >= zz[-1]:

            Z_new = zz[-1]
            angle_new = -angle[-1]

            c_new = cc[-1]
            G = g[-1]

        # ----------------------------------------------------
        # Normal propagation
        # ----------------------------------------------------

        else:

            z_floor = int(np.floor(Z_new))
            z_ceil = int(np.ceil(Z_new))

            if z_floor == z_ceil:
                c_new = cc[z_floor]

            else:
                c_new = (
                    cc[z_floor]
                    + (Z_new - z_floor)
                    * (
                        (cc[z_ceil] - cc[z_floor])
                        / (z_ceil - z_floor)
                    )
                )

            G = g[min(round(Z_new), len(g) - 1)]

        # ----------------------------------------------------
        # Radius of curvature
        # ----------------------------------------------------

        R_new = c_new / (
            np.cos(np.radians(angle_new)) * G
        )

        # ----------------------------------------------------
        # Store values
        # ----------------------------------------------------

        x.append(x_new)
        Z.append(Z_new)

        angle.append(angle_new)

        c0.append(c_new)
        R0.append(R_new)

    return np.array(x), np.array(Z)


def run_ray_tracing(
    theta_values,
    z0,
    xmax,
    dx,
    zz,
    cc
):
    """
    Run ray tracing for multiple start angles.
    """

    g = calculate_gradient(cc, zz)

    X = []
    ZZ = []

    for theta in theta_values:

        x, z = trace_single_ray(
            theta_start=theta,
            z0=z0,
            xmax=xmax,
            dx=dx,
            zz=zz,
            cc=cc,
            g=g
        )

        X.append(x)
        ZZ.append(z)

    return X, ZZ


# ============================================================
# PLOTTING
# ============================================================

def plot_results(
    zz,
    cc,
    X,
    ZZ,
    theta_values,
    z0,
    xmax
):
    """
    Plot SSP and ray traces.
    """

    fig = plt.figure(figsize=(14, 5))

    # --------------------------------------------------------
    # SSP plot
    # --------------------------------------------------------

    ax1 = plt.subplot(1, 5, 1)

    ax1.plot(cc, zz)

    ax1.set_ylim([zz[-1], 0])

    ax1.grid(True)

    ax1.set_title("Sound Speed Profile")
    ax1.set_xlabel("Sound Speed (m/s)")
    ax1.set_ylabel("Depth (m)")

    # --------------------------------------------------------
    # Ray plot
    # --------------------------------------------------------

    ax2 = plt.subplot(1, 5, (2, 5))

    for i in range(len(theta_values)):

        ax2.plot(
            X[i] / 1000,
            ZZ[i],
            label=f"{theta_values[i]:.2f}°"
        )

    ax2.set_xlim([0, xmax / 1000])
    ax2.set_ylim([zz[-1], 0])

    ax2.grid(True)

    ax2.set_title(
        f"Ray Trace - Source Depth: {z0} m"
    )

    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Depth (m)")

    ax2.legend(
        title="Start Angle",
        fontsize=8,
        loc="lower right"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# USER INPUT
# ============================================================

def get_user_input():
    """
    Collect user simulation settings.
    """

    print("\n=== UNDERWATER RAY TRACER ===\n")

    xmax = float(
        input("Maximum range in meters [100000]: ") or 100000
    )

    dx = float(
        input("Step size in meters [50]: ") or 50
    )

    z0 = float(
        input("Source depth in meters [300]: ") or 300
    )

    angle_min = float(
        input("Minimum launch angle in degrees [-4]: ") or -4
    )

    angle_max = float(
        input("Maximum launch angle in degrees [4]: ") or 4
    )

    angle_count = int(
        input("Number of rays [20]: ") or 20
    )

    theta_values = np.linspace(
        angle_min,
        angle_max,
        angle_count
    )

    return xmax, dx, z0, theta_values


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # User settings
    # --------------------------------------------------------

    xmax, dx, z0, theta_values = get_user_input()

    # --------------------------------------------------------
    # Create SSP
    # --------------------------------------------------------

    zz, cc = create_sound_speed_profile(
        DEFAULT_Z,
        DEFAULT_C
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    X, ZZ = run_ray_tracing(
        theta_values=theta_values,
        z0=z0,
        xmax=xmax,
        dx=dx,
        zz=zz,
        cc=cc
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        zz=zz,
        cc=cc,
        X=X,
        ZZ=ZZ,
        theta_values=theta_values,
        z0=z0,
        xmax=xmax
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()