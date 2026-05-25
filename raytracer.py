import numpy as np
from scipy.interpolate import PchipInterpolator

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