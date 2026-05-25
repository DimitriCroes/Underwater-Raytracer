import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt

from raytracer import (
    create_sound_speed_profile,
    run_ray_tracing
)

FT_TO_M = 0.3048
MS_TO_FPS = 3.28084


# ============================================================
# DEFAULT DATA
# ============================================================

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
# UNIT HELPERS
# ============================================================

def convert_depth(value, mode):
    return value * FT_TO_M if mode == "ft" else value


def convert_speed(value, mode):
    return value / MS_TO_FPS if mode == "ft/s" else value


# ============================================================
# GUI APPLICATION
# ============================================================

class RayTracerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Underwater Ray Tracer")

        # ----------------------------
        # Variables
        # ----------------------------

        self.unit_depth = tk.StringVar(value="m")
        self.unit_speed = tk.StringVar(value="m/s")

        self.xmax = tk.DoubleVar(value=100000)
        self.dx = tk.DoubleVar(value=50)
        self.z0 = tk.DoubleVar(value=300)
        self.cmax = tk.DoubleVar(value=2200)

        self.angle_min = tk.DoubleVar(value=-4)
        self.angle_max = tk.DoubleVar(value=4)
        self.angle_count = tk.IntVar(value=20)

        # ----------------------------
        # Layout
        # ----------------------------

        self.build_inputs()
        self.build_ssp_table()
        self.build_buttons()

    # ========================================================
    # INPUT PANEL
    # ========================================================

    def build_inputs(self):

        frame = ttk.LabelFrame(self.root, text="Simulation Settings")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(frame, text="Max range (m):").grid(row=0, column=0)
        ttk.Entry(frame, textvariable=self.xmax).grid(row=0, column=1)

        ttk.Label(frame, text="Step dx (m):").grid(row=1, column=0)
        ttk.Entry(frame, textvariable=self.dx).grid(row=1, column=1)

        ttk.Label(frame, text="Source depth (m):").grid(row=2, column=0)
        ttk.Entry(frame, textvariable=self.z0).grid(row=2, column=1)

        ttk.Label(frame, text="Max depth (m):").grid(row=3, column=0)
        ttk.Entry(frame, textvariable=self.cmax).grid(row=3, column=1)

        ttk.Label(frame, text="Angle min:").grid(row=4, column=0)
        ttk.Entry(frame, textvariable=self.angle_min).grid(row=4, column=1)

        ttk.Label(frame, text="Angle max:").grid(row=5, column=0)
        ttk.Entry(frame, textvariable=self.angle_max).grid(row=5, column=1)

        ttk.Label(frame, text="Ray count:").grid(row=6, column=0)
        ttk.Entry(frame, textvariable=self.angle_count).grid(row=6, column=1)

    # ========================================================
    # SSP TABLE
    # ========================================================

    def build_ssp_table(self):

        frame = ttk.LabelFrame(self.root, text="Sound Speed Profile (SSP)")
        frame.grid(row=1, column=0, padx=10, pady=10)

        # unit selectors
        unit_frame = ttk.Frame(frame)
        unit_frame.grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(unit_frame, text="Depth unit:").grid(row=0, column=0)
        ttk.OptionMenu(unit_frame, self.unit_depth, "m", "m", "ft").grid(row=0, column=1)

        ttk.Label(unit_frame, text="Speed unit:").grid(row=0, column=2)
        ttk.OptionMenu(unit_frame, self.unit_speed, "m/s", "m/s", "ft/s").grid(row=0, column=3)

        # table headers
        ttk.Label(frame, text="Depth").grid(row=1, column=0)
        ttk.Label(frame, text="Speed").grid(row=1, column=1)

        self.table_frame = ttk.Frame(frame)
        self.table_frame.grid(row=2, column=0, columnspan=3)

        self.rows = []
        self.load_default_ssp()

        ttk.Button(frame, text="Add Row", command=self.add_row).grid(row=3, column=0)
        ttk.Button(frame, text="Remove Row", command=self.remove_row).grid(row=3, column=1)

    def load_default_ssp(self):
        for z, c in zip(DEFAULT_Z, DEFAULT_C):
            self.add_row(z, c)

    def add_row(self, z_val=0.0, c_val=1500.0):

        row = len(self.rows)

        z_entry = ttk.Entry(self.table_frame, width=10)
        c_entry = ttk.Entry(self.table_frame, width=10)

        z_entry.insert(0, str(z_val))
        c_entry.insert(0, str(c_val))

        z_entry.grid(row=row, column=0)
        c_entry.grid(row=row, column=1)

        self.rows.append((z_entry, c_entry))

    def remove_row(self):
        if self.rows:
            z, c = self.rows.pop()
            z.destroy()
            c.destroy()

    # ========================================================
    # DATA EXTRACTION
    # ========================================================

    def get_ssp(self):

        z_list = []
        c_list = []

        for z_entry, c_entry in self.rows:

            try:
                z = float(z_entry.get())
                c = float(c_entry.get())
            except ValueError:
                continue

            # unit conversion
            z = convert_depth(z, self.unit_depth.get())
            c = convert_speed(c, self.unit_speed.get())

            z_list.append(z)
            c_list.append(c)

        return np.array(z_list), np.array(c_list)

    # ========================================================
    # RUN SIMULATION
    # ========================================================

    def run(self):

        try:
            z, c = self.get_ssp()

            zz, cc = create_sound_speed_profile(z, c, self.cmax.get())

            theta = np.linspace(
                self.angle_min.get(),
                self.angle_max.get(),
                self.angle_count.get()
            )

            X, ZZ = run_ray_tracing(
                theta_values=theta,
                z0=self.z0.get(),
                xmax=self.xmax.get(),
                dx=self.dx.get(),
                zz=zz,
                cc=cc
            )

            self.plot(zz, cc, X, ZZ, theta)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ========================================================
    # PLOT
    # ========================================================

    def plot(self, zz, cc, X, ZZ, theta):

        fig = plt.figure(figsize=(14, 5))

        ax1 = plt.subplot(1, 5, 1)
        ax1.plot(cc, zz)
        ax1.invert_yaxis()
        ax1.set_title("SSP")
        ax1.set_xlim(left=min(cc), right=max(cc))
        ax1.set_ylim(bottom=max(zz) + 10, top=0)
        ax1.set_xlabel("Sound Speed (m/s)")
        ax1.set_ylabel("Depth (m)")
        ax1.grid()

        ax2 = plt.subplot(1, 5, (2, 5))

        for i in range(len(theta)):
            ax2.plot(X[i] / 1000, ZZ[i])

        ax2.invert_yaxis()
        ax2.set_title("Ray Tracing")
        ax2.set_xlim(left=0, right=self.xmax.get() / 1000)
        ax2.set_ylim(bottom=max(zz) + 10, top=0)
        ax2.set_xlabel("Range (km)")

        plt.show()

    # ========================================================
    # BUTTONS
    # ========================================================

    def build_buttons(self):

        frame = ttk.Frame(self.root)
        frame.grid(row=2, column=0, pady=10)

        ttk.Button(frame, text="RUN SIMULATION", command=self.run).grid(row=0, column=0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RayTracerGUI(root)
    root.mainloop()