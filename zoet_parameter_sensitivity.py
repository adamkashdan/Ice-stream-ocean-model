"""
Parameter Sensitivity Analysis for Zoet & Iverson (2020) Law
================================================================
A script to "play" with the sliding law parameters and visualize 
how they affect Basal Shear Stress (τ_b) and Sediment Flux (q_s).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def regularized_coulomb_drag(U_b, N, mu_c=0.6, A_visc=1e4, p=1.0,
                              has_clasts=True, has_debris=False, bed_type='soft'):
    tau_c = mu_c * N
    clast_factor = 3.0 if has_clasts else 1.0
    tau_visc = A_visc * clast_factor * U_b
    tau_b = (tau_visc**p * tau_c) / (tau_visc**p + tau_c**p) ** (1.0 / p)
    if has_debris and bed_type == 'hard':
        tau_b += 0.05 * A_visc * U_b
    return float(tau_b)

def compute_sediment_flux(tau_b, q_s_base=0.035, erosion_coeff=1e-6, tau_threshold=1000.0):
    excess = max(0.0, tau_b - tau_threshold)
    q_s = q_s_base + erosion_coeff * excess
    # We won't cap it here so we can see the full theoretical curve
    return q_s

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMETER SWEEPS
# ─────────────────────────────────────────────────────────────────────────────
rho_ice = 917.0
g = 9.81
H_ice = 400.0
N_eff = rho_ice * g * H_ice * 0.1  # ~360 kPa

U_speeds = np.linspace(0, 5, 100) # Sliding speeds from 0 to 5 m/yr

# Variations to test:
mu_c_values = [0.3, 0.6, 0.8]  # Friction coefficient (Soft -> Hard till)
clast_settings = [True, False] # Presence of large clasts

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#f7f7f7')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
linestyles = ['-', '--']

for i, mu in enumerate(mu_c_values):
    for j, clast in enumerate(clast_settings):
        tau_b_list = []
        qs_list = []
        for u in U_speeds:
            t = regularized_coulomb_drag(u, N_eff, mu_c=mu, has_clasts=clast)
            q = compute_sediment_flux(t)
            tau_b_list.append(t / 1000) # Convert to kPa
            qs_list.append(q)
            
        label = f"μ_c={mu} | Clasts={clast}"
        ax1.plot(U_speeds, tau_b_list, color=colors[i], linestyle=linestyles[j], lw=2, label=label)
        ax2.plot(U_speeds, qs_list, color=colors[i], linestyle=linestyles[j], lw=2, label=label)

# Panel 1: Shear Stress
ax1.set_title("Regularized Coulomb Drag (Zoet & Iverson)", fontsize=12)
ax1.set_xlabel("Basal Sliding Speed, $U_b$ (m/yr)")
ax1.set_ylabel("Basal Shear Stress, $\\tau_b$ (kPa)")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

# Panel 2: Sediment Flux
ax2.set_title("Exner Sediment Flux", fontsize=12)
ax2.set_xlabel("Basal Sliding Speed, $U_b$ (m/yr)")
ax2.set_ylabel("Sediment Flux, $q_s$ (m²/yr)")
ax2.grid(alpha=0.3)

plt.suptitle("Sensitivity Analysis: GZW Model Parameters", fontsize=14, fontweight='bold', y=1.02)
plt.savefig('zoet_sensitivity_playground.png', dpi=150, bbox_inches='tight')
print("Saved sensitivity plot -> zoet_sensitivity_playground.png")
