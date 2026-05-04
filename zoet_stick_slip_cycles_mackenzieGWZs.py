"""
GZW Formation Simulation — Amundsen Gulf / Mackenzie Margin
================================================================
Physics:  Regularized Coulomb sliding law (Zoet & Iverson 2020)
          Debris-modulated Exner sediment flux
Strat:    Classic vs Lateral grounding-zone wedge (AG 2014 profile)
Concept:  Dual simulation to reproduce the two regimes described in AG 2014
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN & PHYSICS
# ─────────────────────────────────────────────────────────────────────────────
L  = 100000     # profile length (m), 100 km
nx = 1000
x  = np.linspace(0, L, nx)
dx = x[1] - x[0]   # 100 m/cell

rho_ice = 917.0
g       = 9.81
H_ice   = 400.0
N_eff   = rho_ice * g * H_ice * 0.1
U_stick = 0.5
HAS_CLASTS = True
HAS_DEBRIS = False
BED_TYPE   = 'soft'

def regularized_coulomb_drag(U_b, N, mu_c=0.6, A_visc=1e4, p=1.0,
                              has_clasts=True, has_debris=False, bed_type='soft'):
    tau_c = mu_c * N
    clast_factor = 3.0 if has_clasts else 1.0
    tau_visc = A_visc * clast_factor * U_b
    tau_b = (tau_visc**p * tau_c) / (tau_visc**p + tau_c**p) ** (1.0 / p)
    if has_debris and bed_type == 'hard':
        tau_b += 0.05 * A_visc * U_b
    regime = 'coulomb' if tau_visc >= tau_c else 'viscous'
    return float(tau_b), regime

def compute_sediment_flux(tau_b, regime, q_s_base=0.035,
                          erosion_coeff=1e-6, tau_threshold=1000.0):
    excess = max(0.0, tau_b - tau_threshold)
    q_s    = q_s_base + erosion_coeff * excess
    if regime == 'coulomb':
        q_s = min(q_s, q_s_base * 2.0)
    return q_s

def gzw_kernel(nx_domain, peak_idx, sigma_distal=300, sigma_proximal=100):
    k = np.zeros(nx_domain)
    for i in range(nx_domain):
        d      = i - peak_idx
        sigma  = sigma_distal if d > 0 else sigma_proximal
        k[i]   = np.exp(-0.5 * (d / sigma) ** 2)
    return k / k.max()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCENARIOS (AG 2014)
# ─────────────────────────────────────────────────────────────────────────────
scenarios = [
    {
        'id': 'classic',
        'name': 'Classic GZW (Terminus)',
        'target_thickness': 127.0,
        'duration_years': 2650,    # ~127 / 0.048
        'overlying_sediment': 20.0, # Buried beneath till sheets
        'title': 'Classic grounding-zone wedge | Ice-stream terminus'
    },
    {
        'id': 'lateral',
        'name': 'Lateral GZW (Margin)',
        'target_thickness': 200.0,
        'duration_years': 4170,    # ~200 / 0.048
        'overlying_sediment': 0.0,  # Near-surface, no significant overlying sediment
        'title': 'Lateral grounding-zone wedge | Lateral ice-stream margin'
    }
]

# Base bedrock
tpq_base = -350 + 10 * np.sin(np.pi * x / L) - 5 * (x / L) ** 2
tpq_base = gaussian_filter1d(tpq_base, sigma=20)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RUN SIMULATION & PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'hspace': 0.25})
fig.patch.set_facecolor('#d6d0c4')

for ax, sc in zip(axes, scenarios):
    ax.set_facecolor('#d6d0c4')
    
    # 3.1 Calculate Physics
    tau_b, regime = regularized_coulomb_drag(
        U_stick, N_eff, has_clasts=HAS_CLASTS, has_debris=HAS_DEBRIS, bed_type=BED_TYPE
    )
    q_s = compute_sediment_flux(tau_b, regime)
    
    # 3.2 Build Wedge
    gl_idx = int(60000 / dx) # Peak at 60 km
    deposit = q_s * sc['duration_years']
    kernel = gzw_kernel(nx, gl_idx)
    gzws = deposit * kernel
    
    # 3.3 Stratigraphy
    gd_background = gaussian_filter1d(np.ones(nx) * 10.0, sigma=10)
    tpq = tpq_base.copy()
    tg = tpq + gd_background + gzws
    seabed = tg + sc['overlying_sediment']
    
    # Print stats
    print(f"\n--- {sc['name']} ---")
    print(f"τ_b (kPa):      {tau_b/1000:.3f}")
    print(f"Regime:         {regime}")
    print(f"q_s (m²/yr):    {q_s:.4f}")
    print(f"Duration (yrs): {sc['duration_years']}")
    print(f"Peak thick (m): {gzws[gl_idx]:.1f}")
    print(f"Overlying (m):  {sc['overlying_sediment']}")
    
    # 3.4 Plotting
    ax.fill_between(x/1000, tpq - 100, tpq,  color='#b5d99c', alpha=0.95, zorder=1)
    ax.fill_between(x/1000, tpq, tg,         color='#e07b39', alpha=0.88, zorder=2)
    if sc['overlying_sediment'] > 0:
        ax.fill_between(x/1000, tg, seabed,  color='#f4a7b9', alpha=0.85, zorder=3)

    ax.plot(x/1000, tpq,    color='#009688', lw=2.2, zorder=6, label='Bedrock (TPQ)')
    ax.plot(x/1000, tg,     color='#8B4513', lw=1.6, zorder=6, label='Top Glacial (TG)')
    ax.plot(x/1000, seabed, color='#00ACC1', lw=2.0, zorder=7, label='Seabed')

    # Dimensions Annotation
    ax.axvline(x[gl_idx]/1000, color='white', lw=0.8, alpha=0.35, linestyle=':', zorder=8)
    ax.annotate(f"{sc['target_thickness']:.0f} m thick",
                xy=(x[gl_idx]/1000, tg[gl_idx]),
                xytext=(x[gl_idx]/1000 - 4.0, tg[gl_idx] + 25),
                fontsize=11, color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.annotate('', xy=(20, -100), xytext=(70, -100), arrowprops=dict(arrowstyle='<->', color='#FFD700', lw=2.5))
    ax.text(45, -95, '~50 km long', ha='center', fontsize=11, color='black', fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))

    # Styling
    ax.text(0.01,  0.05, 'Sediment transport →', transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.4', fc='white', alpha=0.9))
    
    patches_b = [
        mpatches.Patch(color='#b5d99c', label='Basement'),
        mpatches.Patch(color='#e07b39', label='GZW'),
        mpatches.Patch(color='#f4a7b9', label='Overlying Sediment' if sc['overlying_sediment'] > 0 else 'None'),
    ]
    ax.legend(handles=patches_b, loc='lower right', fontsize=10, framealpha=0.88)
    ax.set_title(sc['title'], fontsize=14, pad=10, fontweight='bold')
    ax.set_xlabel("Distance (km)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_xlim(0, 100); ax.set_ylim(-450, 0)
    ax.grid(alpha=0.2, color='white')

print("\n" + "─" * 40)
plt.savefig('gzw_Mackenzie_strat.png', dpi=150, bbox_inches='tight')
print("\nFigure saved → gzw_Mackenzie_strat.png")
