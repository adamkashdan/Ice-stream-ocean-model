"""
GZW Formation Simulation — calibrated to R_2018 seismic profile
================================================================
Physics:  Regularized Coulomb sliding law (Zoet & Iverson 2020)
          Debris-modulated Exner sediment flux
Strat:    TPQ → GD (+ GZW mounds) → TG → DHD → Seabed
Domain:   36 km N→S profile, calibrated to observed core locations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN
# ─────────────────────────────────────────────────────────────────────────────
L  = 36000      # profile length (m), N→S
nx = 360
x  = np.linspace(0, L, nx)
dx = x[1] - x[0]   # 100 m/cell


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZOET SLIDING LAW — Regularized Coulomb (Zoet & Iverson 2020)
# ─────────────────────────────────────────────────────────────────────────────
def regularized_coulomb_drag(U_b, N, mu_c=0.6, A_visc=1e4, p=1.0,
                              has_clasts=True, has_debris=False,
                              bed_type='soft'):
    """
    Basal shear stress via Regularized Coulomb form.

    Parameters
    ----------
    U_b         : float  — basal sliding speed (m/yr)
    N           : float  — effective pressure (Pa)
    mu_c        : float  — Coulomb friction coefficient
    A_visc      : float  — viscous drag coefficient (Pa yr/m)
    p           : float  — rollover sharpness [Zoet seminar 19:04]
    has_clasts  : bool   — large clasts present → lower transition U_t [16:28]
    has_debris  : bool   — debris-laden basal ice → rate-strengthening [48:21]
    bed_type    : str    — 'soft' (till) or 'hard' (bedrock)

    Returns
    -------
    tau_b  : float — basal drag (Pa)
    regime : str   — 'viscous' or 'coulomb'
    """
    tau_c    = mu_c * N
    # Large clasts concentrate stress → Coulomb transition at lower U_b [17:42]
    clast_factor = 3.0 if has_clasts else 1.0
    tau_visc = A_visc * clast_factor * U_b
    # Regularized Coulomb rollover [19:10]
    tau_b    = (tau_visc**p * tau_c) / (tau_visc**p + tau_c**p) ** (1.0 / p)
    # Debris on hard beds: additional rate-strengthening term [48:21–49:00]
    if has_debris and bed_type == 'hard':
        tau_b += 0.05 * A_visc * U_b
    regime = 'coulomb' if tau_visc >= tau_c else 'viscous'
    return float(tau_b), regime


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEDIMENT FLUX — debris-modulated Exner logic
# ─────────────────────────────────────────────────────────────────────────────
def compute_sediment_flux(tau_b, regime, q_s_base=0.035,
                          erosion_coeff=1e-6, tau_threshold=1000.0):
    """
    Sediment flux at grounding line modulated by basal drag state.

    In Coulomb regime drag saturates → flux caps (stabilizing) [Zoet 51:02]
    """
    excess = max(0.0, tau_b - tau_threshold)
    q_s    = q_s_base + erosion_coeff * excess
    if regime == 'coulomb':
        q_s = min(q_s, q_s_base * 2.0)   # Coulomb cap
    return q_s


# ─────────────────────────────────────────────────────────────────────────────
# 4. GZW SPATIAL KERNEL — asymmetric wedge shape
#    Steeper ice-proximal face, longer distal tail
# ─────────────────────────────────────────────────────────────────────────────
def gzw_kernel(nx_domain, peak_idx, sigma_distal=12, sigma_proximal=6):
    k = np.zeros(nx_domain)
    for i in range(nx_domain):
        d      = i - peak_idx
        sigma  = sigma_distal if d > 0 else sigma_proximal
        k[i]   = np.exp(-0.5 * (d / sigma) ** 2)
    return k / k.max()


# ─────────────────────────────────────────────────────────────────────────────
# 5. PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
rho_ice = 917.0     # kg/m³
g       = 9.81      # m/s²
H_ice   = 400.0     # representative ice thickness at GL (m)
N_eff   = rho_ice * g * H_ice * 0.1    # effective pressure (Pa), ~10% flotation

U_stick = 0.5       # near-zero sliding speed during stick phase (m/yr)

HAS_CLASTS = True   # till with large clasts [Zoet 16:28]
HAS_DEBRIS = False  # no frozen-fringe debris (marine soft bed)
BED_TYPE   = 'soft'


# ─────────────────────────────────────────────────────────────────────────────
# 6. GROUNDING LINE POSITIONS & STICK DURATIONS
#    Anchored to core locations in R_2018 seismic profile
#    Ice retreated S→N; outer GZW formed first
# ─────────────────────────────────────────────────────────────────────────────
gl_outer  = int(24000 / dx)   # ~HH12-1209GC
gl_middle = int(15000 / dx)   # ~JM10-12GC
gl_inner  = int( 5000 / dx)   # ~NP05-86GC

retreat_steps = [gl_outer, gl_middle, gl_inner]
durations     = [600, 1200, 800]    # years: tuned so middle GZW is tallest
labels        = ['Outer', 'Middle', 'Inner']


# ─────────────────────────────────────────────────────────────────────────────
# 7. STICK–SLIP LOOP — build GZW deposits
# ─────────────────────────────────────────────────────────────────────────────
gzws       = np.zeros(nx)
stick_info = []

print(f"\n{'GZW':>6}  {'GL (km)':>8}  {'τ_b (kPa)':>10}  "
      f"{'Regime':>8}  {'q_s (m²/yr)':>12}  {'Peak dep. (m)':>14}")
print("─" * 65)

for i, (gl_idx, years) in enumerate(zip(retreat_steps, durations)):
    tau_b, regime = regularized_coulomb_drag(
        U_stick, N_eff,
        has_clasts=HAS_CLASTS,
        has_debris=HAS_DEBRIS,
        bed_type=BED_TYPE,
    )
    q_s     = compute_sediment_flux(tau_b, regime)
    deposit = q_s * years
    kernel  = gzw_kernel(nx, gl_idx)
    gzws   += deposit * kernel
    stick_info.append((gl_idx, labels[i], deposit))

    print(f"{labels[i]:>6}  {x[gl_idx]/1000:>8.1f}  {tau_b/1000:>10.3f}  "
          f"{regime:>8}  {q_s:>12.4f}  {deposit:>14.1f}")

gzws = np.clip(gzws, 0, 38.0)   # cap to seismic-observed max ~38 m (50 ms TWT)


# ─────────────────────────────────────────────────────────────────────────────
# 8. STRATIGRAPHIC HORIZONS
#    Bottom to top: PQ basement → TPQ → GD → TG → DHD → Seabed
# ─────────────────────────────────────────────────────────────────────────────

# ── TPQ: top pre-Quaternary unconformity ─────────────────────────────────────
# Relatively flat, gentle basin sag, narrow ridge at ~2 km [R_2018 image]
tpq = -195 + 10 * np.sin(np.pi * x / L) - 5 * (x / L) ** 2
tpq = gaussian_filter1d(tpq, sigma=20)
ridge_idx = int(2000 / dx)
tpq += 18 * np.exp(-0.5 * ((np.arange(nx) - ridge_idx) / 4) ** 2)

# ── GD: glacial deposits (TPQ → TG) ─────────────────────────────────────────
# Background thickness from R_2018:
#   ~4 m   between middle and outer GZW (thin zone)
#   ~10 m  between middle and inner GZW (basin)
#   ~7.5 m elsewhere (background)
gd_background = np.ones(nx) * 7.5
gd_background[(x >= 15000) & (x <= 24000)] = 4.0
gd_background[(x >=  5000) & (x <= 15000)] = 10.0
gd_background = gaussian_filter1d(gd_background, sigma=10)

# ── TG: top glacial unconformity ─────────────────────────────────────────────
tg = tpq + gd_background + gzws

# ── DHD: deglacial to Holocene deposits (TG → Seabed) ────────────────────────
# Conformable drape; thicker in topographic lows (hemipelagic settling)
tg_norm   = (tg - tg.min()) / (tg.max() - tg.min())
dhd_thick = 4.0 + 3.0 * (1.0 - tg_norm)
dhd_thick = gaussian_filter1d(dhd_thick, sigma=15)

# ── Seabed ────────────────────────────────────────────────────────────────────
seabed = tg + dhd_thick


# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOT — two panels mirroring R_2018 layout
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                         gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.08})

# ── Panel A: horizon lines only ───────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#f0ece0')
ax.fill_between(x/1000, tg, seabed, color='#f4a7b9', alpha=0.6, label='DHD', zorder=3)
ax.plot(x/1000, tpq,    color='#009688', lw=2.0, label='TPQ',    zorder=5)
ax.plot(x/1000, tg,     color='#8B4513', lw=1.8, label='TG',     zorder=5)
ax.plot(x/1000, seabed, color='#00ACC1', lw=2.0, label='Seabed', zorder=6)
ax.annotate('narrow TPQ ridge',
            xy=(2, tpq[ridge_idx]),
            xytext=(5.5, tpq[ridge_idx] + 9),
            fontsize=8, color='#009688',
            arrowprops=dict(arrowstyle='->', color='#009688', lw=1.0))
ax.set_ylabel("Elevation (m)", fontsize=9)
ax.set_xlim(0, 36); ax.set_ylim(-210, -125)
ax.set_xticklabels([]); ax.grid(alpha=0.2)
ax.text(0.01,  0.08, 'N', transform=ax.transAxes, fontsize=11, fontweight='bold')
ax.text(0.985, 0.08, 'S', transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right')
ax.set_title(
    "GZW Stratigraphy — R_2018 calibrated  |  "
    "Regularized Coulomb sliding law (Zoet) + debris-modulated Exner flux",
    fontsize=10, pad=6)
leg_a = [
    plt.Line2D([0],[0], color='#009688', lw=2.0, label='TPQ'),
    plt.Line2D([0],[0], color='#8B4513', lw=1.8, label='TG'),
    plt.Line2D([0],[0], color='#00ACC1', lw=2.0, label='Seabed'),
    mpatches.Patch(color='#f4a7b9', alpha=0.7, label='DHD'),
]
ax.legend(handles=leg_a, loc='upper right', fontsize=8, framealpha=0.85, ncol=2)

# ── Panel B: filled stratigraphy (R_2018 panel b style) ──────────────────────
ax = axes[1]
fig.patch.set_facecolor('#d6d0c4')
ax.set_facecolor('#d6d0c4')

# PQ basement (light green) — below TPQ
ax.fill_between(x/1000, tpq - 60, tpq,  color='#b5d99c', alpha=0.95, zorder=1)
# GD unit (orange) — TPQ to TG
ax.fill_between(x/1000, tpq, tg,         color='#e07b39', alpha=0.88, zorder=2)
# DHD unit (pink) — TG to seabed
ax.fill_between(x/1000, tg,  seabed,     color='#f4a7b9', alpha=0.85, zorder=3)

# Horizon lines
ax.plot(x/1000, tpq,    color='#009688', lw=2.2, zorder=6, label='TPQ')
ax.plot(x/1000, tg,     color='#8B4513', lw=1.6, zorder=6, label='TG')
ax.plot(x/1000, seabed, color='#00ACC1', lw=2.0, zorder=7, label='Seabed')

# GZW annotations
for gl_idx, name, deposit in stick_info:
    peak = gzws[gl_idx]
    top  = tg[gl_idx]
    ax.axvline(x[gl_idx]/1000, color='white', lw=0.8, alpha=0.35,
               linestyle=':', zorder=8)
    ax.annotate(f'{name} GZW\n~{peak:.0f} m',
                xy=(x[gl_idx]/1000, top),
                xytext=(x[gl_idx]/1000 + 1.0, top + 7),
                fontsize=7.5, color='white', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white', lw=0.9))

# R_2018 thickness annotations
for xpos, xidx, label in [
        (19.5, 190, '~4 m\n(thin zone)'),
        (10.0, 100, '~10 m\n(basin)'),
        (30.0, 300, '~7.5 m\n(background)')]:
    ax.annotate(label,
                xy=(xpos, tpq[xidx] + 3), fontsize=7.5,
                color='#222', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7))

# DHD callout
dhd_mid = int(nx / 2)
ax.annotate(f'DHD ~{dhd_thick[dhd_mid]:.1f} m\n(deglacial–Holocene)',
            xy=(18, tg[dhd_mid] + dhd_thick[dhd_mid] / 2),
            xytext=(22, tg[dhd_mid] + 15),
            fontsize=7.5, color='#880e4f',
            arrowprops=dict(arrowstyle='->', color='#880e4f', lw=0.9))

# N/S labels
ax.text(0.01,  0.05, 'N', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.text(0.985, 0.05, 'S', transform=ax.transAxes, fontsize=12, fontweight='bold', ha='right')

# Scale bar (5 km)
ax.annotate('', xy=(33, -198), xytext=(28, -198),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(30.5, -201, '5 km', ha='center', fontsize=8)

# Legend
patches_b = [
    mpatches.Patch(color='#b5d99c', label='PQ (basement)'),
    mpatches.Patch(color='#e07b39', label='GD'),
    mpatches.Patch(color='#f4a7b9', label='DHD'),
    plt.Line2D([0],[0], color='#009688', lw=2.2, label='TPQ'),
    plt.Line2D([0],[0], color='#8B4513', lw=1.6, label='TG'),
    plt.Line2D([0],[0], color='#00ACC1', lw=2.0, label='Seabed'),
]
ax.legend(handles=patches_b, loc='upper right', fontsize=8,
          framealpha=0.88, ncol=3)
ax.set_xlabel("Distance (km)  [N → S]", fontsize=10)
ax.set_ylabel("Elevation (m)", fontsize=9)
ax.set_xlim(0, 36); ax.set_ylim(-210, -125)
ax.grid(alpha=0.12, color='white')

plt.savefig('gzw_R2018_full_strat.png', dpi=150, bbox_inches='tight')
print("\nFigure saved → gzw_R2018_full_strat.png")
plt.show()
