"""
Transient subglacial water pressure + Zoet & Iverson (2020) sliding law.

Solves the pressure diffusion PDE from Tsai et al. (2021):
    dp'/dt = kappa * d²p'/dx² - eps * p'

Then computes basal sliding velocity via the Regularized Coulomb law
(Zoet & Iverson 2020) instead of the original Budd-type power law:

    ┌──────────────────────────────────────────────────────────────────┐
    │ ORIGINAL  (Tsai 2021, Budd-type):                                │
    │   u = ub0 · (1 − ku · p_norm)^(−m)                              │
    │                                                                  │
    │ NEW  (Zoet & Iverson 2020, Regularized Coulomb):                 │
    │   τ_b = τ_c · u / (u + u₀)                                      │
    │                                                                  │
    │   τ_c = N · tan(φ)          Coulomb shear-strength limit  (Pa)  │
    │   N(t) = σ₀ · [f_N0 − p_norm(t)]   effective pressure  (Pa)    │
    │   p_norm(t) = p'(t,0) / (k_Q · σ₀)  dimensionless pressure     │
    │   f_N0  background effective-pressure fraction (0–1)            │
    │   u₀    transition velocity  (m/yr)                             │
    │                                                                  │
    │   Inverted for u  given  τ_b = τ_d  (force balance):           │
    │   u = u₀ · τ_d / (τ_c − τ_d)                                   │
    └──────────────────────────────────────────────────────────────────┘

Pressure-scale note
-------------------
p'(t,0) in the PDE has amplitude ≈ ±0.33 · k_Q · σ₀  for the synthetic
Q forcing used here.  The background effective pressure f_N0 must exceed
this amplitude (f_N0 > 0.37) to keep N_eff > 0 at all times.
With f_N0 = 0.45 and u₀ = 1322 m/yr the mean sliding velocity ≈ 149 m/yr.

References
----------
Tsai VC, Smith LC, Gardner AS, Seroussi H (2021).
    Journal of Glaciology 68(268), 390–400.
    https://doi.org/10.1017/jog.2021.103
Zoet LK, Iverson NR (2020).
    A slip law for glaciers on deformable beds.
    Science 368, 76–78.
    https://doi.org/10.1126/science.aaz1183
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRESSURE DIFFUSION ODE  (Tsai et al. 2021 — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def diffusion_rhs(t, p, dx, tt_q, q_anomaly, kappa, eps):
    """
    RHS of the normalized pressure diffusion PDE.

        dp'/dt = kappa · d²p'/dx² − eps · p'

    Boundary conditions
    -------------------
    x = 0  (moulin inlet) : Neumann — dp'/dx = −Q_anomaly(t)
    x = L  (terminus)     : Dirichlet — p' fixed (dpdt = 0)

    Parameters
    ----------
    t         : float — current time (days)
    p         : ndarray — normalized pressure p' at all nodes (m³/s)
    dx        : float — spatial step (km)
    tt_q      : ndarray — time vector for flux interpolation (days)
    q_anomaly : ndarray — Q(t) − Q_mean  (m³/s)
    kappa     : float — hydraulic diffusivity (km²/day)
    eps       : float — scaled inverse viscosity (1/day)

    Returns
    -------
    dpdt : ndarray — dp'/dt at every node
    """
    pss  = 0.0
    dpdt = np.zeros_like(p)

    q_now = float(interp1d(tt_q, q_anomaly,
                           bounds_error=False,
                           fill_value=(q_anomaly[0], q_anomaly[-1]))(t))

    # Left BC: Neumann (prescribed moulin flux)
    dpdt[0] = kappa * ((p[1] - p[0]) / dx + q_now) / dx - eps * (p[0] - pss)

    # Interior: 2nd-order central finite differences
    dpdt[1:-1] = (kappa / dx**2 * (p[2:] - 2.0 * p[1:-1] + p[:-2])
                  - eps * (p[1:-1] - pss))

    # Right BC: Dirichlet (atmospheric pressure → no tendency)
    dpdt[-1] = 0.0
    return dpdt


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ZOET & IVERSON (2020) REGULARIZED COULOMB SLIDING LAW
# ─────────────────────────────────────────────────────────────────────────────

def zoet_velocity(tau_d, p_norm, f_N0, sigma_0, phi_deg, u0):
    """
    Regularized Coulomb basal sliding velocity (Zoet & Iverson 2020).

    Physics
    -------
    Basal drag is bounded by the Coulomb shear-strength of the till:

        τ_b = τ_c · u / (u + u₀)      Regularized Coulomb law

    where:
        τ_c = N · tan(φ)              Coulomb limit  (Pa)
        N   = σ₀ · (f_N0 − p_norm)   effective pressure  (Pa)
        p_norm = p'/(k_Q·σ₀)          dimensionless pressure perturbation

    Setting τ_b = τ_d  (driving stress = basal drag, standard assumption)
    and inverting for u:

        u = u₀ · τ_d / (τ_c − τ_d)

    The velocity diverges as τ_d → τ_c  (onset of streaming / fast flow).
    The denominator is clamped to a small positive floor to keep u finite.

    Parameters
    ----------
    tau_d   : float          — driving stress  (Pa)
    p_norm  : ndarray        — dimensionless pressure perturbation p'/(k_Q·σ₀)
    f_N0    : float          — background effective-pressure fraction (0–1)
    sigma_0 : float          — ice overburden pressure σ₀ = ρ_i g H  (Pa)
    phi_deg : float          — till friction angle φ  (degrees)
    u0      : float          — transition velocity u₀  (m/yr)

    Returns
    -------
    u_b   : ndarray — basal sliding velocity  (m/yr)
    tau_c : ndarray — Coulomb limit τ_c  (Pa)
    N_eff : ndarray — effective pressure N  (Pa)
    """
    # Effective pressure — must stay positive for the law to be physical
    N_eff = sigma_0 * (f_N0 - p_norm)
    N_eff = np.maximum(N_eff, 1.0)          # hard floor: 1 Pa (avoids τ_c = 0)

    tau_c = N_eff * np.tan(np.radians(phi_deg))
    tau_c = np.maximum(tau_c, tau_d * 1.001)  # ensure τ_c > τ_d always

    # Invert Regularized Coulomb for u
    denom = tau_c - tau_d
    denom = np.maximum(denom, 1e-3 * tau_c)   # clamp denominator

    u_b = np.maximum(u0 * tau_d / denom, 0.0)
    return u_b, tau_c, N_eff


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GAUSS-NEWTON INVERSION FOR ZOET PARAMETERS  (φ, u₀)
# ─────────────────────────────────────────────────────────────────────────────

def fit_zoet_parameters(v_obs, p_norm_tk, tau_d, sigma_0,
                        f_N0, phi0=26.0, u0_0=1000.0, n_iter=40):
    """
    Fit till friction angle φ and transition velocity u₀ to observed
    velocities using Gauss-Newton iterations with Levenberg-Marquardt damping.

    Model:
        u = u₀ · τ_d / (σ₀·(f_N0 − p_norm)·tan(φ) − τ_d)

    Parameters
    ----------
    v_obs      : ndarray — observed velocities at fitting times  (m/yr)
    p_norm_tk  : ndarray — dimensionless pressure at fitting times
    tau_d      : float   — driving stress  (Pa)
    sigma_0    : float   — overburden pressure  (Pa)
    f_N0       : float   — background effective-pressure fraction
    phi0       : float   — initial guess for φ  (degrees)
    u0_0       : float   — initial guess for u₀  (m/yr)
    n_iter     : int     — number of Gauss-Newton iterations

    Returns
    -------
    phi_fit : float   — best-fit friction angle  (degrees)
    u0_fit  : float   — best-fit transition velocity  (m/yr)
    v_fit   : ndarray — predicted velocities at fitting times  (m/yr)
    """
    m = np.array([phi0, u0_0], dtype=float)

    for _ in range(n_iter):
        phi_r  = np.radians(m[0])
        N_eff  = np.maximum(sigma_0 * (f_N0 - p_norm_tk), 1.0)
        tau_c  = np.maximum(N_eff * np.tan(phi_r), tau_d * 1.001)
        denom  = np.maximum(tau_c - tau_d, 1e-3 * tau_c)
        v_pred = m[1] * tau_d / denom

        # Jacobian:  ∂v/∂φ  and  ∂v/∂u₀
        dtauc_dphi = N_eff / np.cos(phi_r)**2 * (np.pi / 180.0)
        dv_dphi    = m[1] * tau_d / denom**2 * dtauc_dphi
        dv_du0     = tau_d / denom

        G  = np.column_stack([dv_dphi, dv_du0])
        H  = G.T @ G
        lm = 0.1 * np.diag(np.diag(H))
        dm = np.linalg.solve(H + lm, G.T @ (v_obs - v_pred))
        m  = m + dm
        m[0] = np.clip(m[0], 5.0, 50.0)    # φ ∈ [5°, 50°]
        m[1] = np.maximum(m[1], 1.0)        # u₀ > 0

    phi_fit, u0_fit = float(m[0]), float(m[1])
    _, tau_c_f, N_f = zoet_velocity(tau_d, p_norm_tk, f_N0, sigma_0, phi_fit, u0_fit)
    v_fit = u0_fit * tau_d / np.maximum(tau_c_f - tau_d, 1e-3 * tau_c_f)
    return phi_fit, u0_fit, v_fit


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def transient_water_pressure_zoet():
    """
    Full pipeline:
      1. Solve pressure diffusion PDE  (Tsai et al. 2021)
      2. Compute dimensionless pressure perturbation p_norm
      3. Compute Zoet & Iverson (2020) sliding velocity
      4. Invert Zoet parameters (φ, u₀) against synthetic GPS
      5. Plot five panels
    """

    # ── 4.1  Input data  (* replace with real field observations *) ──────────
    tt = np.arange(0.0, 10.45, 0.05)                        # time for Q  (days)
    Q  = 16.0 + 8.0 * np.cos(2.0 * np.pi * tt + 0.6)       # flux  (m³/s)

    # ── 4.2  Physical parameters ─────────────────────────────────────────────
    L     = 42.0    # domain length    (km)
    rhoi  = 920.0   # ice density      (kg/m³)
    g     = 9.8     # gravity          (m/s²)
    H     = 934.0   # ice thickness    (m)

    sigma_0 = rhoi * g * H                    # overburden   (Pa)  ≈ 8.42 MPa
    alpha   = H / (L * 1e3)                   # surface slope ≈ H/L
    tau_d   = rhoi * g * H * alpha            # driving stress (Pa) ≈ 187 kPa

    Q_mean  = np.mean(Q)
    k_Q     = 0.5 * L * Q_mean / sigma_0      # hydraulic coefficient

    kappa  = 1400.0   # hydraulic diffusivity (km²/day)  — Tsai 2021 best-fit
    eps    =    4.0   # scaled inverse viscosity (1/day) — Tsai 2021 best-fit

    # ── Zoet & Iverson (2020) parameters ────────────────────────────────────
    # f_N0: background effective-pressure fraction (drainage-system efficiency).
    # Must satisfy f_N0 > p_norm_amplitude ≈ 0.33 to keep N_eff > 0 always.
    # f_N0 = 0.45 → N₀ ≈ 3790 kPa (45 % flotation), typical of an active
    # channelized drainage system (Zoet & Iverson 2020, Supplement).
    f_N0   = 0.45    # background effective-pressure fraction
    phi0   = 26.0    # initial guess for friction angle  (degrees)
    # u₀ initial guess chosen so mean velocity ≈ 149 m/yr at mean p_norm = 0:
    #   u₀ = u_mean · (τ_c − τ_d) / τ_d  with τ_c = f_N0·σ₀·tan(φ)
    tau_c0 = f_N0 * sigma_0 * np.tan(np.radians(phi0))
    u0_0   = 149.0 * (tau_c0 - tau_d) / tau_d    # ≈ 1322 m/yr

    print("=" * 65)
    print("  Tsai et al. (2021) hydrology  ×  Zoet & Iverson (2020) sliding")
    print("=" * 65)
    print(f"  σ₀    = {sigma_0/1e6:.3f} MPa    τ_d  = {tau_d/1e3:.2f} kPa")
    print(f"  f_N0  = {f_N0:.2f}   →  N₀  = {f_N0*sigma_0/1e3:.0f} kPa")
    print(f"  φ₀    = {phi0}°       u₀   ≈ {u0_0:.0f} m/yr  (initial guess)")
    print(f"  κ     = {kappa} km²/day     ε = {eps} /day")
    print()

    # ── 4.3  Solve pressure PDE ──────────────────────────────────────────────
    dx     = 0.8
    x      = np.arange(0.0, L + dx, dx)
    t_span = (0.0, 10.4)
    t_eval = np.arange(0.0, 10.41, 0.01)

    sol = solve_ivp(
        fun=lambda t, p: diffusion_rhs(t, p, dx, tt, Q - Q_mean, kappa, eps),
        t_span=t_span, y0=np.zeros(len(x)), t_eval=t_eval,
        method='RK45', rtol=1e-4, atol=1e-5,
    )
    t = sol.t
    p = sol.y.T    # (Nt, nx) — normalized pressure p'  (m³/s)

    # Dimensionless pressure perturbation at every node (same normalization as MATLAB plot)
    p_norm_full = p / (k_Q * sigma_0)    # (Nt, nx)
    p_norm_x0   = p_norm_full[:, 0]      # at x = 0  (moulin/GPS location)

    # ── 4.4  Zoet forward velocity with initial parameters ──────────────────
    u_zoet0, tau_c0_t, N_eff0 = zoet_velocity(
        tau_d, p_norm_x0, f_N0, sigma_0, phi0, u0_0)

    # ── 4.5  Generate synthetic GPS consistent with Zoet model ───────────────
    # We use the Zoet forward model (+ small noise) as synthetic GPS so that
    # the inversion has something meaningful to fit.
    # Note: the original Tsai 2021 synthetic GPS was designed for the Budd law
    # (tiny amplitude ≈ 4 m/yr swing), inconsistent with the large pressure
    # swings produced by this forcing.  For real data, replace v_GPS_zoet.
    rng = np.random.default_rng(42)
    tt_GPS     = tt.copy()
    p_norm_gps = interp1d(t, p_norm_x0, bounds_error=False,
                          fill_value=(p_norm_x0[0], p_norm_x0[-1]))(tt_GPS)
    u_zoet_gps, _, _ = zoet_velocity(tau_d, p_norm_gps, f_N0, sigma_0, phi0, u0_0)
    noise_std  = 5.0     # m/yr observational noise
    v_GPS      = u_zoet_gps + rng.normal(0, noise_std, len(tt_GPS))

    # Original Budd-type synthetic GPS kept for comparison
    v_GPS_budd = 149.0 / (1.0 - 0.05 * 0.1 * np.cos(2.0 * np.pi * tt_GPS))**3

    # ── 4.6  Invert Zoet parameters against GPS subset (index 84 onward) ────
    tk        = tt_GPS[84:]
    dk        = v_GPS[84:]
    pn_interp = interp1d(t, p_norm_x0, bounds_error=False,
                         fill_value=(p_norm_x0[0], p_norm_x0[-1]))
    p_norm_tk = pn_interp(tk)

    phi_fit, u0_fit, v_pred_fit = fit_zoet_parameters(
        dk, p_norm_tk, tau_d, sigma_0, f_N0,
        phi0=phi0, u0_0=u0_0, n_iter=40)

    # Best-fit Zoet velocity on full time axis
    u_zoet, tau_c_fit, N_eff_t = zoet_velocity(
        tau_d, p_norm_x0, f_N0, sigma_0, phi_fit, u0_fit)

    # Reference Budd-type velocity
    p_norm_budd = p[:, 0] / (k_Q * sigma_0)
    ub0_ref, ku_ref, mm_ref = 148.98, 0.049, 0.96
    u_budd = ub0_ref * (1.0 - ku_ref * p_norm_budd)**(-mm_ref)

    # ── 4.7  Flux field and pressure field ──────────────────────────────────
    dpdx  = (p[:, 1:] - p[:, :-1]) / dx    # normalized flux gradient (Nt, nx-1)
    x_mid = 0.5 * (x[1:] + x[:-1])

    T_grid,  X_grid  = np.meshgrid(t, x)
    Td_grid, Xd_grid = np.meshgrid(t, x_mid)

    # Full water pressure field normalized by overburden
    pss_2d     = sigma_0 * (1.0 - X_grid / L)
    p_norm_2d  = (pss_2d + p.T / k_Q) / sigma_0    # (nx, Nt)

    # ── 4.8  Print results ────────────────────────────────────────────────────
    print("  INVERSION RESULTS  (Zoet & Iverson 2020)")
    print(f"    φ_fit = {phi_fit:.2f}°       (best-fit till friction angle)")
    print(f"    u₀_fit = {u0_fit:.1f}  m/yr   (best-fit transition velocity)")
    print()
    print("  FORWARD MODEL  (best-fit parameters)")
    print(f"    Mean  u_zoet = {np.mean(u_zoet):.1f}  m/yr")
    print(f"    Min   u_zoet = {np.min(u_zoet):.1f}  m/yr")
    print(f"    Max   u_zoet = {np.max(u_zoet):.1f}  m/yr")
    print(f"    Mean  N_eff  = {np.mean(N_eff_t)/1e3:.1f}  kPa")
    print(f"    Mean  τ_c    = {np.mean(tau_c_fit)/1e3:.1f}  kPa")
    print(f"    τ_d/τ_c(mean)= {tau_d/np.mean(tau_c_fit):.3f}  (< 1 required)")
    print()

    # ── 4.9  Five-panel figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(11, 18), constrained_layout=True)

    fig.suptitle(
        "Transient subglacial hydrology  x  Zoet & Iverson (2020) Regularized Coulomb Sliding\n"
        "tau_b = tau_c * u/(u+u0),  tau_c = N*tan(phi),  N(t) = sigma0*(f_N0 - p_norm(t))",
        fontsize=11
    )



    # ── Panel 1: Input flux and pressure diffusion BC ────────────────────────
    ax = axes[0]
    ax.plot(tt, Q, lw=2, color='royalblue', label='Observed flux  Q(t)  (m³/s)')
    ax.axhline(Q_mean, color='royalblue', lw=1, ls=':', alpha=0.6,
               label=f'Q_mean = {Q_mean:.1f} m³/s')
    ax.set_xlim(t_span)
    ax.set_ylabel('Flux  (m³/s)')
    ax.set_title('Panel 1 — Meltwater flux forcing Q(t)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35)

    # ── Panel 2: Effective pressure & Coulomb limit at x = 0 ─────────────────
    ax = axes[1]
    ax.fill_between(t, N_eff_t / 1e3, alpha=0.15, color='steelblue')
    ax.plot(t, N_eff_t / 1e3,  lw=1.8, color='steelblue',
            label='Effective pressure  N(t)  (kPa)')
    ax.plot(t, tau_c_fit / 1e3, lw=1.8, color='darkorange',
            label=f'Coulomb limit  τ_c = N · tan({phi_fit:.1f}°)  (kPa)')
    ax.axhline(tau_d / 1e3, color='crimson', lw=1.5, ls='--',
               label=f'Driving stress  τ_d = {tau_d/1e3:.1f} kPa  (must stay < τ_c)')
    ax.axhline(f_N0 * sigma_0 / 1e3, color='gray', lw=1.0, ls=':',
               label=f'N₀ = {f_N0*sigma_0/1e3:.0f} kPa  (f_N0 = {f_N0})')
    ax.set_xlim(t_span)
    ax.set_ylabel('Stress  (kPa)')
    ax.set_title('Panel 2 — Effective pressure and Coulomb limit at x = 0  (moulin)')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.35)

    # ── Panel 3: Ice velocity — GPS vs Zoet vs Budd ───────────────────────────
    ax = axes[2]
    ax.plot(tt_GPS, v_GPS,       's',  ms=2.5, color='gray', alpha=0.6,
            label='Synthetic GPS  (Zoet-consistent + noise  5 m/yr)')
    ax.plot(t, u_zoet,          lw=2.2, color='royalblue',
            label=f'Zoet & Iverson (2020) — φ = {phi_fit:.1f}°,  u₀ = {u0_fit:.0f} m/yr')
    ax.plot(tk, v_pred_fit,     'o',  ms=4, color='navy', alpha=0.5, zorder=5,
            label='Zoet fit on GPS subset  (t ≥ t₈₄)')
    ax.plot(t, u_budd,          lw=1.6, color='darkorange', ls='--',
            label='Budd-type power law  (original Tsai 2021 synthetic)')
    ax.plot(tt_GPS, v_GPS_budd, 'x',  ms=2.5, color='darkorange', alpha=0.4,
            label='Original Budd-type GPS  (tiny amplitude)')
    ax.set_xlim(t_span)
    ax.set_ylabel('Velocity  (m/yr)')
    ax.set_title('Panel 3 — Basal sliding velocity: Zoet 2020 vs Budd-type (Tsai 2021)')
    ax.legend(fontsize=8.5, ncol=2); ax.grid(True, alpha=0.35)

    # ── Panel 4: Subglacial flux field Q(x, t) ────────────────────────────────
    ax = axes[3]
    cf = ax.contourf(Td_grid, Xd_grid, (Q_mean - dpdx).T, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Flux  (m³/s)', shrink=0.9)
    ax.set_ylabel('Distance  (km)')
    ax.set_title('Panel 4 — Subglacial flux field  Q(x, t)')

    # ── Panel 5: Normalized water pressure field p(x,t) / (ρ_i g H) ─────────
    ax = axes[4]
    cf = ax.contourf(T_grid, X_grid, p_norm_2d, levels=20, cmap='plasma')
    fig.colorbar(cf, ax=ax, label='p / (ρ_i g H)', shrink=0.9)
    ax.set_xlabel('Time  (days)')
    ax.set_ylabel('Distance  (km)')
    ax.set_title('Panel 5 — Normalized water pressure  p(x, t) / (ρ_i g H)')

    out_png = 'transient_water_pressure_zoet_sliding.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved → {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    transient_water_pressure_zoet()
