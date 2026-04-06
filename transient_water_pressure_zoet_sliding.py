"""
Transient subglacial water pressure + Zoet & Iverson (2020) sliding law.

Solves the pressure diffusion PDE (Tsai et al. 2021):
    dp'/dt = kappa * d^2p'/dx^2 - eps * p'

Replaces the Budd-type power law with the Regularized Coulomb law
(Zoet & Iverson 2020), keeping the SAME synthetic GPS data as the
original script (velocities ~ 146-151 m/yr).

    ORIGINAL  (Tsai 2021, Budd-type):
        u = ub0 * (1 - ku * pk)^(-m)
        params:  ub0 [m/yr],  ku [-],  m [-]

    NEW  (Zoet & Iverson 2020, Regularized Coulomb):
        tau_b = tau_c(t) * u / (u + u0)
        tau_c(t) = N(t) * tan(phi)
        N(t)     = N_ref * (1 - alpha * pk(t))   [Pa]
        N_ref    = tau_d * (1 + u0/u_mean) / tan(phi)
        params:  phi [deg],  u0 [m/yr],  alpha [-]

    Inverted for u given tau_b = tau_d (shallow-ice force balance):
        u = u0 * tau_d / (tau_c(t) - tau_d)

    Parameter correspondence with Budd law:
        u0    ~ ub0   : sets the mean velocity scale  (m/yr, IN GPS RANGE)
        alpha ~ ku    : pressure-coupling coefficient (dimensionless, << 1)
        phi   ~ m     : till friction angle (fixed at 26 deg from literature)

    N_ref is DERIVED analytically from (phi, u0) via the mean-velocity
    condition u(pk=0) = u_mean, so there are only 2 free inversion params.

    Why N(t) = N_ref*(1 - alpha*pk) rather than N_ref - pk*sigma ?
    ---------------------------------------------------------------
    The PDE pressure pk at x=0 has amplitude +-0.33 (33% of sigma).
    With realistic u0 ~ 200 m/yr, N_ref ~ 900 kPa ~ 10.8% of sigma.
    Direct subtraction N_ref - pk*sigma would give N < 0 whenever
    |pk| > 0.108.  The coupling factor alpha << 1 (here ~0.027) means
    that surface-melt pressure perturbations reach the bed attenuated
    by ~37x, consistent with distributed subglacial drainage storing
    most of the pressure signal before it affects sliding.

References
----------
Tsai VC, Smith LC, Gardner AS, Seroussi H (2021).
    Journal of Glaciology 68(268), 390-400.
Zoet LK, Iverson NR (2020).
    A slip law for glaciers on deformable beds.  Science 368, 76-78.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 1.  PRESSURE DIFFUSION ODE  (Tsai et al. 2021 — unchanged)
# ---------------------------------------------------------------------------

def diffusion_rhs(t, p, dx, tt_q, q_anomaly, kappa, eps):
    """
    RHS of the normalised pressure diffusion PDE.
        dp'/dt = kappa * d^2p'/dx^2 - eps * p'
    BCs: Neumann at x=0 (flux),  Dirichlet at x=L (atmospheric).
    """
    pss  = 0.0
    dpdt = np.zeros_like(p)
    q_now = float(interp1d(tt_q, q_anomaly,
                           bounds_error=False,
                           fill_value=(q_anomaly[0], q_anomaly[-1]))(t))
    dpdt[0]    = kappa * ((p[1]-p[0])/dx + q_now) / dx - eps*(p[0]-pss)
    dpdt[1:-1] = kappa/dx**2*(p[2:] - 2.*p[1:-1] + p[:-2]) - eps*(p[1:-1]-pss)
    dpdt[-1]   = 0.0
    return dpdt


# ---------------------------------------------------------------------------
# 2.  ZOET & IVERSON (2020) REGULARIZED COULOMB SLIDING LAW
# ---------------------------------------------------------------------------

def zoet_velocity(pk, phi_deg, u0, alpha, tau_d, sigma, u_mean):
    """
    Basal sliding velocity from the Regularized Coulomb law.

    Physics
    -------
    tau_b = tau_c * u / (u + u0)     [Regularized Coulomb, Eq. 1 in Z&I 2020]
    tau_c = N(t) * tan(phi)          [Coulomb limit]
    N(t)  = N_ref * (1 - alpha*pk)  [effective pressure, always positive]
    N_ref = tau_d*(1 + u0/u_mean)/tan(phi)  [derived from mean-velocity condition]

    Inverting for u given tau_b = tau_d (force balance):
        u = u0 * tau_d / (tau_c(t) - tau_d)

    Parameters
    ----------
    pk      : ndarray — dimensionless pressure  p'/(k_Q * sigma)
    phi_deg : float   — till friction angle  (degrees)
    u0      : float   — transition velocity  (m/yr)  — IN GPS RANGE
    alpha   : float   — pressure-coupling coefficient  (dimensionless, << 1)
    tau_d   : float   — driving stress  (Pa)
    sigma   : float   — ice overburden sigma = rho_i*g*H  (Pa)
    u_mean  : float   — target mean velocity  (m/yr)

    Returns
    -------
    u_b   : ndarray — basal sliding velocity  (m/yr)
    tau_c : ndarray — Coulomb shear-strength limit  tau_c  (Pa)
    N_eff : ndarray — effective pressure  N(t)  (Pa)
    N_ref : float   — background effective pressure  (Pa)
    """
    tan_phi = np.tan(np.radians(phi_deg))

    # Background effective pressure — derived analytically so that at pk=0,
    # u = u0 * tau_d / (N_ref*tan_phi - tau_d) = u_mean
    N_ref = tau_d * (1.0 + u0 / u_mean) / tan_phi

    # Time-varying effective pressure
    N_eff = N_ref * (1.0 - alpha * pk)
    N_eff = np.maximum(N_eff, 1e-3 * sigma)     # physical floor

    tau_c = np.maximum(N_eff * tan_phi, tau_d * 1.001)   # must exceed tau_d
    denom = np.maximum(tau_c - tau_d, 1e-3 * tau_c)

    u_b = np.maximum(u0 * tau_d / denom, 0.0)
    return u_b, tau_c, N_eff, N_ref


# ---------------------------------------------------------------------------
# 3.  GAUSS-NEWTON INVERSION FOR u0 AND alpha  (phi fixed from literature)
# ---------------------------------------------------------------------------

def fit_zoet_parameters(v_obs, pk_tk, phi_deg, tau_d, sigma, u_mean,
                        u0_0=200.0, alpha0=0.027, n_iter=40):
    """
    Fit Zoet parameters u0 and alpha to observed GPS velocities.

    phi is fixed at the literature value (26 deg for soft subglacial till,
    Zoet & Iverson 2020) because the synthetic GPS barely varies (~4 m/yr
    range) and cannot independently constrain phi, u0, and alpha together.

    Model:  u = u0 * tau_d / (N_ref*(1-alpha*pk)*tan(phi) - tau_d)

    Uses Gauss-Newton iterations with Levenberg-Marquardt damping,
    identical in structure to the original Tsai 2021 Budd inversion.

    Parameters
    ----------
    v_obs   : ndarray — observed velocities at fitting times  (m/yr)
    pk_tk   : ndarray — dimensionless pressure at fitting times
    phi_deg : float   — till friction angle (fixed, degrees)
    tau_d   : float   — driving stress  (Pa)
    sigma   : float   — overburden  (Pa)
    u_mean  : float   — mean GPS velocity  (m/yr)
    u0_0    : float   — initial guess for u0  (m/yr)
    alpha0  : float   — initial guess for alpha
    n_iter  : int     — number of iterations

    Returns
    -------
    u0_fit    : float   — best-fit transition velocity  (m/yr)
    alpha_fit : float   — best-fit coupling coefficient
    v_fit     : ndarray — predicted velocities at fitting times  (m/yr)
    """
    m = np.array([u0_0, alpha0], dtype=float)

    for _ in range(n_iter):
        v_pred, _, _, _ = zoet_velocity(pk_tk, phi_deg, m[0], m[1],
                                        tau_d, sigma, u_mean)
        # Numerical Jacobian (analytical is complex due to N_ref(u0) dependency)
        du, da = 1.0, 1e-4
        vp_u0, _, _, _ = zoet_velocity(pk_tk, phi_deg, m[0]+du, m[1],
                                       tau_d, sigma, u_mean)
        vp_al, _, _, _ = zoet_velocity(pk_tk, phi_deg, m[0], m[1]+da,
                                       tau_d, sigma, u_mean)
        G_hat = np.column_stack([(vp_u0 - v_pred)/du,
                                  (vp_al - v_pred)/da])

        Hess = G_hat.T @ G_hat
        dm   = np.linalg.solve(Hess + 0.1*np.diag(np.diag(Hess)),
                               G_hat.T @ (v_obs - v_pred))
        m    = m + dm
        m[0] = np.clip(m[0], 10.0, 500.0)   # u0 in [10, 500] m/yr
        m[1] = np.clip(m[1], 1e-6,  1.0)    # alpha in (0, 1]

    u0_fit, alpha_fit = float(m[0]), float(m[1])
    v_fit, _, _, _ = zoet_velocity(pk_tk, phi_deg, u0_fit, alpha_fit,
                                   tau_d, sigma, u_mean)
    return u0_fit, alpha_fit, v_fit


# ---------------------------------------------------------------------------
# 4.  MAIN ROUTINE
# ---------------------------------------------------------------------------

def transient_water_pressure_zoet():
    """
    Full pipeline — mirrors the 4-panel layout of the original Tsai 2021 script.
    Sliding law replaced: Budd power law -> Zoet & Iverson (2020) Regularized Coulomb.
    GPS data and PDE are IDENTICAL to the original.
    """

    # -----------------------------------------------------------------------
    # 4.1  Synthetic data  (* Replace with real data as desired *)
    #      IDENTICAL to original Tsai 2021 MATLAB script
    # -----------------------------------------------------------------------
    tt    = np.arange(0.0, 10.45, 0.05)
    Q     = 16.0 + 8.0 * np.cos(2.0*np.pi*tt + 0.6)         # flux (m^3/s)
    ttGPS = tt.copy()
    vGPS  = 149.0 / (1.0 - 0.05*0.1*np.cos(2.0*np.pi*ttGPS + 0.0))**3  # (m/yr)

    # -----------------------------------------------------------------------
    # 4.2  Physical parameters  (* Replace with modifications as desired *)
    #      IDENTICAL to original Tsai 2021 MATLAB script
    # -----------------------------------------------------------------------
    L    = 42.0;   rhoi = 920.0;   g = 9.8;   H = 934.0
    Q_mean = np.mean(Q)
    k_Q    = 0.5 * L * Q_mean / (rhoi*g*H)
    v_mean = np.mean(vGPS)
    kappa  = 1400.0    # hydraulic diffusivity (km^2/day)
    eps    =    4.0    # scaled inverse viscosity (1/day)

    # Derived
    sigma = rhoi * g * H                  # overburden  (Pa)
    alpha_slope = H / (L * 1e3)           # surface slope
    tau_d = rhoi * g * H * alpha_slope    # driving stress  (Pa) ~ 187 kPa

    # -----------------------------------------------------------------------
    # 4.3  Zoet & Iverson (2020) parameters
    # -----------------------------------------------------------------------
    # phi = 26 deg: till friction angle, fixed from Zoet & Iverson (2020)
    # laboratory experiments on soft subglacial till (range 20-32 deg).
    phi_deg = 26.0

    # Initial guesses for inversion:
    # u0_0 ~ 200 m/yr  (in GPS range, physically meaningful)
    # alpha0 ~ 0.027   (pressure coupling, calibrated to match Budd sensitivity)
    u0_0    = 200.0
    alpha0  = 0.027

    print("=" * 65)
    print("  Tsai et al. (2021) hydrology  x  Zoet & Iverson (2020) sliding")
    print("=" * 65)
    print(f"  sigma = {sigma/1e6:.3f} MPa   tau_d = {tau_d/1e3:.2f} kPa")
    print(f"  phi   = {phi_deg} deg  (fixed, Zoet & Iverson 2020)")
    print(f"  u0_0  = {u0_0} m/yr   alpha0 = {alpha0}")
    print(f"  kappa = {kappa} km^2/day   eps = {eps} /day")
    print()

    # -----------------------------------------------------------------------
    # 4.4  Solve pressure PDE  (identical to original)
    # -----------------------------------------------------------------------
    dx     = 0.8
    x      = np.arange(0.0, L+dx, dx)
    tspan  = (0.0, 10.4)
    t_eval = np.arange(0.0, 10.41, 0.01)

    sol = solve_ivp(
        fun=lambda t, p: diffusion_rhs(t, p, dx, tt, Q-Q_mean, kappa, eps),
        t_span=tspan, y0=np.zeros(len(x)), t_eval=t_eval,
        method='RK45', rtol=1e-4, atol=1e-5,
    )
    t = sol.t
    p = sol.y.T    # (Nt, nx) — normalised pressure p'  (m^3/s)

    # Dimensionless pressure perturbation  (same normalisation as original MATLAB)
    pk_full = p / (k_Q * sigma)       # (Nt, nx)
    pk_x0   = pk_full[:, 0]           # at x=0 (GPS / moulin location)

    # -----------------------------------------------------------------------
    # 4.5  Invert Zoet parameters against GPS subset  (index 84 onward)
    #      SAME subset as original Tsai 2021 script
    # -----------------------------------------------------------------------
    tk   = ttGPS[84:]
    dk   = vGPS[84:]
    pk_tk = interp1d(t, pk_x0, bounds_error=False,
                     fill_value=(pk_x0[0], pk_x0[-1]))(tk)

    u0_fit, alpha_fit, v_pred = fit_zoet_parameters(
        dk, pk_tk, phi_deg, tau_d, sigma, v_mean,
        u0_0=u0_0, alpha0=alpha0, n_iter=40)

    # Best-fit forward model on full time axis
    u_zoet, tau_c_t, N_eff_t, N_ref = zoet_velocity(
        pk_x0, phi_deg, u0_fit, alpha_fit, tau_d, sigma, v_mean)

    # Original Budd-type prediction (reference)
    ub0_ref, ku_ref, mm_ref = 148.98, 0.049, 0.96
    u_budd = ub0_ref * (1.0 - ku_ref * pk_x0)**(-mm_ref)

    # -----------------------------------------------------------------------
    # 4.6  Flux / pressure fields for contour panels
    # -----------------------------------------------------------------------
    dpdx   = (p[:, 1:] - p[:, :-1]) / dx
    x_mid  = 0.5 * (x[1:] + x[:-1])
    T_grid,  X_grid  = np.meshgrid(t, x)
    Td_grid, Xd_grid = np.meshgrid(t, x_mid)
    pss_2d    = rhoi*g*H * (1.0 - X_grid/L)
    p_norm_2d = (pss_2d + p.T / k_Q) / (rhoi*g*H)   # (nx, Nt)

    # -----------------------------------------------------------------------
    # 4.7  Print results
    # -----------------------------------------------------------------------
    rmse_zoet = np.sqrt(np.mean((v_pred - dk)**2))
    u_budd_tk = ub0_ref * (1.0 - ku_ref*pk_tk)**(-mm_ref)
    rmse_budd = np.sqrt(np.mean((u_budd_tk - dk)**2))

    print("  INVERSION RESULTS  (Zoet & Iverson 2020):")
    print(f"    phi   = {phi_deg:.2f} deg   (fixed from literature)")
    print(f"    u0    = {u0_fit:.2f} m/yr  (transition velocity)")
    print(f"    alpha = {alpha_fit:.5f}    (pressure-coupling coefficient)")
    print(f"    N_ref = {N_ref/1e3:.1f} kPa  ({N_ref/sigma*100:.1f}% of sigma)")
    print()
    print("  COMPARISON (at GPS subset):")
    print(f"    RMSE Zoet = {rmse_zoet:.4f} m/yr")
    print(f"    RMSE Budd = {rmse_budd:.4f} m/yr")
    print()
    print("  FORWARD MODEL at x=0:")
    print(f"    u_zoet: {u_zoet.min():.2f} – {u_zoet.max():.2f} m/yr"
          f"  mean={u_zoet.mean():.2f}")
    print(f"    N_eff:  {N_eff_t.min()/1e3:.1f} – {N_eff_t.max()/1e3:.1f} kPa")
    print(f"    tau_c:  {tau_c_t.min()/1e3:.1f} – {tau_c_t.max()/1e3:.1f} kPa"
          f"  (tau_d={tau_d/1e3:.1f} kPa)")
    print()
    print("  The best fitting Zoet sliding law parameters are:")
    print(f"  u0 = {u0_fit:.2f},  alpha = {alpha_fit:.5f},  phi = {phi_deg:.2f}")

    # -----------------------------------------------------------------------
    # 4.8  Four-panel figure  (mirrors original MATLAB layout)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    fig.suptitle(
        "Zoet & Iverson (2020) Regularized Coulomb Sliding\n"
        "tau_b = tau_c(t)*u/(u+u0),  "
        "tau_c = N(t)*tan(phi),  "
        "N(t) = N_ref*(1 - alpha*pk)",
        fontsize=11
    )

    # Panel 1 — Flux Q(0,t) and Q(L,t)
    ax = axes[0]
    ax.plot(t, Q_mean - dpdx[:, 0],  lw=1.8, label='Q(x=0)  inlet')
    ax.plot(t, Q_mean - dpdx[:, -1], lw=1.8, label='Q(x=L)  outlet')
    ax.set_xlim(tspan); ax.grid(True, alpha=0.35)
    ax.set_title('Flux  (m^3/s)'); ax.legend(fontsize=9)

    # Panel 2 — Ice velocity
    ax = axes[1]
    ax.plot(ttGPS, vGPS,   's', ms=4,  color='gray',
            label='GPS data')
    ax.plot(tk,    v_pred, '-', lw=2.5, color='royalblue',
            label=f'Zoet fit:  u0={u0_fit:.1f} m/yr,  '
                  f'alpha={alpha_fit:.4f},  phi={phi_deg:.0f}deg')
    ax.plot(t,     u_budd, '--', lw=1.6, color='darkorange',
            label=f'Budd fit:  ub0={ub0_ref},  ku={ku_ref},  m={mm_ref}')
    ax.set_xlim(tspan); ax.grid(True, alpha=0.35)
    ax.set_title('Ice Velocity  (m/yr)'); ax.legend(fontsize=9)

    # Panel 3 — Flux field Q(x,t)
    ax = axes[2]
    cf = ax.contourf(Td_grid, Xd_grid, (Q_mean - dpdx).T, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='m^3/s', shrink=0.9)
    ax.set_ylabel('Dist  (km)'); ax.set_title('Flux  (m^3/s)')

    # Panel 4 — Pressure field p(x,t)/(rho_i g H)
    ax = axes[3]
    cf = ax.contourf(T_grid, X_grid, p_norm_2d, levels=20, cmap='plasma')
    fig.colorbar(cf, ax=ax, label='p / (rho_i g H)', shrink=0.9)
    ax.set_xlabel('Time  (days)'); ax.set_ylabel('Dist  (km)')
    ax.set_title('Pressure  (rho_i g H)')

    plt.tight_layout()
    out_png = 'transient_water_pressure_zoet_sliding.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved -> {out_png}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    transient_water_pressure_zoet()
