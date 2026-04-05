"""
Transient subglacial water pressure and basal sliding model.

Translated from MATLAB to Python.

Uses known input water flux at one location to solve for predicted subglacial
water pressure and predicted glacier sliding speed using the model described in:

    Tsai VC, Smith LC, Gardner AS, and Seroussi H (2021).
    "A unified model for transient subglacial water pressure and basal sliding",
    Journal of Glaciology.

Original MATLAB code by Victor C. Tsai, (c) 2021, contact: victor_tsai@brown.edu

Numerically solves:
    dp/dt = kappa * d²p/dx² - eps * (p - pss)

Boundary conditions (BC):
    BC1: dp/dx(0, t) = g0(t)        [known input flux at x=0]
    BC2: p(L, t) = p0               [atmospheric pressure at terminus]

Initial condition (IC):
    p(x, 0) = pss                   [start at steady-state]

g0 is scaled flux, p0 is atm, pss is linear (hydrostatic).
Output for p is used to solve for best-fitting sliding law coefficients:
    u_predicted = ub0 * (1 - ku * p(tk))^(-mm)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# ODE right-hand side (equivalent to MATLAB nested function diffusion_data)
# ---------------------------------------------------------------------------

def diffusion_data(t, p, dx, tt_combo, Q_avg_combo, kappa, eps):
    """
    RHS of the normalized pressure diffusion PDE:
        dp/dt = kappa * d²p/dx² - eps * (p - pss)

    Boundary conditions:
        - Left  (x=0): Neumann — flux prescribed via interpolated Q anomaly
        - Right (x=L): Dirichlet — dpdt = 0 (pressure fixed at atmospheric)

    Parameters
    ----------
    t          : float  — current time (days)
    p          : array  — pressure perturbation at all grid nodes
    dx         : float  — spatial step (km)
    tt_combo   : array  — time vector for Q interpolation
    Q_avg_combo: array  — Q anomaly (Q - Q_mean) time series
    kappa      : float  — hydraulic diffusivity (km²/day)
    eps        : float  — scaled inverse viscosity (1/day)

    Returns
    -------
    dpdt : array — time derivative of p at every grid node
    """
    pss = 0.0
    n = len(p)
    dpdt = np.zeros(n)

    # Interpolate Q anomaly at current time t
    Q_interp = interp1d(tt_combo, Q_avg_combo,
                        bounds_error=False,
                        fill_value=(Q_avg_combo[0], Q_avg_combo[-1]))
    Q_now = float(Q_interp(t))

    # Left boundary: Neumann BC (prescribed flux)
    dpdt[0] = kappa * ((p[1] - p[0]) / dx + Q_now) / dx - eps * (p[0] - pss)

    # Interior nodes: standard second-order finite difference
    dpdt[1:-1] = (kappa / dx**2 * (p[2:] - 2.0 * p[1:-1] + p[:-2])
                  - eps * (p[1:-1] - pss))

    # Right boundary: Dirichlet BC — pressure is fixed (dpdt = 0)
    dpdt[-1] = 0.0

    return dpdt


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def transient_water_pressure_sliding():
    """
    Main routine: solve PDE, invert for sliding-law parameters, plot results.
    """

    # ------------------------------------------------------------------
    # Synthetic input data  (* Replace with real data as desired *)
    # ------------------------------------------------------------------
    tt = np.arange(0, 10.45, 0.05)           # model times for Q  (days)
    Q = 16 + 8 * np.cos(2 * np.pi * tt + 0.6)   # observed flux  (m³/s)

    tt_GPS = tt.copy()                        # GPS sample times  (days)
    v_GPS = 149.0 / (1 - 0.05 * 0.1 * np.cos(2 * np.pi * tt_GPS + 0.0))**3
    # GPS velocities  (m/year)

    # ------------------------------------------------------------------
    # Physical parameters  (* Modify as desired *)
    # ------------------------------------------------------------------
    L    = 42.0    # domain length          (km)
    rhoi = 920.0   # ice density            (kg/m³)
    g    = 9.8     # gravitational accel.   (m/s²)
    H    = 934.0   # ice thickness          (m)

    Q_mean = np.mean(Q)                              # mean flux  (m³/s)
    k_Q    = 0.5 * L * Q_mean / (rhoi * g * H)      # scaled conductivity × area
    v_mean = np.mean(v_GPS)                          # mean GPS velocity  (m/year)
    kappa  = 1400.0   # hydraulic diffusivity  (km²/day)
    eps    = 4.0      # scaled inverse viscosity  (1/day)

    print(f"Q_mean  = {Q_mean:.4f}  m³/s")
    print(f"k_Q     = {k_Q:.6e}  (km·m³/s / (kg·m/s²·m))")
    print(f"v_mean  = {v_mean:.4f}  m/year")

    # ------------------------------------------------------------------
    # Solve normalized PDE  (p' where p = pss + p'/k_Q)
    # ------------------------------------------------------------------
    dx    = 0.8                                      # spatial step  (km)
    x     = np.arange(0, L + dx, dx)                # spatial grid
    t_span = (0.0, 10.4)
    t_eval = np.arange(0, 10.41, 0.01)              # output times  (days)
    p_IC  = np.zeros(len(x))                        # zero-perturbation IC

    sol = solve_ivp(
        fun=lambda t, p: diffusion_data(t, p, dx, tt, Q - Q_mean, kappa, eps),
        t_span=t_span,
        y0=p_IC,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-4,
        atol=1e-5,
        dense_output=False,
    )

    t = sol.t                   # shape (Nt,)
    # solve_ivp returns y with shape (n_states, Nt); transpose to (Nt, nx)
    p = sol.y.T                 # shape (Nt, nx)

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    # Normalised pressure gradient  (m³/s)  — shape (Nt, nx-1)
    dpdx = (p[:, 1:] - p[:, :-1]) / dx

    # Mid-point x coordinates for dpdx
    x_mid = 0.5 * (x[1:] + x[:-1])

    # ------------------------------------------------------------------
    # Inversion for best-fit sliding-law coefficients
    # Sliding law:  u_pred = ub0 * (1 - ku * p_norm)^(-mm)
    # Parameters:   m_vec = [ub0, ku, mm]
    # ------------------------------------------------------------------
    # Subset of GPS points used for fitting (index 84 onward, 0-based)
    tk = tt_GPS[84:]
    dk = v_GPS[84:]

    # Normalised pressure at x=0 interpolated at GPS times
    p_norm_t = p[:, 0] / (k_Q * rhoi * g * H)
    pk_interp = interp1d(t, p_norm_t,
                         bounds_error=False,
                         fill_value=(p_norm_t[0], p_norm_t[-1]))
    pk = pk_interp(tk)

    # Gauss–Newton iterations (20 steps, Levenberg–Marquardt damping = 0.1)
    m_vec = np.array([150.0, 0.15, 3.0])
    for _ in range(20):
        ub0, ku, mm = m_vec
        base   = 1.0 - ku * pk
        v_pred = ub0 * base**(-mm)

        # Jacobian  G_hat  (shape: len(tk) × 3)
        G_hat = np.column_stack([
            base**(-mm),                        # ∂v/∂ub0
            v_pred * mm * pk / base,            # ∂v/∂ku
            -v_pred * np.log(base),             # ∂v/∂mm
        ])

        Hess  = G_hat.T @ G_hat
        lm    = 0.1 * np.diag(np.diag(Hess))   # Levenberg–Marquardt damping
        resid = dk - v_pred
        dm    = np.linalg.solve(Hess + lm, G_hat.T @ resid)
        m_vec = m_vec + dm

    ub0_fit, ku_fit, mm_fit = m_vec
    v_pred_fit = ub0_fit * (1.0 - ku_fit * pk)**(-mm_fit)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\nThe best fitting sliding law parameters are:")
    print(f"  ub0 = {ub0_fit:.2f}   (m/year)")
    print(f"  ku  = {ku_fit:.3f}")
    print(f"  m   = {mm_fit:.2f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # Meshgrids for contour plots — note: axes are (time, space)
    T_grid, X_grid   = np.meshgrid(t, x)          # (nx, Nt)
    Td_grid, Xd_grid = np.meshgrid(t, x_mid)      # (nx-1, Nt)

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    fig.suptitle("Transient subglacial water pressure & basal sliding\n"
                 "(Tsai et al., 2021)", fontsize=13)

    # --- Panel 1: Flux at inlet and outlet ---
    ax = axes[0]
    ax.plot(t, Q_mean - dpdx[:, 0],  label='Q(x=0) — inlet')
    ax.plot(t, Q_mean - dpdx[:, -1], label='Q(x=L) — outlet')
    ax.set_xlim(t_span)
    ax.set_ylabel('Flux (m³/s)')
    ax.set_title('Flux (m³/s)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Panel 2: GPS velocity vs. model velocity ---
    ax = axes[1]
    ax.plot(tt_GPS, v_GPS,      's', ms=3, label='GPS data')
    ax.plot(tk,     v_pred_fit, '-',       label='Model fit')
    ax.set_xlim(t_span)
    ax.set_ylabel('Velocity (m/year)')
    ax.set_title('Ice Velocity (m/year)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Panel 3: Flux field (x, t) ---
    ax = axes[2]
    cf = ax.contourf(Td_grid, Xd_grid, (Q_mean - dpdx).T, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='m³/s')
    ax.set_ylabel('Dist (km)')
    ax.set_title('Flux (m³/s)')

    # --- Panel 4: Normalised pressure field (x, t) ---
    ax = axes[3]
    pss_field = rhoi * g * H * (1.0 - X_grid / L)        # hydrostatic pss (Pa)
    p_norm_field = (pss_field + p.T / k_Q) / (rhoi * g * H)
    cf = ax.contourf(T_grid, X_grid, p_norm_field, levels=20, cmap='plasma')
    fig.colorbar(cf, ax=ax, label='p / (ρ_i g H)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Dist (km)')
    ax.set_title('Pressure (ρ_i g H)')

    plt.tight_layout()
    plt.savefig('transient_water_pressure_sliding.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved to transient_water_pressure_sliding.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    transient_water_pressure_sliding()
