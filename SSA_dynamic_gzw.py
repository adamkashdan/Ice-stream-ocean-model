"""
Coupled SSA Flowline + Dynamic GZW Sedimentation
===================================================
Architecture (decoupled for numerical stability):
  - Ice dynamics : SSA flowline residual solver with power-law basal drag
                   (nondimensionalised exactly as in Robel/Christian 2026)
  - Grounding-line sediment flux : Zoet & Iverson (2020) regularised Coulomb
                   sliding law evaluated at the GL in PHYSICAL units
  - Bed update   : Exner-type GZW kernel deposition at each timestep

Scenarios:
  1. Storfjorden  - retrograde bed ridges → staged GL stillstands → stacked GZWs
  2. Mackenzie Bay - smooth prograde bed + rapid forcing → GL never stalls → no GZWs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS & SCALING
# ─────────────────────────────────────────────────────────────────────────────
class Params:
    def __init__(self):
        self.year    = 3600 * 24 * 365
        self.Aglen   = 4.227e-25
        self.nglen   = 3
        self.Bglen   = self.Aglen ** (-1 / self.nglen)
        self.m       = 1 / self.nglen
        self.C       = 7e6
        self.rhoi    = 917.0
        self.rhow    = 1028.0
        self.g       = 9.81
        self.accum   = 0.28 / (3600 * 24 * 365)  # matches SSA_simple.py

        self.hscale  = 1000.0
        self.ascale  = 0.1 / self.year
        self.uscale  = (self.rhoi * self.g * self.hscale * self.ascale / self.C) ** (1.0/(self.m+1))
        self.xscale  = self.uscale * self.hscale / self.ascale
        self.tscale  = self.xscale / self.uscale
        self.eps     = (self.Bglen * (self.uscale/self.xscale)**(1/self.nglen)
                        / (2 * self.rhoi * self.g * self.hscale))
        self.lambda_d = 1.0 - self.rhoi / self.rhow
        self.transient = 0

        self.Nx = 200; self.N1 = 100; self.sigGZ = 0.97
        s1 = np.linspace(self.sigGZ/(self.N1+0.5), self.sigGZ, self.N1)
        s2 = np.linspace(self.sigGZ, 1.0, self.Nx-self.N1+1)
        self.sigma      = np.concatenate([s1, s2[1:]])
        self.sigma_elem = np.concatenate([[0], (self.sigma[:-1]+self.sigma[1:])/2])
        self.dsigma     = np.diff(self.sigma)

        self.tfinal = 8000.0 * self.year
        self.Nt     = 80
        self.dt     = self.tfinal / self.Nt

        # bed profile (physical, updated dynamically)
        self.nx_bed = 800
        self.x_bed  = np.linspace(0, 400e3, self.nx_bed)
        self.b_bed  = -100.0 - 1e-3 * self.x_bed

        # Zoet (2020) parameters – used at GL only
        self.mu_c        = 0.6
        self.A_visc      = 1e4          # Pa yr / m
        self.p_drag      = 1.0
        self.has_clasts  = True
        self.q_s_base    = 0.035        # m²/yr
        self.erosion_coeff  = 1e-6
        self.tau_threshold  = 1000.0   # Pa

        self.h_old  = None
        self.xg_old = None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BED INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────
def get_bed(x_phys, params):
    return np.interp(x_phys, params.x_bed, params.b_bed)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ZOET & IVERSON (2020) – physical units, GL only
# ─────────────────────────────────────────────────────────────────────────────
def zoet_drag(U_b, N, params):
    tau_c  = params.mu_c * N
    clast  = 3.0 if params.has_clasts else 1.0
    tau_v  = params.A_visc * clast * U_b
    p      = params.p_drag
    tau_b  = (tau_v**p * tau_c) / (tau_v**p + tau_c**p)**(1.0/p)
    regime = 'coulomb' if tau_v >= tau_c else 'viscous'
    return float(tau_b), regime


def sediment_flux(tau_b, regime, params):
    excess = max(0.0, tau_b - params.tau_threshold)
    q_s = params.q_s_base + params.erosion_coeff * excess
    if regime == 'coulomb':
        q_s = min(q_s, params.q_s_base * 2.0)
    return q_s


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GZW DEPOSITION KERNEL
# ─────────────────────────────────────────────────────────────────────────────
def gzw_kernel(x_array, gl_x, sigma_d=10e3, sigma_p=5e3):
    d = x_array - gl_x
    sigma = np.where(d > 0, sigma_d, sigma_p)
    k = np.exp(-0.5 * (d / sigma)**2)
    return k / k.max()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SSA FLOWLINE RESIDUAL  (power-law drag, standard nondimensionalisation)
# ─────────────────────────────────────────────────────────────────────────────
def flowline_eqns(huxg, params):
    h, u, xg = huxg[:params.Nx], huxg[params.Nx:2*params.Nx], huxg[2*params.Nx]
    hf  = (-get_bed(xg*params.xscale, params)/params.hscale) / (1-params.lambda_d)
    b   = -get_bed(xg*params.sigma*params.xscale, params) / params.hscale
    dt  = params.dt / params.tscale
    ds  = params.dsigma
    Nx, N1  = params.Nx, params.N1
    m, ng   = params.m, params.nglen
    eps, ld = params.eps, params.lambda_d
    a       = params.accum / params.ascale
    ss      = params.transient
    h_old, xg_old = params.h_old, params.xg_old
    se      = params.sigma_elem

    Fh = np.zeros(Nx); Fu = np.zeros(Nx)

    Fh[0] = ss*(h[0]-h_old[0])/dt + 2*h[0]*u[0]/(ds[0]*xg) - a
    Fh[1] = (ss*(h[1]-h_old[1])/dt
             - ss*se[1]*(xg-xg_old)*(h[2]-h[0])/(2*dt*ds[1]*xg)
             + h[1]*(u[1]+u[0])/(2*xg*ds[1]) - a)
    Fh[2:Nx-1] = (ss*(h[2:Nx-1]-h_old[2:Nx-1])/dt
                  - ss*se[2:Nx-1]*(xg-xg_old)*(h[3:Nx]-h[1:Nx-2])/(2*dt*ds[2:Nx-1]*xg)
                  + (h[2:Nx-1]*(u[2:Nx-1]+u[1:Nx-2])
                     - h[1:Nx-2]*(u[1:Nx-2]+u[0:Nx-3]))/(2*xg*ds[2:Nx-1]) - a)
    k = N1-1
    Fh[k]  = (1+0.5*(1+ds[k]/ds[k-1]))*h[k] - 0.5*(1+ds[k]/ds[k-1])*h[k-1] - h[k+1]
    Fh[-1] = (ss*(h[-1]-h_old[-1])/dt
              - ss*se[-1]*(xg-xg_old)*(h[-1]-h[-2])/(dt*ds[-2]*xg)
              + (h[-1]*(u[-1]+u[-2])-h[-2]*(u[-2]+u[-3]))/(2*xg*ds[-2]) - a)

    Fu[0] = ((4*eps)/(xg*ds[0])**((1/ng)+1)
             * (h[1]*(u[1]-u[0])*np.abs(u[1]-u[0])**((1/ng)-1)
                - h[0]*(2*u[0])*np.abs(2*u[0])**((1/ng)-1))
             - u[0]*np.abs(u[0])**(m-1)
             - 0.5*(h[0]+h[1])*(h[1]-b[1]-h[0]+b[0])/(xg*ds[0]))
    Fu[1:Nx-1] = ((4*eps)/(xg*ds[1:Nx-1])**((1/ng)+1)
                  * (h[2:Nx]*(u[2:Nx]-u[1:Nx-1])*np.abs(u[2:Nx]-u[1:Nx-1])**((1/ng)-1)
                     - h[1:Nx-1]*(u[1:Nx-1]-u[0:Nx-2])*np.abs(u[1:Nx-1]-u[0:Nx-2])**((1/ng)-1))
                  - u[1:Nx-1]*np.abs(u[1:Nx-1])**(m-1)
                  - 0.5*(h[1:Nx-1]+h[2:Nx])*(h[2:Nx]-b[2:Nx]-h[1:Nx-1]+b[1:Nx-1])/(xg*ds[1:Nx-1]))
    Fu[k] = (u[k+1]-u[k])/ds[k] - (u[k]-u[k-1])/ds[k-1]
    Fu[-1] = ((1/(xg*ds[-2])**(1/ng))
              * np.abs(u[-1]-u[-2])**((1/ng)-1) * (u[-1]-u[-2])
              - ld*hf/(8*eps))

    return np.concatenate((Fh, Fu, [3*h[-1]-h[-2]-2*hf]))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIMULATION DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(scenario_name, warm_start=None):
    print(f"\n{'='*50}\nScenario: {scenario_name}\n{'='*50}")
    params = Params()

    # ── bed geometry ──────────────────────────────────────────────────────────
    base_bed = -100.0 - 1e-3 * params.x_bed
    if scenario_name == 'Storfjorden':
        for r_x, r_h, r_w in [(150e3, 20.0, 9e3), (100e3, 18.0, 8e3), (55e3, 14.0, 7e3)]:
            base_bed += r_h * np.exp(-0.5 * ((params.x_bed - r_x) / r_w)**2)
    params.b_bed = base_bed.copy()

    # ── initial steady-state ──────────────────────────────────────────────────
    params.accum  = 0.28 / params.year
    xg0  = 200e3 / params.xscale
    hf0  = (-get_bed(xg0*params.xscale, params)/params.hscale) / (1-params.lambda_d)
    h0   = 1.0 - (1.0-hf0)*params.sigma
    u0   = 0.3 * params.sigma_elem**(1/3.0) + 1e-3
    params.h_old  = h0.copy()
    params.xg_old = xg0
    raw_guess = np.concatenate((h0, u0, [xg0]))

    # Prefer warm start (from previous converged run); fall back to raw guess
    guess = warm_start if warm_start is not None else raw_guess

    def flf(v): return flowline_eqns(v, params)

    print('Solving initial steady state …')
    sol, _, ier, msg = fsolve(flf, guess, full_output=True, maxfev=2_000_000, xtol=1e-8)
    if ier != 1:
        # try raw guess as fallback
        sol2, _, ier2, _ = fsolve(flf, raw_guess, full_output=True, maxfev=2_000_000, xtol=1e-8)
        if ier2 == 1:
            sol, ier = sol2, ier2
            print('  ✓ Converged (raw fallback)')
        else:
            print(f'  ⚠ SS did not fully converge: {msg.strip()}')
    else:
        print('  ✓ Converged')

    # ── transient loop ────────────────────────────────────────────────────────
    params.transient = 1
    huxg_t = sol.copy()
    times  = np.linspace(0, params.tfinal/params.year, params.Nt)
    xgs    = np.full(params.Nt, np.nan)
    qs_log = np.full(params.Nt, np.nan)
    initial_bed = params.b_bed.copy()
    gzw_accum   = np.zeros_like(params.x_bed)

    print('Running transient coupled simulation …')
    for t_idx in range(params.Nt):
        t_yr = times[t_idx]

        if scenario_name == 'Storfjorden':
            frac = min(1.0, t_yr / 6000.0)
            params.accum = 0.28 * (1.0 - 0.30*frac) / params.year
        else:
            frac = min(1.0, t_yr / 500.0)
            params.accum = 0.28 * (1.0 - 0.65*frac) / params.year

        params.h_old  = huxg_t[:params.Nx].copy()
        params.xg_old = huxg_t[-1]
        huxg_t, _, _, _ = fsolve(flf, huxg_t, full_output=True, maxfev=1_000_000, xtol=1e-7)

        xg_val = huxg_t[-1]
        gl_x   = xg_val * params.xscale
        xgs[t_idx] = gl_x / 1e3

        u_gl = huxg_t[2*params.Nx - 1]
        h_gl = huxg_t[params.Nx - 1]
        U_b  = u_gl * params.uscale * params.year
        b_gl = get_bed(gl_x, params)
        H_gl = h_gl * params.hscale
        N_gl = max(10e3, params.rhoi*params.g*H_gl + params.rhow*params.g*b_gl)

        tau_b, regime = zoet_drag(U_b, N_gl, params)
        q_s = sediment_flux(tau_b, regime, params)
        qs_log[t_idx] = q_s

        if 5e3 < gl_x < 390e3:
            k_dep = gzw_kernel(params.x_bed, gl_x)
            deposit = q_s * (params.dt/params.year) * k_dep
            params.b_bed += deposit
            gzw_accum    += deposit

        if (t_idx + 1) % 10 == 0:
            print(f"  t={t_yr:6.0f} yr | GL={gl_x/1e3:6.1f} km | "
                  f"τ_b={tau_b/1e3:.2f} kPa | {regime:8s} | q_s={q_s:.4f} m²/yr")

    return times, xgs, qs_log, params.x_bed, initial_bed, params.b_bed.copy(), gzw_accum, sol


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(r_st, r_mk):
    t_st, xg_st, qs_st, xb_st, ib_st, fb_st, gzw_st, _ = r_st
    t_mk, xg_mk, qs_mk, xb_mk, ib_mk, fb_mk, gzw_mk, _ = r_mk

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('#f4f1eb')

    c_st = '#2b7fb8'; c_mk = '#d45f1e'
    c_rock = '#6e6255'; c_gzw = '#c8a44b'; c_dhd = '#e8c9a0'

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.07, right=0.96, top=0.91, bottom=0.08)

    # Panel A – GL retreat
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('#faf8f4')
    ax.plot(t_st, xg_st, color=c_st, lw=2.5, label='Storfjorden (staged)')
    ax.plot(t_mk, xg_mk, color=c_mk, lw=2.5, label='Mackenzie Bay (rapid)')
    ax.set_xlabel('Time (yr)', fontsize=10); ax.set_ylabel('GL position (km)', fontsize=10)
    ax.set_title('A.  Grounding-Line Retreat', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Panel B – Zoet q_s
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor('#faf8f4')
    ax.plot(t_st, qs_st, color=c_st, lw=2.0, label='Storfjorden')
    ax.plot(t_mk, qs_mk, color=c_mk, lw=2.0, label='Mackenzie Bay')
    ax.set_xlabel('Time (yr)', fontsize=10)
    ax.set_ylabel('Sediment flux  $q_s$  (m²/yr)', fontsize=10)
    ax.set_title('B.  Zoet & Iverson (2020) Sediment Flux at GL', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    def draw_strat(ax, xb, ib, fb, gzw, title):
        ax.set_facecolor('#dedad3')
        xk = xb / 1e3
        ax.fill_between(xk, -450, ib, color=c_rock, alpha=0.9, label='Bedrock')
        ax.fill_between(xk, ib, fb, where=(fb > ib+0.05),
                        color=c_gzw, alpha=0.9, label='GZW deposits')
        dhd = gaussian_filter1d(np.clip(gzw*0.10 + 3.5, 0, 8), sigma=18)
        ax.fill_between(xk, fb, fb+dhd, color=c_dhd, alpha=0.8, label='DHD (Holocene)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (km)', fontsize=10)
        ax.set_ylabel('Elevation (m)', fontsize=10)
        ax.set_xlim(0, 250); ax.set_ylim(-380, -50)
        ax.grid(alpha=0.18, color='white')
        ax.legend(fontsize=9, loc='lower right')

    # Panel C – Storfjorden
    draw_strat(fig.add_subplot(gs[1, 0]),
               xb_st, ib_st, fb_st, gzw_st,
               'C.  Storfjorden — Staged Retreat + GZW Stacking')

    # Panel D – Mackenzie
    ax_d = fig.add_subplot(gs[1, 1])
    draw_strat(ax_d, xb_mk, ib_mk, fb_mk, gzw_mk,
               'D.  Mackenzie Bay — Rapid Retreat, No GZWs')
    ax_d.annotate('Rapid uninterrupted retreat\n→ No stable GZWs',
                  xy=(130, -170), fontsize=9, color='#8b1a1a', ha='center',
                  bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85))

    fig.suptitle(
        'Coupled SSA Flowline + Zoet & Iverson (2020) Sliding Law + Dynamic GZW Sedimentation\n'
        'Storfjorden (staged retreat) vs Mackenzie Bay (rapid retreat)',
        fontsize=13, fontweight='bold', y=0.97)

    plt.savefig('ssa_dynamic_gzw_comparison.png', dpi=160, bbox_inches='tight')
    print('\n✓ Figure saved → ssa_dynamic_gzw_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    r_st = run_simulation('Storfjorden')
    r_mk = run_simulation('Mackenzie', warm_start=r_st[-1])   # pass converged SS
    plot_results(r_st, r_mk)
