import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

# Suppress fsolve warnings if desired, though we handle output manually
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

class Params:
    pass

def bed(x, params):
    is_scalar = np.isscalar(x)
    x = np.atleast_1d(x)
    
    xsill = (x > params.sill_min) & (x < params.sill_max)
    xdsill = x >= params.sill_max
    sill_length = params.sill_max - params.sill_min
    
    b = params.b0 + params.bx * x
    
    b[xsill] = params.b0 + (params.bx * params.sill_min) + params.sill_slope * (x[xsill] - params.sill_min)
    b[xdsill] = params.b0 + (params.bx * params.sill_min) + params.sill_slope * sill_length + params.bx * (x[xdsill] - params.sill_max)
    
    if is_scalar:
        return b[0]
    return b

def u_schoof(hg, params):
    part1 = (params.Aglen * (params.rhoi * params.g)**(params.nglen + 1) * params.lambda_d**params.nglen) / (4**params.nglen * params.C)
    us = (part1**(1.0 / (params.m + 1))) * (hg)**(((params.m + params.nglen + 3) / (params.m + 1)) - 1)
    return us

def flowline_eqns(huxg, params):
    h = huxg[:params.Nx]
    u = huxg[params.Nx:2*params.Nx]
    xg = huxg[2*params.Nx]
    hf = (-bed(xg * params.xscale, params) / params.hscale) / (1 - params.lambda_d)

    dt = params.dt / params.tscale
    ds = params.dsigma
    Nx = params.Nx
    N1 = params.N1
    sigma = params.sigma
    sigma_elem = params.sigma_elem
    b = -bed(xg * sigma * params.xscale, params) / params.hscale
    Fh = np.zeros(Nx)
    Fu = np.zeros(Nx)
    
    m = params.m
    nglen = params.nglen
    lambda_d = params.lambda_d
    accum = params.accum
    a = accum / params.ascale
    eps = params.eps
    ss = params.transient
       
    h_old = params.h_old
    xg_old = params.xg_old
    
    # thickness
    Fh[0] = ss * (h[0] - h_old[0]) / dt + (2 * h[0] * u[0]) / (ds[0] * xg) - a
    
    Fh[1] = ss * (h[1] - h_old[1]) / dt - \
            ss * sigma_elem[1] * (xg - xg_old) * (h[2] - h[0]) / (2 * dt * ds[1] * xg) + \
            (h[1] * (u[1] + u[0])) / (2 * xg * ds[1]) - a
            
    Fh[2:Nx-1] = ss * (h[2:Nx-1] - h_old[2:Nx-1]) / dt - \
                 ss * sigma_elem[2:Nx-1] * (xg - xg_old) * (h[3:Nx] - h[1:Nx-2]) / (2 * dt * ds[2:Nx-1] * xg) + \
                 (h[2:Nx-1] * (u[2:Nx-1] + u[1:Nx-2]) - h[1:Nx-2] * (u[1:Nx-2] + u[0:Nx-3])) / (2 * xg * ds[2:Nx-1]) - a
    
    N1_i = N1 - 1
    Fh[N1_i] = (1 + 0.5 * (1 + (ds[N1_i] / ds[N1_i-1]))) * h[N1_i] - \
               0.5 * (1 + (ds[N1_i] / ds[N1_i-1])) * h[N1_i-1] - h[N1_i+1]
                        
    Fh[-1] = ss * (h[-1] - h_old[-1]) / dt - \
             ss * sigma_elem[-1] * (xg - xg_old) * (h[-1] - h[-2]) / (dt * ds[-2] * xg) + \
             (h[-1] * (u[-1] + u[-2]) - h[-2] * (u[-2] + u[-3])) / (2 * xg * ds[-2]) - a

    # velocity
    Fu[0] = (4 * eps) * (1 / (xg * ds[0])**((1 / nglen) + 1)) * \
            (h[1] * (u[1] - u[0]) * np.abs(u[1] - u[0])**((1 / nglen) - 1) - \
             h[0] * (2 * u[0]) * np.abs(2 * u[0])**((1 / nglen) - 1)) - \
            u[0] * np.abs(u[0])**(m - 1) - \
            0.5 * (h[0] + h[1]) * (h[1] - b[1] - h[0] + b[0]) / (xg * ds[0])
            
    Fu[1:Nx-1] = (4 * eps) * (1 / (xg * ds[1:Nx-1])**((1 / nglen) + 1)) * \
                 (h[2:Nx] * (u[2:Nx] - u[1:Nx-1]) * np.abs(u[2:Nx] - u[1:Nx-1])**((1 / nglen) - 1) - \
                  h[1:Nx-1] * (u[1:Nx-1] - u[0:Nx-2]) * np.abs(u[1:Nx-1] - u[0:Nx-2])**((1 / nglen) - 1)) - \
                 u[1:Nx-1] * np.abs(u[1:Nx-1])**(m - 1) - \
                 0.5 * (h[1:Nx-1] + h[2:Nx]) * (h[2:Nx] - b[2:Nx] - h[1:Nx-1] + b[1:Nx-1]) / (xg * ds[1:Nx-1])
                 
    Fu[N1_i] = (u[N1_i+1] - u[N1_i]) / ds[N1_i] - (u[N1_i] - u[N1_i-1]) / ds[N1_i-1]
    
    Fu[-1] = (1 / (xg * ds[-2])**(1 / nglen)) * \
             (np.abs(u[-1] - u[-2])**((1 / nglen) - 1)) * (u[-1] - u[-2]) - lambda_d * hf / (8 * eps)
             
    Fxg = 3 * h[-1] - h[-2] - 2 * hf         
    
    return np.concatenate((Fh, Fu, [Fxg]))


def main():
    params = Params()
    
    # Bed parameters
    params.b0 = -100           
    params.bx = -1e-3          
    params.sill_min = 2000e3   
    params.sill_max = 2100e3   
    params.sill_slope = 1e-3   

    # Physical parameters
    params.year = 3600*24*365  
    params.Aglen = 4.227e-25   
    params.nglen = 3           
    params.Bglen = params.Aglen**(-1/params.nglen)
    params.m = 1/params.nglen  
    params.accum = 0.28/params.year 
    params.C = 7e6        
    params.rhoi = 917         
    params.rhow = 1028        
    params.g = 9.81        

    # Scaling params
    params.hscale = 1000               
    params.ascale = 0.1/params.year    
    params.uscale = (params.rhoi*params.g*params.hscale*params.ascale/params.C)**(1.0/(params.m+1)) 
    params.xscale = params.uscale*params.hscale/params.ascale  
    params.tscale = params.xscale/params.uscale                
    params.eps = params.Bglen*((params.uscale/params.xscale)**(1/params.nglen))/(2*params.rhoi*params.g*params.hscale)  
    params.lambda_d = 1 - (params.rhoi/params.rhow)  
    params.transient = 0   

    # Grid parameters
    params.tfinal = 10e3*params.year   
    params.Nt = int(100)                    
    params.dt = params.tfinal/params.Nt
    params.Nx = 200                    
    params.N1 = 100                    
    params.sigGZ = 0.97                

    sigma1 = np.linspace(params.sigGZ/(params.N1+0.5), params.sigGZ, params.N1)
    sigma2 = np.linspace(params.sigGZ, 1, params.Nx-params.N1+1)
    params.sigma = np.concatenate([sigma1, sigma2[1:]])    
    params.sigma_elem = np.concatenate([[0], (params.sigma[:-1] + params.sigma[1:])/2.0]) 
    params.dsigma = np.diff(params.sigma) 

    # Solve for steady-state initial conditions
    params.accum = 1/params.year
    xg = 200e3/params.xscale
    hf = (-bed(xg*params.xscale, params)/params.hscale)/(1-params.lambda_d)
    h = 1 - (1-hf)*params.sigma
    u = 0.3*(params.sigma_elem**(1/3.0)) + 1e-3

    params.h_old = h.copy()
    params.xg_old = xg

    huxg0 = np.concatenate((h, u, [xg]))

    def flf(huxg_val):
        return flowline_eqns(huxg_val, params)

    print('Solving initial steady state...')
    huxg_init, info, ier, mesg = fsolve(flf, huxg0, full_output=True, maxfev=1000000, xtol=1e-6)
    if ier != 1:
        print("Warning: fsolve initial SS did not converge:", mesg)

    h = huxg_init[:params.Nx]
    u = huxg_init[params.Nx:2*params.Nx]
    xg = huxg_init[-1]
    
    u_bl = u_schoof(-bed(xg*params.xscale, params)/(1-params.lambda_d), params)
    u_num = u[-1]*params.uscale
    num_err = np.abs((u_num-u_bl)/u_bl)
    print(f'Error on initial S-S: {100*num_err:.4f}%')

    # Calculate transient GL evolution over bedrock peak
    xgs = np.nan * np.ones(params.Nt)
    hs = np.nan * np.ones((params.Nt, params.Nx))
    us = np.nan * np.ones((params.Nt, params.Nx))
    
    huxg_t = huxg_init.copy()
    params.h_old = huxg_t[:params.Nx].copy()
    params.xg_old = huxg_t[-1]

    params.transient = 1
    params.accum = 0.8 / params.year
    
    print('Solving transient solution...')
    for t in range(params.Nt):
        huxg_t, info, ier, mesg = fsolve(flf, huxg_t, full_output=True, maxfev=1000000, xtol=1e-6)
        if ier != 1:
            print(f"Warning: fsolve at timestep {t+1} did not converge:", mesg)
            
        params.h_old = huxg_t[:params.Nx].copy()
        params.xg_old = huxg_t[-1]
        
        print(f"Time step: {t+1}")
        
        xgs[t] = huxg_t[-1]
        hs[t, :] = huxg_t[:params.Nx]
        us[t, :] = huxg_t[params.Nx:2*params.Nx]

    # Plot transient solution
    plt.figure(1, figsize=(8, 10))
    ts = np.linspace(0, params.tfinal/params.year, params.Nt)

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(ts, xgs*params.xscale/1e3, linewidth=3)
    ax1.set_xlabel('time (yr)')
    ax1.set_ylabel('x_g')

    ax2 = plt.subplot(3, 1, 2)
    c2 = ax2.contourf(ts, params.sigma_elem, hs.T * params.hscale, 20)
    plt.colorbar(c2, ax=ax2)
    ax2.set_xlabel('time (yr)')
    # Note: inverted title from MATLAB `xlabel('sigma')` logic, we add it to ylabel
    ax2.set_ylabel('sigma')
    ax2.set_title('thickness (m)')
    ax2.invert_yaxis()

    ax3 = plt.subplot(3, 1, 3)
    c3 = ax3.contourf(ts, params.sigma, us.T * params.uscale * params.year, 20)
    plt.colorbar(c3, ax=ax3)
    ax3.set_xlabel('time (yr)')
    ax3.set_ylabel('sigma')
    ax3.set_title('velocity (m/yr)')
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
