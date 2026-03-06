# README: GZW-Morphodynamics Framework

**Model Version:** 1.0 (Calibrated to R_2018 Seismic Profile)

**Lead Developer:** Adam Y. Kashdan

**Core Physics:** Regularized Coulomb Sliding (Zoet & Iverson, 2020)

## 1. Physical Basis

This simulation explores the rhythmic formation of Grounding Zone Wedges (GZWs) by coupling basal mechanics with sediment transport. The core engine is a **Regularized Coulomb Drag** function that transitions between viscous and plastic regimes based on velocity and effective pressure.

### Key Implementation Details (IGS 2026 Seminar Benchmarks):

* **Rollover Parameter ($p$):** Set to $1.0$ as the baseline for transition sharpness between the viscous and Coulomb regimes.
* **Clast-Induced Transition:** A `clast_factor` scaling of $3.0$ is applied to the viscous drag component. This accounts for large clasts (observed in Beaufort Sea cores) which concentrate stress and trigger the Coulomb yield at lower velocities ($U_b$) than clean till.
* **Debris Strengthening:** Includes a conditional rate-strengthening term for hard-bed interactions, simulating the "frozen fringe" effect where increased vertical velocity presses debris into the bed.

## 2. Mathematical Components

### Basal Shear Stress ($\tau_b$)

The model uses the regularized form to avoid the numerical singularities associated with pure plastic laws:


$$\tau_b = \frac{\tau_{visc} \cdot \tau_c}{(\tau_{visc}^p + \tau_c^p)^{1/p}}$$


Where $\tau_c = \mu_c N$.

### Asymmetric Aggradation Kernel

GZWs are modeled using a piecewise Gaussian kernel to reflect the asymmetry of subglacial deposition:

* **Proximal ($\sigma=6$):** Steep ice-contact face.
* **Distal ($\sigma=12$):** Progradational debris-flow slope.

## 3. Stratigraphic Horizon Definitions

The script reconstructs five primary horizons mirroring the **R_2018 Beaufort Sea** dataset:

1. **PQ:** Pre-Quaternary basement.
2. **TPQ:** Top Pre-Quaternary unconformity (includes narrow ridge features).
3. **GD:** Glacial Deposits (GZW mounds + background till).
4. **TG:** Top Glacial unconformity.
5. **DHD:** Deglacial to Holocene drape (hemipelagic settling).

## 4. Usage

The script is written in Python 3.x and requires `numpy`, `matplotlib`, and `scipy`.

```python
# To test sensitivity to clast presence:
HAS_CLASTS = True # Toggle to False to see GZW height reduction

```
