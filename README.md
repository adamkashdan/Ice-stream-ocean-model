# Ice-stream-ocean-model
Convert ice stream ocean model from MATLAB to Python
Model for Synchronization of Heinrich and Dansgaard-Oeschger Events through Ice-Ocean Interactions in Python, using numerical approach from [Mann et al 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021PA004334) 

Integrating Luke Zoet's experimental findings (The Ring Shear Device) into your Python script is a brilliant move. It transforms the model from a generic "fluid flow" simulation into a specialized tool for subglacial mechanics.

Based on Zoet's IGA lecture (specifically the 49:27 timestamp (https://youtu.be/QFY2H5DVWpw?si=WsjMqNuydUynR7Gx)), the "Regularized Coulomb" law is what allows for the "Stick-Slip" behavior needed for GZW spacing, while the "Debris" component adds stability.

## 1. The Physics one:
Adding the Zoet Slip Law Currently, many simple models use a power-law (Weertman) sliding: $\tau_b = C u^{1/m}$. To reflect Zoet's research, you should replace or modify your basal shear stress function to a Regularized Coulomb Law.
This law states that the basal drag ($\tau_b$) increases with velocity ($u$) but is capped by the strength of the till (which is controlled by effective pressure $N$ and the friction coefficient $\tan \phi$):

$\tau_b = \dfrac{\tau_c u}{u + u_0}$

Where

- **$τ_c$ (Coulomb limit)**  
  Maximum shear stress the till can sustain before failure:  
  $τ_c$ = N · tan(φ)

- **u**  
  Basal sliding velocity.

- **$u_o$**  
  Transition velocity at which basal sliding behavior changes from viscous to plastic.

## 2. Implementation in Python Script
Can add a new function to ice_stream_ocean_model.py to calculate the basal friction based on these experimental parameters.

## 3. How this creates "Periodic GZWs" in model 
By using this function, model will now behave like a "Sticking and Slipping" system:
Phase A (Stick): Velocity ($u$) is low. Drag is below the Coulomb limit. The grounding line stays still. Sediment builds up (GZW forms).
Phase B (Pressure Build): Water pressure increases, decreasing $N$. This lowers the "ceiling" ($\tau_c$).
Phase C (Slip): Once the driving stress exceeds the lowered $\tau_c$, the ice "breaks" into the plastic regime. It surges forward to a new position.
Phase D (Reset): The surge thins the ice, $N$ increases again, and the ice "sticks" at a new location. A new GZW starts to form at the new gap.

Adding a Sediment Flux component is the next step in turning our physics model into a geomorphological tool. In glaciology, the growth of a Grounding Zone Wedge (GZW) is essentially a mass-balance problem: sediment is delivered to the grounding line by the ice stream and "dumped" where the ice starts to float.
## 4. The Physics 2:
   The Exner Equation for Subglacial SedimentTo model the changing height of the seafloor (the GZW), we use a version of the Exner Equation. It states that the change in bed elevation ($z_b$) over time depends on the divergence of the sediment flux ($q_s$):

$\frac{\partial z_b}{\partial t} = -\frac{1}{1-\lambda}\nabla\cdot q_s$

Where

- **z_b**  
  Bed elevation.

- **q_s**  
  Sediment flux — the volume of sediment transported per unit width.

- **λ**  
  Porosity of the till (typically 0.3–0.4).

- **∇ · q_s**  
  Divergence of the sediment flux.

## 5. Calculating q_s (the “Conveyor Belt”)

Following the *stick–slip* concept of Zoet, sediment flux is assumed to be proportional to the basal sliding velocity and the thickness of the deforming till layer:

$q_s = u \, h_t$

Where

- **q_s**  
  Sediment flux (volume of sediment transported per unit width).

- **u**  
  Basal sliding velocity.

- **h_t**  
  Thickness of the deforming till layer.

## 6. Python Implementation for model
Can add this logic to our time-stepping loop in ice_stream_ocean_model.py. This function calculates how much the "wedge" grows at the specific coordinate of the grounding line ($x_{gl}$).
  
## 7. Why this proves "Periodic Spacing"
When you run this in loop, the following behavior emerges:
Stick Phase: $x_{gl}$ is stationary. update_gzw_height repeatedly adds height to the same spot. A steep GZW grows.
Slip Phase: The "Zoet Slip Law" triggers. $x_{gl}$ jumps forward 10km in the model.
New Stick Phase: The grounding line stops at $x_{gl} + 10km$. A new wedge starts to grow.
The Result: Your z_bed array will eventually show a series of bumps—periodically spaced GZWs—whose distance is determined entirely by your till mechanics. 

## 8. Technical Note: Preventing "Overprinting"
Mentions "overprinting." This is the biggest challenge in model.
The Problem: If the glacier retreats too slowly, the new GZW will just sit on top of the old one, creating one giant pile.
The Solution: It need to ensure your retreat rate ($dR/dt$) is faster than the wedge growth rate ($dH/dt$). In Python code, this means the "Slip" phase must move the grounding line far enough that the next "Stick" phase creates a distinct, separate bump.

Concern about overprinting; the model needs a "threshold-and-reset" logic. The goal is to ensure that when the ice slips, it doesn't merely slide a few meters; it must "jump" to a new stable position. This creates the gaps between the wedges that C.L. Batchelor, J.A. Dowdeswell, et al. see in the Beaufort Sea.

## 9. The Logic: "The Jump Distance"
To avoid overprinting, we need to introduce a Minimum Jump Distance ($L_{jump}$). In nature, this is often controlled by the bathymetry (the next "hill" or shallow spot on the seafloor). In Phyton code, we can tie the "Slip" phase to a rapid drop in basal resistance ($\tau_b$) that only stops when the ice reaches a new equilibrium.

## 10. Updating Python Script (ice_stream_ocean_model.py)
Add this logic to simulation loop to handle the transition from "Building a GZW" to "Jumping to avoid overprinting."
