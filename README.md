# Ice-stream-ocean-model
Convert ice stream ocean model from MATLAB to Python
Model for Synchronization of Heinrich and Dansgaard-Oeschger Events through Ice-Ocean Interactions in Python, using numerical approach from [Mann et al 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021PA004334) 

Integrating Luke Zoet's experimental findings (The Ring Shear Device) into your Python script is a brilliant move. It transforms the model from a generic "fluid flow" simulation into a specialized tool for subglacial mechanics.

Based on Zoet's IGA lecture (specifically the 49:27 timestamp (https://youtu.be/QFY2H5DVWpw?si=WsjMqNuydUynR7Gx)), the "Regularized Coulomb" law is what allows for the "Stick-Slip" behavior needed for GZW spacing, while the "Debris" component adds stability.

1. The Physics: Adding the Zoet Slip Law
   Currently, many simple models use a power-law (Weertman) sliding: $\tau_b = C u^{1/m}$. To reflect Zoet's research, you should replace or modify your basal shear stress function to a Regularized Coulomb Law.
   This law states that the basal drag ($\tau_b$) increases with velocity ($u$) but is capped by the strength of the till (which is controlled by effective pressure $N$ and the friction coefficient $\tan \phi$):

$\tau_b = \dfrac{\tau_c u}{u + u_0}$

Where

- **τ_c (Coulomb limit)**  
  Maximum shear stress the till can sustain before failure:  
  τ_c = N · tan(φ)

- **u**  
  Basal sliding velocity.

- **u₀**  
  Transition velocity at which basal sliding behavior changes from viscous to plastic.

2. Implementation in your Python Script
You can add a new function to our ice_stream_ocean_model.py to calculate the basal friction based on these experimental parameters.
3. How this creates "Periodic GZWs" in our model. By using this function, our model will now behave like a "Sticking and Slipping" system:
   Phase A (Stick): Velocity ($u$) is low. Drag is below the Coulomb limit. The grounding line stays still. Sediment builds up (GZW forms).
   Phase B (Pressure Build): Water pressure increases, decreasing $N$. This lowers the "ceiling" ($\tau_c$).
   Phase C (Slip): Once the driving stress exceeds the lowered $\tau_c$, the ice "breaks" into the plastic regime. It surges forward to a new position.
   Phase D (Reset): The surge thins the ice, $N$ increases again, and the ice "sticks" at a new location. A new GZW starts to form at the new gap.

Adding a Sediment Flux component is the next step in turning our physics model into a geomorphological tool. In glaciology, the growth of a Grounding Zone Wedge (GZW) is essentially a mass-balance problem: sediment is delivered to the grounding line by the ice stream and "dumped" where the ice starts to float.
1. The Physics: The Exner Equation for Subglacial SedimentTo model the changing height of the seafloor (the GZW), we use a version of the Exner Equation. It states that the change in bed elevation ($z_b$) over time depends on the divergence of the sediment flux ($q_s$):

∂z_b / ∂t = − 1 / (1 − λ) · ∇ · q_s

Where

- **z_b**  
  Bed elevation.

- **q_s**  
  Sediment flux — the volume of sediment transported per unit width.

- **λ**  
  Porosity of the till (typically 0.3–0.4).

- **∇ · q_s**  
  Divergence of the sediment flux.
