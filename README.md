# Ice-stream-ocean-model
Convert ice stream ocean model from MATLAB to Python
Model for Synchronization of Heinrich and Dansgaard-Oeschger Events through Ice-Ocean Interactions in Python, using numerical approach from [Mann et al 2021](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021PA004334) 

Integrating Luke Zoet's experimental findings into your Python script is a brilliant move. It transforms the model from a generic "fluid flow" simulation into a specialized tool for subglacial mechanics.

Based on Zoet's IGA lecture (specifically the 49:27 timestamp (https://youtu.be/QFY2H5DVWpw?si=WsjMqNuydUynR7Gx)), the "Regularized Coulomb" law is what allows for the "Stick-Slip" behavior needed for GZW spacing, while the "Debris" component adds stability.

1. The Physics: Adding the Zoet Slip LawCurrently, many simple models use a power-law (Weertman) sliding: $\tau_b = C u^{1/m}$.To reflect Zoet's research, you should replace or modify your basal shear stress function to a Regularized Coulomb Law.This law states that the basal drag ($\tau_b$) increases with velocity ($u$) but is capped by the strength of the till (which is controlled by effective pressure $N$ and the friction coefficient $\tan \phi$):$$\tau_b = \frac{\tau_c u}{(u + u_0)}$$ Where:$\tau_c$ (Coulomb Limit): The maximum stress the till can take before it "breaks" ($\tau_c = N \tan \phi$).$u_0$: A transition velocity where the sliding switches from viscous to plastic.
