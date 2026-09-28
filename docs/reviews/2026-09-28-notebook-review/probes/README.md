# Review probes — Philippine Biodiversity Field-Survey Capstone

Companion to DIMER_Philippine_Biodiversity_Capstone_Notebook_Review.md (2026-09-28).

Target notebook SHA-256: fa2be8830eb9a15de1f1e9946ea7761950776415329ec567caedbeb55c569894

Run:  python run_probes.py path/to/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb
Requires Python >= 3.10, standard library only. Each probe writes <name>.out.json beside itself;
the .out.json files in this ZIP are the outputs recorded during the review.

Evidence boundary: no notebook cell and no carried NumPy/PyTorch code was executed. P3 and P4 are
pure-Python re-implementations of the carried logic (P4 uses synthetic embeddings, not BioCLIP).
P5 is source inspection. Nothing here is evidence that the notebook runs on Colab.
