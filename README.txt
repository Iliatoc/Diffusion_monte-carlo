INCOMPLETE README AS OF 25/01/2026 - requires further clarification

Current version utilises two files: "Radiolum_decay_512ps_YAG.xlsx" and "Diffusion_aerogel_withTDCR_clean.py". The rest of the files in the directory are from the earlier versions.

The program aims to model the behaviour of the individual photons in a diffusive medium (aerogel) via Monte-Caro simulation of the linear photon path trough the material and individual random Rayleigh scattering events. This is supplemented with refraction upon exit from the material and chance of total internal reflection. 

The position of the photons at the end of simulation is collected from the collection sphere (normally 6cm in radius) centered on the middle of the aerogel cylinder. This collection is recorded separately. Furthermore, the photons can be checked if they occupy space assigned to one of the three PMT detectors.

In addition, the photons are initially sorted into batches of adjustable size to simulate the flashes of light from radioactive decay energy depositions. After the assignement to the collection sphere and detector, these photons are distributed with delays to their arrival based on the file "Radiolum_decay_512ps_YAG.xlsx" which is followed by calculation if each batch produces a double, triple or no coincidence detection (with adjustable coincidence window of 40ns and dead time equal to the maximum decay delay time (refer to doi:10.1016/j.apradiso.2008.02.062 and doi: for mechanisms of TDCR as was assumed in the program)). The fractions of double and triple coincidences to the number of batches is then saved to a new file.
