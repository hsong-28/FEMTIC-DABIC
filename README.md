# FEMTIC-DABIC
FEMTIC-DABIC is a 3-D magnetotelluric (MT) inversion code developed based on FEMTIC (v4.2). It is designed to determine and use the statistically optimal regularization parameter (*α*) in each iteration, thereby controlling the overall smoothness of the final subsurface resistivity model.

FEMTIC-DABIC leverages Akaike’s Bayesian Information Criterion (ABIC) to quantify the magnitude of the marginal likelihood function: a key metric in Bayesian theory, where higher values indicate a more optimal hyperparameter (here, *α*) that balances data fitting and prior model information. To render ABIC computationally tractable for 3-D MT inversion, we propose a data-space variant of ABIC, termed DABIC. The 3-D MT data-space inversion workflow built on this DABIC variant is formally referred to as the D-DABIC inversion method.

For details on the methodology and workflow of the D-DABIC inversion method, please refer to:\
*Song, H., Yu, P., Usui, Y., Uyeshima, M., Diba, D., & Zhang, L. Three-dimensional Magnetotelluric Inversion based on a Data Space variant of Akaike’s Bayesian Information Criterion. Geophysics (2025), https://doi.org/10.1190/geo-2025-0233.*

For more information about the FEMTIC code, please refer to the FEMTIC repository: https://github.com/yoshiya-usui/femtic.


## Release note

***v1.4*** Jan. 12, 2025: I've revised the D-DABIC workflow to ensure it is capable of incorporating the distortion correction functionality.

***v1.3*** Sep. 13, 2025: Added Minimum Norm (MN) Stabilizer with Depth of Investigation (DOI) Support. Introduced a new regularization option (|m - m_r|) to constrain inversion toward a reference model (m_r); the primary purpose of this option (for now) is to enable DOI analysis for model appraisal.

***v1.2*** Sep. 11, 2025: Reference Model (m_r) Configuration Option. Added support for defining a user-provided reference model (m_r), enabling physics-based constraints in the inversion.

***v1.1*** Dec. 30, 2024: Laplacian Filter (LF) for Marginal Likelihood Maximization. Enabled the LF as an alternative regularization during the D-DABIC optimization.

***v1.0*** Nov. 28, 2024: Core FEMTIC-DABIC Framework. Implemented a 3-D data-space inversion method using a data-space variant of the Akaike’s Bayesian Information Criterion (D-DABIC).


## Contact information
If you encounter any issues, have questions, or want to provide feedback, please feel free to contact:
Han SONG (Email: 1831736@tongji.edu.cn)
You may also report issues via the project's repository (e.g., GitHub Issues).
