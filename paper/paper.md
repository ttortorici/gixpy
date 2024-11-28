---
title: 'GixPy: A Python package for transforming grazing incidence X-ray scattering images'
tags:
    - Python
    - X-ray
    - X-ray diffraction
    - XRD
    - grazing incidence
    - GIXS
    - GIWAXS
    - GISAXS
    - GIXD
    - educational
    - thin films
authors:
    - name: Edward Tortorici
      affiliation: 1
    - name: Charles T. Rogers
      affiliation: 1
affiliations:
    - name: Department of Physics, University of Colorado Boulder
      index: 1
date: 22 November 2024
bibliography: paper.bib
---

# Summary

Grazing incidence X-ray scattering techniques are utilized to investigate the crystal structure and orientation of crystallites in thin films [@how-to-giwaxs]. X-ray experiments that utilize area detectors to observe the resulting interference pattern, in general, require images to be transformed into data with respect to reciprocal space. However, for grazing incidence X-ray experiments, the experimental geometry leads to additional considerations to this transformation.

# Statement of need

There currently exists many tools for transforming wide-angle X-ray scattering (WAXS) and small-angle X-ray scattering (SAXS) images into reciprocal space, including pyFAI [@pyfai] and Nika [@nika]. However, these tools lack the capability of processing raw images from grazing incidence wide/small-angle X-ray scattering (GIWAXS/GISAXS) experiments. Here we refer to both GIWAXS and GISAXS as grazing incidence X-ray scattering (GIXS). There exists an existing tool, [pygix](https://github.com/tgdane/pygix), that is capable of processing GIWAXS and GISAXS images into reciprocal space. However, GixPy, differentiates itself through its design goals of transparency and workflow agnosticism.

GixPy seeks transparency in order to serve not only as a useful tool, but also an educational tool for those who are less experienced with of grazing incidence experiments. This goal is achieved by maintaining well documented and commented code that utilizes direct computation (as opposed to relying on look-up tables), and is written with source-code readability in mind. This is intended to allow students and researchers to have an accessible resource, with examples, that helps them learn how to process GIXS image processing and why they need to be processed for analysis.

GixPy's agnosticism allows it to be utilized as an intermediary step for anyone who already has a preferred WAXS/SAXS image processing software. This allows users to not need to learn an entirely new system to do their analysis in, and can simply use GixPy to pre-process an image before giving it to their preferred environment for analysis. However, since GixPy is built as a Python tool, it has been built to seamlessly integrate with pyFAI to serve as a complete processing tool.

# Powder transformation

Existing tools, such as Nika and pyFAI transform images with the assumption that samples are a powder, such that the scattering results in Debye-Scherrer cones [@cullity3rd]. A typical experimental setup is exemplified in Figure \ref{fig:waxs}. An area detector is used to intersect the Debye-Scherrer cones to detect rings of constructive interference.

![A typical WAXS/SAXS experiment. A powder sample is exposed to an incident beam, resulting in Debye-Scherrer cones of constructive interference. An area detector is used to intersect the cones to detect rings.\label{fig:waxs}](images/waxs.png)

The scattering angle $2\theta$ can be related to reciprocal space through Bragg's law:

\begin{equation}
\label{eq:bragg}
    q = \frac{4\pi}{\lambda}\sin\theta,
\end{equation}

where $\lambda$ is the wavelength of the scattered X-rays. The scattering angle can be determined from the radius of the ring on the detector $r$ and the sample-detector distance $d_{sd}$:

\begin{equation}
    \tan 2\theta = \frac{r}{d_{sd}},
\end{equation}

so a powder image transformation calculates $q$ from the ring radii using

\begin{equation}
\label{eq:powder}
    q = \frac{4\pi}{\lambda}\sin\bigg[\frac{1}{2}\tan^{-1}\bigg(\frac{r}{d_{sd}}\bigg)\bigg].
\end{equation}

A GixPy transformation processes an image, such that a processed image can be transformed assuming powder symmetry will produce correct results.

# Geometric assumptions

GixPy supports geometries where the incident beam is perpendicular to the detector and the sample is brought into the beam path (see Figure \ref{fig:experimental-geometry}). This means that the point of normal incidence (PONI) on the detector and where the incident beam hits the detector (the beam center) are the same locations on the detector.

The top-left pixel of the detector is the origin of the data array and defines the PONI as the distance from the bottom-left corner of the detector (consistent with pyFAI), as seen in Figure \ref{fig:pixel-poni}. Transforming between $\mathbf{r}_{\text{poni}_{i,j}}$ and $\mathbf{r}_\text{poni}$ can be done with the following relation:

![The supported experimental geometry has the detector positioned surface normal to the incident beam, and the grazing angle is set by rotating the sample by $\alpha_i$ relative to the beam.\label{fig:experimental-geometry}](images/experimental-geometry.png)

\begin{align}
    \text{poni}_1 &= \bigg(R - i_\text{poni} - \frac{1}{2}\bigg)p_z\\
    \text{poni}_2 &= \bigg(j_\text{poni} + \frac{1}{2}\bigg)p_x,
\end{align}

where $R$ is the number of rows in the image and $p_x$ and $p_z$ are the horizontal and vertical widths of a pixel respectively. This transformation can be done with 

```Python
gixpy.poni.convert_to(poni_ij, pixel_widths, image_shape)
```

and reversed with

```Python
gixpy.poni.convert_from(poni, pixel_widths, image_shape)
```

Where each input can be a tuple, list, or NumPy array, with the first element being the vertical value and the second element being the horizontal value.

![An example detector with $10 \times 10$ pixels. The PONI is described by the distance (in meters) from the bottom left corner. A user can convert a PONI in the $(i_\text{poni}, j_{poni})$ format using the `gixpy.poni.convert_to()` function.\label{fig:pixel-poni}](images/pixel-poni.png)

# Scattering geometry

In grazing incidence X-ray scattering, there is a very small angle, called the grazing angle or incident angle, between the plane of the film and the incident beam. The incident beam, with wavelength $\lambda$, has a wavevector $\mathbf{k}_i$ with magnitude $2\pi/\lambda$. Elastic scattering, due to bound electrons in the film, will result in a scattered ray with wavevector $\mathbf{k}_s$ with the same magnitude. In the sample frame (Figure \ref{fig:scattering-angles}a), the axes are oriented such that the $z$-direction is surface-normal to the film plane, and the $x$-direction is perpendicular to the direction of the incident beam. In the sample frame, the direction of the scattered ray can be described by rotations from the $y$-direction:

\begin{equation}
\mathbf{k}_s = \frac{2\pi}{\lambda} R_x(\alpha_s)R_z(\phi_s)\ \hat{y},
\end{equation}

where $\hat{y}$ is the $y$-direction in the sample frame. This is a non-conventional order of operations, but it leads to simplifications in the calculations. In the lab frame (see Figure \ref{fig:scattering-angles}), the axes are denoted $\hat{x}'$, $\hat{y}'$, and $\hat{z}'$, and the $\hat{y}'$ direction is in the direction of the beam. A $R_x(\alpha_i)$ rotation will move from the sample frame to the lab frame, so in the lab frame, the scattered wavevector is

\begin{equation}
\begin{split}
    \mathbf{k}_s &= \frac{2\pi}{\lambda} R_x(\alpha_i)R_x(\alpha_s)R_z(\phi_s)\ \hat{y}'\\
    &= \frac{2\pi}{\lambda} R_x(\alpha_s+\alpha_i)R_z(\phi_s)\ \hat{y}'.
\end{split}
\end{equation}

![(a) Coordinates in the sample frame. (b) Coordinates in the lab frame.\label{fig:scattering-angles}](images/scattering-angles.png)

The scattering angles can then be related to coordinates on the detector as seen in Figure \ref{fig:scattering-detector}:

\begin{align}
    z'' &= d_{sd} \tan(\alpha_s+\alpha_i)\\
    x'' &= \sqrt{d_{sd}^2+z^2}\tan(\phi_s),
\end{align}

where $z''$ and $x''$ are coordinates on the detector with respect to the $x''$-$z''$-plane. Note: the $z''$-direction is the same as the $z'$-direction, but has its origin at the PONI instead of the sample, but the $x''$-direction is reversed from the $x'$-direction.

Row $i$ and column $j$ coordinates can be related to $\mathbf{r}$ through the equations

\begin{align}
    x'' &= (j_\text{poni} - j)p_x\\
    z'' &= (i_\text{poni} - i)p_z,
\end{align}

where $i_\text{poni}$ and $j_\text{poni}$ are the row and column index of the PONI respectively, and $p_x$ and $p_z$ are the horizontal and vertical widths of a rectangular pixel.

![In the lab frame, the scattering angles can be related to coordinates ($x$ and $z$) on the detector relative to the PONI.\label{fig:scattering-detector}](images/scattering-detector.png)

![The detector origin is the center of the top-left pixel, and $i$ and $j$ are the row and column indices respectively. Distances from the PONI $\mathbf{r}=x\;\hat{x}''+z\;\hat{z}''$ can also be described by their magnitude $r=\sqrt{x^2+z^2}$ and azimuthal angle $\psi$.\label{fig:detector-origin}](images/detector-origin.png)

# Reciprocal space

The scattering vector $\mathbf{q}$ is defined as

\begin{equation}
    \mathbf{q} = \mathbf{k}_s - \mathbf{k}_i,
\end{equation}

and the magnitude of the scattering vector can be related to the Bragg angle $\theta$ through Bragg's law: Equation (\ref{eq:bragg}). The magnitude of the scattering vector is also related to a lattice plane spacing $d$ via

\begin{equation}
d = \frac{2\pi}{q}.
\end{equation}

In the sample frame (Figure \ref{fig:scattering-angles}a),

\begin{align}
    \mathbf{k}_i &= \frac{2\pi}{\lambda}
    \begin{bmatrix}
        0\\
        \cos{\alpha_i}\\
        -\sin{\alpha_i}
    \end{bmatrix},\\
    \mathbf{k}_s &= \frac{2\pi}{\lambda}
    \begin{bmatrix}
        -\sin{\phi_s}\\
        \cos{\alpha_s}\cos{\phi_s}\\
        \sin{\alpha_s}\cos{\phi_s}
    \end{bmatrix},
\end{align}

so in the sample frame, the scattering vector can be written

\begin{equation}
    \mathbf{q} = \mathbf{k}_s - \mathbf{k}_i = \frac{2\pi}{\lambda}
    \begin{bmatrix}
        -\sin{\phi_s}\\
        \cos{\alpha_s}\cos{\phi_s} - \cos{\alpha_i}\\
        \sin{\alpha_s}\cos{\phi_s} + \sin{\alpha_i}
    \end{bmatrix}.
\end{equation}

Many thin films have cylindrical symmetry, in that individual crystallites have a preferred orientation of a lattice vector in the $z'$-direction, but are disordered in rotations on the surface of the substrate [@film-texture-sim]. The cylindrical symmetry of the crystallites leads to cylindrical symmetry in reciprocal space, where $q_{xy}=\sqrt{q_x^2+q_y^2}$ represents the radial axis. A grazing incidence X-ray image transformation into reciprocal space then requires the following calculations:

\begin{align}
\label{eq:qxy}
    q_{xy} &= \frac{2\pi}{\lambda}(\sin^2\phi_s+(\cos{\alpha_s}\cos{\phi_s} - \cos{\alpha_i})^2)\\
\label{eq:qz}
    q_z &= \frac{2\pi}{\lambda}(\sin{\alpha_s}\cos{\phi_s} + \sin{\alpha_i}).
\end{align}

Equations (\ref{eq:qxy}) and (\ref{eq:qz}) can be calculated using $\alpha_s$, $\alpha_i$, $\cos\phi_s$, and $\sin\phi_s$ as determined by the detector coordinates $x''$ and $z''$ and the sample-detector distance $d_{sd}$ (Figure \ref{fig:scattering-detector}):

\begin{align}
\label{eq:alpha_s}
    \alpha_s &= \tan^{-1}\bigg(\frac{z''}{d_{sd}}\bigg)-\alpha_i\\
\label{eq:cos_phi}
    \cos\phi_s & = \sqrt{\frac{z''^2+d_{sd}^2}{x''^2+z''^2+d_{sd}^2}}\\
\label{eq:sin_phi}
    \sin\phi_s & = \frac{x''}{\sqrt{x''^2+z''^2+d_{sd}^2}}
\end{align}

# Reverse transform

In order to suffice the agnosticism goal, after GixPy calculates $q_{xy}$ and $q_z$ for each pixel location, it then relates these to $r_{xy}$ and $r_{z}$ such that a powder transformation (utilizing Equation (\ref{eq:powder})) will produce the correct results. This is done by reversing the powder transformation:

\begin{align}
    r = d_{sd}\tan\bigg[2\sin^{-1}\bigg(\frac{\lambda q}{4\pi}\bigg)\bigg],
\end{align}

where $q=\sqrt{q_{xy}^2+q_z^2}$. The following trig identities [@schaum]:

\begin{align}
    \tan2u &= \frac{2\tan u}{1-\tan^2 u}\\
    \tan \bigg[\sin^{-1}\bigg(\frac{u}{2}\bigg)\bigg] &= \frac{u}{\sqrt{4-u^2}},
\end{align}

can be used to show

\begin{align}
    r = d_{sd}q'\frac{\sqrt{4 - q'^2}}{2 - q'^2},
\end{align}

where $q'=\lambda q/4\pi$.

The azimuthal angle $\psi$ (as seen in Figures \ref{fig:scattering-detector} and \ref{fig:detector-origin}) is related to both $r$ and $q$ in the same way:

\begin{align}
    \cos\psi &= \frac{r_{xy}}{r} = \frac{q_{xy}}{q}\\
    \sin\psi &= \frac{r_z}{r} = \frac{q_z}{q},
\end{align}

so

\begin{align}
\label{eq:rxy}
    r_{xy} &= d_{sd}q'_{xy} \frac{\sqrt{4 - q'^2_{xy} - q'^2_z}}{2 - q'^2_{xy} - q'^2_z}\\
\label{eq:rz}
    r_z &= d_{sd} q'_z\frac{\sqrt{4 - q'^2_{xy} - q'^2_z}}{2 - q'^2_{xy} - q'^2_z},
\end{align}

where $q'_{xy}=\lambda q_{xy}/4\pi$ and $q'_{z}=\lambda q_{z}/4\pi$.

# Seeding the transformed image

For every pixel's location relative to the PONI, GixPy calculates an $r_{xy}$ and $r_z$ using Equations (\ref{eq:rxy}) and (\ref{eq:rz}) and then creates a new image where all the counts from each pixel is moved to a location corresponding to $r_{xy}$ and $r_z$ for that pixel. As illustrated in Figure \ref{fig:detector-transform}, the new image will have a PONI corresponding to the maximum value of $r_{xy}$ and $r_z$ of all the pixels:

\begin{align}
\label{eq:poni-iT}
    i^T_\text{poni} &= \text{max}(r_z) / p_z\\
\label{eq:poni-jT}
    j^T_\text{poni} &= \text{max}(r_{xy}) / p_x,
\end{align}

where $p_z$ and $p_x$ are the vertical and horizontal widths of a pixel respectively. $r_{xy}$ and $r_z$, for each pixel, correspond to row $i^T$ and column $j^T$ in the transformed image according to

\begin{align}
\label{eq:iT}
    i^T &= \text{max}(r_z) - r_z\\
\label{eq:jT}
    j^T &= \text{max}(r_{xy}) - r_{xy}.
\end{align}

The transformed image will have rows $R^T$ and columns $C^T$ as determined by

\begin{align}
\label{eq:rowsT}
    R^T &= \text{ceil}(\text{max}(r_z) - \text{min}(r_z)) + 1\\
\label{eq:colsT}
    C^T &= \text{ceil}(\text{max}(r_{xy})- \text{min}(r_{xy})) + 1,
\end{align}

where the minimums are negatively valued if the PONI is on the detector, $\text{ceil}(x)$ is the ceiling function, and the extra 1 is padding to guarantee that there is room for the pixel splitting step. The transformed image is seeded by creating a NumPy array of zeros with shape $(R^T,\ C^T)$. To account for how many pixels are moved to a new pixel location, a second NumPy array, referred to as the transformed flat field is also created.

![The transformed image's PONI and shape can be determined by the minimums and maximuns of the $r_{xy}$ and $r_z$ found in the transformation calculation.\label{fig:detector-transform}](images/detector-transform.png)

# Pixel splitting

A pixel index is determined by flooring $i^T$ and $j^T$, and the counts are split amongst that pixel's neighbors, as seen in Figure \ref{fig:pixel-split}. Remainders $\rho$ are determined by

\begin{align}
    \rho_i &= i^T - \text{floor}(i^T)\\
    \rho_j &= j^T - \text{floor}(j^T),
\end{align}

and the counts get distributed according to following weights

\begin{align}
\label{eq:w1}
    w_\text{current pixel} &= (1-\rho_i)(1-\rho_j)\\
\label{eq:w2}
    w_\text{column neighbor} &= (1-\rho_i)\rho_j\\
\label{eq:w3}
    w_\text{row neighbor} &= \rho_i(1-\rho_j)\\
\label{eq:w4}
    w_\text{diagonal neighbor} &= \rho_i\rho_j,
\end{align}

where the sum of the weights adds to 1. It is clear that when the remainders are zero, then the "current pixel" gets all the counts, and when both remainders are 0.5, all the pixels get 1/4 the counts.

![The counts are split amongst neighboring pixels.\label{fig:pixel-split}](images/pixel-split.png)

# Moving pixels

Every pixel in the original image is looped over, and the new row and column indices $(i^T,\ j^T)$ are determined using Equations (\ref{eq:iT}) and (\ref{eq:jT}) by first calculating scattering angles using Equations (\ref{eq:alpha_s}) to (\ref{eq:sin_phi}). Then $q_{xy}$ and $q_z$ are computed with Equations (\ref{eq:qxy}) and (\ref{eq:qz}), $r_{xy}$ and $r_z$ with Equations (\ref{eq:rxy}) and (\ref{eq:rz}), and the new PONI and image shape with Equations (\ref{eq:poni-iT}), (\ref{eq:poni-jT}), (\ref{eq:rowsT}), and (\ref{eq:colsT}). The weights are calculated for each pixel using Equations (\ref{eq:w1}) to (\ref{eq:w4}), and the counts in pixel $(i,\ j)$ from the original image are added to the counts in pixel $(i^T,\ j^T)$ and its neighbors according to the pixel splitting weights. This is executed by compiled C code written in gixpy.c, but a Pythonic version of this step would look like:

```Python
new_image = np.zeros((R_T, C_T))  # as determined by Eq (34) and (35)
new_flatfield = np.zeros((R_T, C_T))
for i in range(image.shape[0]):      # loop over rows of the original image
    for j in range(image.shape[1]):  # loop over columns of the original image
        new_i = int(i_T[i, j])       # floor of i^T, as calculated by Eq (32)
        new_j = int(j_T[i, j])       # floor of j^T, as calculated by Eq (33)
        
        # calculate weights
        remainder_i = i_T[i, j] - new_i  # Eq (36)
        remainder_j = j_T[i, j] - new_j  # Eq (37)
        w_current_pixel = (1 - remainder_i) * (1 - remainder_j)  # Eq (38)
        w_column_neighbor = (1 - remainder_i) * remainder_j      # Eq (39)
        w_row_neighbor = remainder_i * (1 - remainder_j)         # Eq (40)
        w_diagonal_neighbor = remainder_i * remainder_j          # Eq (41)

        # split pixel
        new_image[new_i, new_j] += image[i, j] * w_current_pixel
        new_image[new_i + 1, new_j] += image[i, j] * w_row_neighbor
        new_image[new_i, new_j + 1] += image[i, j] * w_column_neighbor
        new_image[new_i + 1, new_j + 1] += image[i, j] * w_diagonal_neighbor

        # account for pixel movement
        new_flatfield[new_i, new_j] += w_current_pixel
        new_flatfield[new_i + 1, new_j] += w_row_neighbor
        new_flatfield[new_i, new_j + 1] += w_column_neighbor
        new_flatfield[new_i + 1, new_j + 1] += w_diagonal_neighbor
```




# Refrences
