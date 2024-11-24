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
        - affiliation: 1
    - name: Charles T. Rogers
        - affiliation: 1
affiliations:
    - name: Department of Physics, University of Colorado Boulder
      index: 1
date: 22 November 2024
bibliography: paper.bib
---

# Summary

Grazing incidence X-ray scattering techniques are utilized to investigate the crystal structure and orientation of crystallites in thin films. X-ray experiments that utilize area detectors to observe the resulting interference pattern, in general, require images to be transformed into data with respect to reciprocal space. However, for grazing incidence X-ray experiments, the experimental geometry leads to additional considerations to this transformation.

# Statement of need

There currently exists many tools for transforming wide-angle X-ray scattering (WAXS) and small-angle X-ray scattering (SAXS) images into reciprocal space, including pyFAI [@pyFAI] and Nika [@nika]. However, these tools lack the capability of processing raw images taken for grazing incidence wide/small-angle X-ray scattering (GIWAXS/GISAXS). Here we refer to both GIWAXS and GISAXS as grazing incidence X-ray scattering (GIXS). There exists an existing tool, [pygix](https://github.com/tgdane/pygix), that is capable of processing GIWAXS and GISAXS images into reciprocal space, but this package, GixPy, differentiates itself from pygix through transparency and agnosticism.

GixPy seeks transparency in order to serve not only as a useful tool, but also an educational tool for those who are less familiar with the intricacies of grazing incidence experiments. This goal is achieved by maintaining well documented and commented code that utilizes direct computation (as opposed to relying on look-up tables), and is written with source-code readability in mind. This is intended to allow students to have an accessible resource, with examples, that helps them learn some of the intricacies of GIXS image processing for analysis.

GixPy's agnosticism allows it to be utilized as an intermediary step for anyone who already has a preferred WAXS/SAXS image processing software. This allows users to not need to learn an entirely new system to do their analysis in, and can simply use GixPy to pre-process an image before giving it to their preferred environment for analysis.

