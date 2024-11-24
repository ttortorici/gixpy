# Installation

GixPy is distributed on PyPI and can be installed using pip:

```
pip install gixpy
```

It can be built from source by cloning https://github.com/ttortorici/gixpy.git and using Python's `build` tool (don't forget to `cd` into the repository):

```
pip install -U setuptools build
python -m build --wheel
pip install dist/*.whl
```

where `*` should be replaced with the actual wheel's filename, and `\` should be used instead of `/` on Windows.


# How to use

Here's an example of N files from a directory and tranforming them all at once. ***WARNING*** Currently there is a bug when you feed in a list or tuple of arrays. Every transformation gets saved over the first image of the return arrays. Instead just run the function on each array separately. The function is fast, so this shouldn't be a big problem for now (I'm moving on to more important things at the moment, and may come back and fix this bug).

``` python
import gixpy as gp
from pathlib import Path
import numpy as np
import fabio
import pyFAI

dir = Path("path/to/data")

poni = pyFAI.load(dir / "name-of-poni-file.poni")

rows, columns = poni.get_shape()

# inspect data
im_num = 0
for file in dir.glob("*.tif"):
    im_num += 1

# instantiate list to put data in
data = [None] * (im_num + 1)  # adding an extra space to put "weights"
transformed_data = [None] * (im_num + 1)

# make weights based on exposure time to track how many pixels get moved to each new location
exposure_time = 1800  # in seconds
data[-1] = np.ones((rows, columns)) * expsoure_time

# set geometric parameters not saved in poni file
incident_angle = 0.3  # in degrees
tilt_angle = 0  # in degrees

for ii, file in enumerate(dir.glob("*.tif")):
    data[ii] = fabio.open(file).data
    transformed_data[ii], new_beam_center = gp.transform(
    data,
    incident_angle,
    poni.get_pixel1(),
    poni.get_poni1(),
    poni.get_poni2(),
    poni.get_dist(),
    tilt_angle
)

transfromed_weights = transformed_data[-1]
adjusted_data = np.array(transformed_data[:-1]) * (exposure_time / transformed_weights)
```

