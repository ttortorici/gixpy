"""
Tools for working with experimental geometry using pyFAI.
The point of normal incidence (PONI) refers to the intersection point on the detector
corresponding to a line that is both surface normal to the detector and intersects the sample.
For a detector that is surface normal to the incident beam, the PONI is equivalent to the beam center.
"""

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib
import pyFAI
from pathlib import Path
import pyFAI.azimuthalIntegrator
import pyFAI.detectors
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def new_poni(dist: float, poni1: float, poni2: float, shape: tuple, pixel1: float = 75e-6, pixel2: float = 75e-6,
        wavelength: float = 1.54185e-10, rot1: float = 0, rot2: float = 0, rot3: float = 0, orientation: int = 2):
    """
    Creates a pyFAI.azimuthalIntegrator object from user defined parameters

    :param dist: sample-detector distance in meters.
    :param poni1: distance of PONI 
    """
    detector = pyFAI.detectors.Detector(pixel1=float(pixel1), pixel2=float(pixel2), max_shape=shape, orientation=orientation)
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(float(dist), float(poni1), float(poni2), rot1, rot2, rot3, pixel1, pixel2,
                                                       detector=detector, wavelength=wavelength, orientation=orientation)
    return ai


def save_new_poni(poni_name: Path, dist: float, poni1: float, poni2: float,
             shape: tuple, pixel1: float = 75e-6, pixel2: float = 75e-6,
             wavelength: float = 1.54185, rot1: float = 0, rot2: float = 0, rot3: float = 0, orientation: int = 2):
    ai = new_poni(dist, poni1, poni2, shape, pixel1, pixel2, wavelength, rot1, rot2, rot3, orientation)
    ai.save(poni_name)


def convert_from_poni(poni: np.ndarray, pixel_size: np.ndarray, shape: np.ndarray = None, orientation: int = 2):
    """
    Convert poni to (row, column)

    :param poni: array of distances for the PONI from a corner of the detector (y, x) (in meters)
    :param pixel_size: size of the pixels (y, x) in meters (can just pass a float if pixels are square)
    :param shape: number of rows and columns in detector
    :param orientation: Which corner of the detector (when looking at the sample from behind the detector)
                        is the (0, 0) pixel. 2 and 3 are most common for X-ray detectors.
                        1: Top-left origin
                        2: Top-right origin *
                        3: Bottom-right origin
                        4: Bottom-left origin
    :return: PONI coordinates in pixels 
    """
    poni = np.array(poni)
    pixel_size = np.array(pixel_size)
    if shape is not None:
        shape = np.array(shape)
    poni_pixel = poni / pixel_size
    if orientation == 1:
        poni_pixel = shape - poni_pixel
    elif orientation == 2:
        poni_pixel[0] = shape[0] - poni_pixel[0]
    elif orientation == 4:
        poni_pixel[1] = shape[1] - poni_pixel[1]
    poni_pixel -= 0.5      # shift from corner of pixel to center of pixel
    return poni_pixel

def convert_to_poni(poni_pixel: np.ndarray, pixel_size: np.ndarray, shape: np.ndarray = None, orientation: int = 2):
    """
    Convert (row, column) to PONI

    :param poni: array of distances for the PONI from a corner of the detector (y, x) (in meters)
    :param pixel_size: size of the pixels (y, x) in meters (can just pass a float if pixels are square)
    :param shape: number of rows and columns in detector
    :param orientation: Which corner of the detector (when looking at the sample from behind the detector)
                        is the (0, 0) pixel. 2 and 3 are most common for X-ray detectors.
                        1: Top-left origin
                        2: Top-right origin *
                        3: Bottom-right origin
                        4: Bottom-left origin
    :return: PONI coordinates in meters, from corner of detector
    """
    poni_pixel = np.array(poni_pixel)
    pixel_size = np.array(pixel_size)
    if shape is not None:
        shape = np.array(shape)
    poni = poni_pixel + 0.5     # shift to corner of pixel from center
    if orientation == 1:
        poni = shape - poni
    elif orientation == 2:
        poni[0] = shape[0] - poni[0]
    elif orientation == 3:
        poni[1] = shape[1] - poni[1]
    poni *= pixel_size          # switch from pixel units to meters
    return tuple(poni)


class PoniNudger:
    """
    PoniNudger objects can be used to manually find/adjust the beam center.

    This is best used in a Jupyter Notebook, as exemplified in transform-GIWAXS.ipynb in examples.
    """

    def __init__(self, ai: pyFAI.AzimuthalIntegrator, data: np.ndarray, flat_field: np.ndarray = None,
                 incident_angle: float = 0, radii: list = [12, 5], IM_SIZE: tuple = (6.3, 3)):
        """
        Instantiate PoniNudger obect

        :param ai: starting geometry using pyFAI.AzimuthalIntegrator. This can be generated with `new_poni()`.
        :param data: image file.
        :param flat_field: optional flat field correction file.
        :param incident_angle: grazing incident angle in degrees.
        :param radii: list of lattice plane spacings to represent as rings.
        :param IM_SIZE: the size of the matplotlib plot.
        """
        self.ai = ai
        self.incident_angle = np.radians(float(incident_angle))
        self.tilt_angle = 0
        self.nudge1 = 0
        self.nudge2 = 0
        self.radii = radii
        if flat_field is None:
            self.data = data
        else:
            self.data = data / flat_field
        self.fig = plt.figure(figsize=IM_SIZE)
        self.ax = None
        self.shape = data.shape
        self.show()

    def set_nudge(self, nudge1, nudge2):
        """
        Move the starting PONI by this many pixels.

        :param nudge1: for nudging poni1 (vertical)
        :param nudge2: for nudging poni2 (horizontal)
        """
        self.nudge1 = nudge1
        self.nudge2 = nudge2

    def nudge(self, nudge1, nudge2):
        """
        Move the amount of nudging by this many pixels (relative move).
        This can be difficult to use iteratively in a Jupyter cell.

        :param nudge1: for nudging poni1 (vertical)
        :param nudge2: for nudging poni2 (horizontal)
        """
        self.nudge1 += nudge1
        self.nudge2 += nudge2

    def reset_nudge(self):
        """
        Set the nudge values back to 0 to restore the starting PONI.
        """
        self.nudge1 = 0
        self.nudge2 = 0

    def get_poni(self) -> tuple:
        """
        :return: PONI with current nudging adjustments applied.
        """
        poni1 = self.ai.poni1 + self.ai.pixel1 * self.nudge1
        poni2 = self.ai.poni2 + self.ai.pixel1 * self.nudge2
        print(f"poni1: {poni1}")
        print(f"poni2: {poni2}")
        return poni1, poni2

    def set_incident(self, incident_angle):
        """
        Change the incident angle.

        :param incident_angle: grazing angle of incidence in degrees.
        """
        self.incident_angle = np.radians(incident_angle)

    def set_tilt(self, tilt_angle):
        """
        Change the tilt angle.

        :param tilt_angle: the angle (in degrees) that the sample is rotated relative to the vertical direction on the detector.
        """
        self.tilt_angle = np.radians(tilt_angle)
    
    def get_incident(self):
        """
        :return: the grazing incidence angle in degrees.
        """
        return np.rad2deg(self.incident_angle)
    
    def get_tilt(self):
        """
        :return: the tilt angle in degrees.
        """
        return np.rad2deg(self.tilt)
    
    def show(self, display_max=None):
        """
        Update the plot

        :param display_max: Optionally set a clip on the counts.
        """
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("k")
        if display_max is None:
            display_max = self.data.max()
        pos = self.ax.imshow(self.data, norm=LogNorm(1, display_max))
        self.ax.set_title("Find PONI")
        self.ax.set_xlabel("column (pixels)")
        self.ax.set_ylabel("row (pixels)")
        self.ax.set_xlim(0, self.data.shape[1])
        self.ax.set_ylim(self.data.shape[0], 0)
        self.fig.tight_layout()

        poni1, poni2 = self.get_poni()

        x_pos = poni2 / self.ai.pixel2 - 0.5
        y_pos = self.data.shape[0] - poni1 / self.ai.pixel1 + 0.5
        radii = self.ai.dist / self.ai.pixel1 * np.tan(2. * np.arcsin(0.5 * self.ai.wavelength / np.array(self.radii) * 1e10))

        self.ax.scatter(x_pos, y_pos, s=30, color="r")
        for r in radii:
            x = np.linspace(x_pos - r, x_pos + r, 1000)
            y = np.sqrt(r * r - (x - x_pos) ** 2)
            self.ax.plot(x, y + y_pos, "r", linewidth=0.5)
            self.ax.plot(x, -y + y_pos, "r", linewidth=0.5)
        y = np.linspace(0, self.data.shape[0], 100)
        vertical = np.tan(self.tilt_angle) * (y - y_pos) + x_pos
        self.ax.plot(vertical, y, "r", linewidth=0.5)

        x = np.linspace(0, self.data.shape[1], 100)
        horizontal = -np.tan(self.tilt_angle) * (x - x_pos) + y_pos
        horizon = horizontal - self.ai.dist / self.ai.pixel1 * np.tan(self.incident_angle)
        self.ax.plot(x, horizontal, "r", linewidth=0.5)
        self.ax.plot(x, horizon, "r", linewidth=0.5)

    def save(self, poni_name: Path, orientation: int = 2):
        """
        Once done, this will save the PONI with the nudges applied.

        :param poni_name: the name of the file to save.
        """
        self.ai.poni1 = self.ai.poni1 + self.nudge1 * self.ai.pixel1
        self.ai.poni2 = self.ai.poni2 + self.nudge2 * self.ai.pixel2
        self.ai.detector = pyFAI.detectors.Detector(pixel1=self.ai.pixel1, pixel2=self.ai.pixel2,
                                                    max_shape=self.ai.detector.shape, orientation=orientation)
        print("Saving geometry:")
        print(self.ai)
        if isinstance(poni_name, Path):
            path = poni_name.parent
            name = poni_name.name
            if name[:-5] != ".poni":
                name = name.replace(".", "_") + ".poni"
            poni_name = path / name
        else:   # a string is also a valid input
            if poni_name[:-5] != ".poni":
                poni_name = poni_name(".", "_") + ".poni"
        self.ai.save(poni_name)



