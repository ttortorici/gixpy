import pyFAI
import fabio
from pathlib import Path
import numpy as np
import copy
import gixpy as gixpy_c


def load(filename: Path):
    return pyFAI.load(filename)


class GIXS:
    def __init__(self, incident_angle_degrees: float, tilt_angle_degrees: float = 0.0, poni_file: Path = None):
        self.incident_angle = np.radians(incident_angle_degrees)
        self.tilt_angle = np.radians(tilt_angle_degrees)
        if poni_file is None:
            self.ai_original = None
            self.dir = None
        else:
            self.load(poni_file)
        self.ai = None
        self.data = None
        self.flat_field = None
        self.mask
        self.header = {}

    def load(self, poni_file: Path):
        self.ai_original = pyFAI.load(poni_file)
        try:
            self.dir = poni_file.parent
        except AttributeError:
            if "/" in poni_file:
                self.dir = Path("/".join(poni_file.split("/")[:-1]))
            elif "\\" in poni_file:
                self.dir = Path("\\".join(poni_file.split("\\")[:-1]))
            else:
                self.dir = Path.cwd()

    def transform(self, image: np.ndarray, flat_field: np.ndarray = None, waveguiding: bool = False, refraction_angle_degrees: float = None):
        if self.ai_original is None:
            raise AttributeError("Must load a poni file first using .load(poni_file: Path)")
        if refraction_angle_degrees is None:
            if waveguiding:
                refraction_angle = self.incident_angle
            else:
                refraction_angle = 0.0
        else:
            refraction_angle = np.radians(refraction_angle_degrees)
        if flat_field is None:
            flat_field = np.ones_like(image)
        self.data, self.flat_field, new_poni = gixpy_c.transform(image, flat_field,
                                                               self.incident_angle,
                                                               refraction_angle,
                                                               self.ai_original.pixel1, self.ai_original.pixel2,
                                                               self.ai_original.poni1, self.ai_original.poni2,
                                                               self.ai_original.dist,
                                                               self.tilt_angle)
        self.ai = copy.deepcopy(self.ai_original)
        self.ai.poni1 = new_poni[0]
        self.ai.poni2 = new_poni[1]
        self.ai.detector = pyFAI.detectors.Detector(pixel1=self.ai.pixel1, pixel2=self.ai.pixel2, max_shape=self.data_t.shape, orientation=2)
        self.ai.save(self.dir / "GIXS.poni")
        self.header = {
            "IncidentAngle": self.incident_angle,
            "TiltAngle": self.tilt_angle,
            "WaveGuiding": waveguiding,
            "RefractionAngle": refraction_angle,
        }
        return self.data, self.flat_field

    def save_edf(self, filename: str, directory_override: Path = None):
        if self.data is None:
            raise AttributeError("Must transform an image first using .transform(image: np.ndarray)")
        if directory_override is None:
            directory = self.dir
        else:
            directory = directory_override
        filename = filename.rstrip(".edf")
        edf_data_obj = fabio.edfimage.EdfImage(data=self.data, header=self.header)
        edf_flat_obj = fabio.edfimage.EdfImage(data=self.flat_field, header=self.header)
        edf_data_obj.write(directory / (filename + "_transformed.edf"))
        edf_flat_obj.write(directory / (filename + "_flat_field_t.edf"))

    def load_mask(self, mask_file: Path = None):
        if self.data is None:
            raise AttributeError("Must transform an image first using .transform(image: np.ndarray)")
        if mask_file is None:
            self.mask = np.logical_not(self.flat_field)
        else:
            mask = fabio.open(mask_file).data.astype(bool)
            self.mask = np.logical_or(mask, np.logical_not(np.logical_not(self.flat_field)))

    def integrate2d(self, q_bins: int = 500, azimuthal_bins: int = 180, radial_range: tuple = None,
                    azimuth_range: tuple = None, unit: str = "q_A^-1"):
        if self.data is None:
            raise AttributeError("Must transform an image first using .transform(image: np.ndarray)")
        if self.mask is None:
            self.load_mask()
        cake = self.ai.integrate2d_ng(
            self.data, q_bins, azimuthal_bins,
            radial_range=None,   # In units specified below
            azimuth_range=None,  # Start from 180 degrees to start from the axis to the right
            mask=self.mask, flat=self.flat,
            error_model="poisson",  unit=unit,
            polarization_factor=None, correctSolidAngle=False,
        )
        return cake

    def integrate1d(self, file_to_save: str = "reduction", q_range: tuple = None, azimuth_range: tuple = None,
                    exposure_time: float = None, q_bins: int = 1000, unit="q_A^-1"):
        if self.data is None:
            raise AttributeError("Must transform an image first using .transform(image: np.ndarray)")
        if self.mask is None:
            self.load_mask()
        if azimuth_range is None:
            azimuth_range = (-180, 0)
        file_to_save += ".edf"
        if (self.dir / file_to_save).is_file():
            (self.dir / file_to_save).unlink()
        if exposure_time is None:
            normalization_factor = 1
        else:
            normalization_factor = exposure_time / 60
        redu = self.ai.integrate1d_ng(
            self.data, q_bins, 
            radial_range=q_range,   # In units specified below
            azimuth_range=azimuth_range,  # Start from 180 degrees to start from the axis to the right
            mask=self.mask, flat=self.flat, error_model="poisson",
            correctSolidAngle=False,
            unit=unit, filename=self.dir / file_to_save, normalization_factor=normalization_factor
        )
        return redu

    def sector(self, file_to_save: str = "sector", q_range: tuple = None,
               azimuth_range: tuple = None, center: float = None, size: float= None,
               exposure_time: float = None,
               q_bins: int = 1000, unit="q_A^-1"):
        if azimuth_range is None and center is None and size is None:
            raise ValueError("Must provide either azimuth_range or center and size")
        if azimuth_range is not None:
            azimuth_range = (-azimuth_range[1], -azimuth_range[0])
        else:
            azimuth_range = (-center - 0.5 * size, -center + 0.5 * size)
        file_to_save += "_({},{})".format(*azimuth_range)
        return self.integrate1d(file_to_save, q_range, azimuth_range, exposure_time, q_bins, unit)