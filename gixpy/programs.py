import fabio.tifimage
import _gixpy
import gixpy.poni as poni
import numpy as np
import argparse
import fabio
import pyFAI
import json
from pathlib import Path


def transform():
    parser = argparse.ArgumentParser(
        prog="gixpy-transform",
        description="Transform a grazing incidence X-ray scattering image so that a WAXS/SAXS transform will properly put the data into reciprocal space",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>",
    )
    parser.add_argument("filename", help="Path to file containing data.")
    parser.add_argument("poni", help="Either comma-separated poni1,poni2,pixel1,pixel2,distance,(optional)wavelength or path to PONI file (all in meters). If wavelength is not given, 1.5415 Angstrom will be used.")
    parser.add_argument("-P", "--row_column_pixel_index", action="store_true", help="Use row/column index instead of pyFAI convention for beam center (PONI).")
    parser.add_argument("-F", "--flatfield", help="Path to flat field file. Will use array of ones, if not given.")
    parser.add_argument("-C", "--critical_angle", type=float, default=0., help="Critical angle of the film (in degrees). This will optionally correct for refraction.")
    parser.add_argument("-I", "--incident_angle", type=float, default=0., help="Angle of incidence (in degrees). Will set to 0 (small angle approximation), if not given.")
    parser.add_argument("-T", "--tilt_angle", type=float, default=0., help="Angle the sample is tilted relative to the vertical direction (in degrees). Will set to 0, if not given.")
    parser.add_argument("-TIF", "--tiff", action="store_true", help="Save as TIFF and EDF (instead of just EDF).")
    args = parser.parse_args()

    file_path = Path(args.filename).resolve()
    save_dir = file_path.parent / "transformed-files"
    file_obj = fabio.open(file_path)
    print(f"Opening {file_path.as_posix()}")
    print(json.dumps(file_obj.header, indent=2))

    if args.poni[-5:] == ".poni":
        geometry = pyFAI.load(args.poni)
        poni1 = geometry.poni1
        poni2 = geometry.poni2
        pixel1 = geometry.pixel1
        pixel2 = geometry.pixel2
        distance = geometry.dist
        wavelength = geometry.wavelength
    else:
        geometry = args.poni.split(",")
        poni1 = float(geometry[0])
        poni2 = float(geometry[1])
        pixel1 = float(geometry[2])
        pixel2 = float(geometry[3])
        distance = float(geometry[4])
        if len(geometry) > 5:
            wavelength = float(geometry[5])
        else:
            wavelength = 1.54185e-10

    if args.row_column_pixel_index:
        # convert from pixel index to pyFAI format
        pixel1, pixel2 = poni.convert_to(poni_pixel=(poni1, poni2), pixel_size=(pixel1, pixel2), shape=file_obj.data.shape)

    if args.flatfield is None:
        flat_field = np.ones_like(file_obj.data)
    else:
        flat_field = fabio.open(args.flatfield).data.astype(np.float64)
    
    # if args.critial_angle is None:
    #     critical_angle = 0.
    # else:
    #     critical_angle = np.radians(float(args.critical_angle))
    # 
    # if args.incident_angle is None:
    #     incident_angle = 0.
    # else:
    #     incident_angle = np.radians(float(args.incident_angle))
    # 
    # if args.tilt_angle is None:
    #     tilt_angle = 0.
    # else:
    #     tilt_angle = np.radians(float(args.tilt_angle))

    data_transformed, flat_field_transformed, new_poni = _gixpy.transform(
            file_obj.data.astype(np.float64), flat_field, pixel1, pixel2,
            poni1, poni2, distance, np.radians(args.incident_angle),
            np.radians(args.tilt_angle), np.radians(args.critical_angle)
        )

    new_geometry = poni.new(distance, new_poni[0], new_poni[1], data_transformed.shape, pixel1, pixel2, wavelength)
    new_geometry.save(save_dir / "geometry.poni")

    detector = pyFAI.detectors.Detector(pixel1=pixel1, pixel2=pixel2, max_shape=data_transformed.shape, orientation=2)
    detector.save(save_dir / "detector.h5")

    header = {
        "IncidentAngle(deg)": np.rad2deg(incident_angle),
        "TiltAngle(deg)": np.rad2deg(tilt_angle),
        "CriticalAngle(deg)": np.rad2deg(critical_angle)
    }

    edf_data_obj = fabio.edfimage.EdfImage(data=data_transformed, header=header)
    edf_flat_obj = fabio.edfimage.EdfImage(data=flat_field_transformed, header=header)
    edf_data_obj.write(save_dir / (file_path.stem + "_data_transformed.edf"))
    edf_flat_obj.write(save_dir / (file_path.stem + "_flat_field_transformed.edf"))

    if args.tiff:
        tif_data_obj = fabio.tifimage.tifimage(data_transformed)
        tif_flat_obj = fabio.tifimage.tifimage(flat_field_transformed)
        tif_data_obj.write(save_dir / tif_data_obj)
        tif_flat_obj.write(save_dir / tif_flat_obj)
        
