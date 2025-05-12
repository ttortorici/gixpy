import gixpy
from time import perf_counter
from pyFAI import detectors
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import numpy as np; import fabio
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib

directory = Path(__file__).parent / "test-data"
file1 = directory / "stitch.edf"
file2 = directory / "flat-field.edf"
data = fabio.open(file1).data.astype(np.float64)
flat = fabio.open(file2).data.astype(np.float64)

pixel = 75e-6
poni1 = 0.013600000000000001
poni2 = 0.11215
det_dist = 0.157160
incident_angle = np.radians(.17)
tilt_angle = np.radians(.18)
critical_angle = np.radians(0.165)

start = perf_counter()
y = gixpy.transform(data, flat, pixel, pixel, poni1, poni2, det_dist, incident_angle, tilt_angle, critical_angle)
end=perf_counter()
print(f"time: {end-start}")

fig1 = plt.figure()
ax1 = plt.subplot()
pos1 = ax1.imshow(y[0], norm=LogNorm(1, np.max(y[0])))
fig1.colorbar(pos1, ax=ax1)
ax1.set_title("Transformed image")

fig2 = plt.figure()
ax2 = plt.subplot()
pos2 = ax2.imshow(y[1])
fig2.colorbar(pos2, ax=ax2)
ax2.set_title("Transformed flat field")

fig3 = plt.figure()
ax3 = plt.subplot()
pos3 = ax3.imshow(y[0]/y[1], norm=LogNorm(1, np.max(y[0])))
fig3.colorbar(pos3, ax=ax3)
ax3.set_title("Transformed image with flat field correction")

ax3.axvline(y[2][1] / pixel - .5, color='r', linewidth=0.5)
ax3.axhline(y[0].shape[0] - y[2][0]/ pixel + 0.5, color='r', linewidth=.5)
detector = detectors.Detector(pixel1=pixel, pixel2=pixel, max_shape=y[0].shape, orientation=2)
ai = AzimuthalIntegrator(
    det_dist,
    y[2][0],
    y[2][1],
    0, 0, 0,
    pixel, pixel,
    detector=detector,
    wavelength=1.54185,
    orientation=2
)

cake = ai.integrate2d_ng(
    y[0],
    500,
    180,
    radial_range=None,
    azimuth_range=None,
    flat=y[1],
    error_model="poisson",
    unit="q_A^-1",
    polarization_factor=None,
    correctSolidAngle=False
)

fig4, ax4 = plt.subplots(1, 1, figsize=(6, 3), facecolor="w")
pos4 = ax4.imshow(cake[0], norm=LogNorm(1, np.max(cake[0])),
                  extent=(np.min(cake[1]), np.max(cake[1]), np.min(cake[2]), np.max(cake[2])), aspect='auto')
ax4.set_xlabel(r"$q\ (\mathregular{\AA}^{-1})$")
ax4.set_ylabel(r"$\psi\ (\degree)$")
ax4.set_yticks(np.arange(0, 181, 30))
ax4.set_ylim(0, 180)
ax4.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax4.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax4.tick_params(right=True, top=True, which="both")
ax4.set_title("Cake")
fig4.tight_layout()

plt.show()