"""
Pythonized version of the calculation of reciprocal space for visualizing the transformation.

Requires:
    - numpy
    - matplotlib

Functions:
    calc_angles(rows: int, columns: int, incident_angle: float, det_dist: float, beamcenter_x: float, beamcenter_z: float) -> tuple:
        Calculate the scattering angles for each pixel in the detector.
    calc_q(incident_angle: float, alpha_scattering: float, phi_scattering: float) -> np.ndarray:
        Calculate the q-vector for given scattering angles.
Main Execution:
    The script calculates and visualizes the reciprocal space transformation for given parameters.
    It uses matplotlib to plot the q-vectors and their contours.
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def calc_angles(rows: int, columns: int, incident_angle: float, det_dist: float,
                beamcenter_x: float, beamcenter_z: float) -> tuple:
    """
    :param rows: number of rows in image
    :param columns: number of columns in image
    :param incident_angle: grazing angle of incidence in radians
    :param det_dist: sample-detector distance in number of pixels
    :param beamcenter_x: point of normal incidence from the left of the detector in pixels
    :param beamcenter_z: point of normal incidence from the top of the detector in pixels
    :return: The two scattering angles
    """
    x = np.arange(columns).reshape(1, columns)
    z = np.arange(rows).reshape(rows, 1)
    x_pos = beamcenter_x - x
    z_pos = beamcenter_z - z
    alpha_scattering = np.arctan2(z_pos, det_dist) - incident_angle
    phi_scattering = np.arcsin(x_pos / np.sqrt(x_pos ** 2 + det_dist ** 2 + z_pos ** 2))
    return alpha_scattering, phi_scattering

def calc_q(incident_angle: float, alpha_scattering: float, phi_scattering: float) -> np.ndarray:
    """
    :param incident_angle: grazing angle of incidence in radians
    :param alpha_scattering: vertical scattering angle (this rotation applied second)
    :param phi_scattering: horizontal scattering angle (this rotation applied first)
    :return q: unitless q: wavelength * q / (2pi)
    """
    q = np.array([
        np.sin(phi_scattering),
        np.cos(phi_scattering) * np.cos(alpha_scattering) - np.cos(incident_angle),
        np.cos(phi_scattering) * np.sin(alpha_scattering) + np.sin(incident_angle),
    ])
    return q

if __name__ == "__main__":
    incident_angle = np.radians(.1)

    shape = (1000, 1000)

    det_dist_px = 1e3
    det_dist_px_zoom = 1e5
    beam_center_x = shape[1] / 2
    beam_center_y = shape[0] - shape[0] / 2.9
    two_alpha_point_waxs = beam_center_y - det_dist_px * np.tan(2*incident_angle)
    two_alpha_point_saxs = beam_center_y - det_dist_px_zoom * np.tan(2*incident_angle)

    alpha_scattering_waxs, phi_scattering_waxs = calc_angles(
        shape[0], shape[1], incident_angle, det_dist_px, beam_center_x, beam_center_y
    )
    q_vector_waxs = calc_q(incident_angle, alpha_scattering_waxs, phi_scattering_waxs)

    alpha_scattering_saxs, phi_scattering_saxs = calc_angles(
        shape[0], shape[1], incident_angle, det_dist_px_zoom, beam_center_x, beam_center_y
    )
    q_vector_saxs = calc_q(incident_angle, alpha_scattering_saxs, phi_scattering_saxs)

    det_dist = (("\\Delta r_\\mathregular{{specular}}\\approx h_\\mathregular{{pixel}}", "\\Delta r_\\mathregular{{specular}}\\approx h_\\mathregular{{pixel}}", "\\Delta r_\\mathregular{{specular}}\\approx h_\\mathregular{{pixel}}"),
                ("\\Delta r_\\mathregular{{specular}}\\approx h_\\mathregular{{pixel}}", "\\Delta r_\\mathregular{{specular}} > h_\\mathregular{{pixel}}", "\\Delta r_\\mathregular{{specular}} > h_\\mathregular{{pixel}}"))
    directions = (("x", "y", "z"),
                  ("xy", "y", "xy"))
    contours = (((-.5,.5,5e-2), (0,2e-1,5e-2), (-.5,.5,5e-2)),
                ((0,.5,5e-2), (-.5,.5,5e-2), (-.5,.5,5e-2)))
    q_xy_waxs = np.sqrt(q_vector_waxs[0] * q_vector_waxs[0] + q_vector_waxs[1] * q_vector_waxs[1])
    q_xy_saxs = np.sqrt(q_vector_saxs[0] * q_vector_saxs[0] + q_vector_saxs[1] * q_vector_saxs[1])

    q_to_plot = np.array([[q_vector_waxs[0], q_vector_waxs[1], q_vector_waxs[2]],
                          [q_xy_waxs, q_vector_saxs[1], q_xy_saxs]])

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.81))

    q_to_plot = np.abs(q_to_plot)

    pad = .05
    plt.subplots_adjust(left=0, bottom=pad, right=1, top=1-pad, wspace=0, hspace=.0)
    for ii, axsr in enumerate(axs):
        for jj, ax in enumerate(axsr):
            pos = ax.imshow(q_to_plot[ii, jj], cmap='viridis')
            ax.contour(q_to_plot[ii, jj], np.arange(*contours[ii][jj]), colors='white', linewidths=1.25)
            ax.contour(q_to_plot[ii, jj], np.array([0]), colors='red', linewidths=1.25)

            if ii:
                ax.set_xlabel("$q_{{{}}}\\ ({})$".format(directions[ii][jj], det_dist[ii][jj]), fontsize=12)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            else:
                ax.set_title("$q_{{{}}}\\ ({})$".format(directions[ii][jj], det_dist[ii][jj]), fontsize=12)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axs[0, 1].plot(beam_center_x, beam_center_y, 'ro', markersize=2)
    axs[1, 0].plot(beam_center_x, beam_center_y, 'ro', markersize=2)
    axs[1, 2].plot(beam_center_x, beam_center_y, 'ro', markersize=2)
    axs[1, 2].plot(beam_center_x, two_alpha_point_saxs, 'ro', markersize=2)
    axs[0, 2].plot(np.ones(shape[1], dtype=np.float64) * beam_center_y, 'red')
    axs[1, 1].contour(q_to_plot[1, 0], np.array([1e-8]), colors='red', linewidths=1.25)
    axs[1, 1].contour(q_vector_saxs[1], np.array([0]), colors='red', linewidths=1.25)

    #fig.tight_layout()
    plt.show()