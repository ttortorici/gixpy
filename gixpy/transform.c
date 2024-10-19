#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>

// const double DEG2RAD = 0.0174532925199432957692369076848861271344287188854;

static double sign(double x) {
    return (signbit(x) == 0) ? 1 : -1;
}

/*
 * Implements an example function.
 */
PyDoc_STRVAR(gixpy_transform_doc, 
    "transform(image_data, incident_angle_deg, pixel_size_m, beam_center_y_m, beam_center_x_m, det_dist_m, tilt_angle_deg)\n\n"
    "Transform XRD images with forbidden wedge (move pixels to rotate recipricol space vectors into the detector plane). The resulting image will be a different shape to preserve pixel size and detector distance.\n\n"
    "Parameters:\n"
    "image_data : ndarray\n"
    "    Input array of a single image (2D with shape (rows, columns)) or several images (3D with shape (image_num, rows, columns).\n"
    "incident_angle_rad : float\n"
    "    Angle of incidence on the sample (in radians).\n"
    "pixel_size_m : float\n"
    "    Size of a pixel (in meters).\n"
    "poni_y_m : float\n"
    "    Distance from the bottom of the detector to the point of normal incidence (in meters).\n"
    "poni_x_m : float\n"
    "    Distance from the left of the detector to the point of normal incidence (looking at the detector from the sample) (in meters).\n"
    "det_dist_m : float\n"
    "    Distance of the detector from the sample (in meters).\n"
    "yaw_angle_rad : float\n"
    "    Angle the beam is yawed relative to the detector (in radians).\n"
    "pitch_angle_rad : float\n"
    "    Angle the beam is pitched relative to the detector (in radians).\n"
    "tilt_angle_rad : float\n"
    "    Angle the detector is rotated relative to the sample normal (in radians).\n"
    "Returns:\n"
    "transformed_array : ndarray\n"
    "    Resulting array with same dimensionality as input. The rows and columns of the image(s) will change to preserve pixel size.\n"
    "transformed_poni: tuple(float, float).\n"
    "    (y-direction, x-direction) poni (in pixels) from top-left corner (facing detector from sample)."
);

struct Point2D {
    double x;
    double z;
};

struct Point3D {
    double x;
    double z;
    double solid_angle;
};

struct Shape {
    npy_intp rows;
    npy_intp cols;
};

struct PixelInfo {
    int64_t row;
    int64_t col;
    double solid_angle;
    double weight_curr;
    double weight_col_neighbor;
    double weight_row_neighbor;
    double weight_dia_neighbor;
};

struct Geometry {
    double pixel_x;
    double pixel_z;
    double beamcenter_x;
    double beamcenter_z;
    double det_dist;
    double incident_angle;
	double refraction_angle;
    //double yaw_angle;
    //double pitch_angle;
    double tilt_angle;
    int64_t rows;
    int64_t columns;
	//uint8_t orientation;
};

static int move_pixels(double* data_ii, double* flat_ii, double* data_t_ii, double* flat_t_ii,
    struct PixelInfo* pixel_info, struct Point3D* beam_center_t, struct Shape* shape_t, int64_t im_size) {
    double current_pixel_intensity, current_pixel_flat;
    int64_t index_t_prev;
    int64_t index_t_curr = 0;

    for (int64_t px_index = 0; px_index < im_size; ++px_index) {
        index_t_prev = index_t_curr;
        index_t_curr = pixel_info->row * shape_t->cols + pixel_info->col;
        data_t_ii += index_t_curr - index_t_prev;
		flat_t_ii += index_t_curr - index_t_prev;

        current_pixel_intensity = *data_ii * pixel_info->solid_angle;
		current_pixel_flat = *flat_ii;
        // move intensity at current pixel
        *data_t_ii += current_pixel_intensity * pixel_info->weight_curr;
        *flat_t_ii += current_pixel_flat * pixel_info->weight_curr;
        data_t_ii++;                    // move to column neighbor
        flat_t_ii++;

        *data_t_ii += current_pixel_intensity * pixel_info->weight_col_neighbor;
        *flat_t_ii += current_pixel_flat * pixel_info->weight_col_neighbor;
        data_t_ii += shape_t->cols;     // move to diagonal neighbor
        flat_t_ii += shape_t->cols;

        *data_t_ii += current_pixel_intensity * pixel_info->weight_dia_neighbor;
        *flat_t_ii += current_pixel_flat * pixel_info->weight_dia_neighbor;
        data_t_ii--;                    // move to row neighbor
        flat_t_ii--;

        *data_t_ii += current_pixel_intensity * pixel_info->weight_row_neighbor;
        *flat_t_ii += current_pixel_flat * pixel_info->weight_row_neighbor;
        data_t_ii -= shape_t->cols;     // move back to current pixel
        flat_t_ii -= shape_t->cols;

        // move to next pixel in image
        data_ii++;
        flat_ii++;
        pixel_info++;
    }

    return 1;
}


static int calc_pixel_info(struct Point3D* r_ii, struct PixelInfo* pixel_info,
                           struct Point2D* beam_center_t, struct Shape* new_im_shape,
                           int64_t image_size) {
    double x_deto, z_deto, x_floor, z_floor, x_r, x_rc, z_r, z_rc;
    for (int64_t ii = 0; ii < image_size; ++ii) {
        x_deto = r_ii->x + beam_center_t->x;    // shift origin to left of detector
        z_deto = beam_center_t->z - r_ii++->z;  // shift origin to bottom of detector (and move pointer to next in array)
        x_floor = floor(x_deto);
        z_floor = floor(z_deto);
        x_r = x_deto - x_floor;     // Determine how much the intensity spills over to the neighbor to the right
        x_rc = 1. - x_r;            // Determine how much of the intensity stays at this pixel
        z_r = z_deto - z_floor;     // Determine how much the intensity spills over to the neighbor below
        z_rc = 1. - z_r;            // Determine how much of the intensity stays at this pixel
        pixel_info->row = (int64_t)z_floor;  // This is the new row for pixel at (rr, cc)
        pixel_info->col = (int64_t)x_floor;  // This is the new column for pixel at (rr, cc)
		pixel_info->solid_angle = r_ii->solid_angle;
        pixel_info->weight_curr = x_rc * z_rc;          // fraction that stays
        pixel_info->weight_col_neighbor = x_r * z_rc;   // fraction spills to the right
        pixel_info->weight_row_neighbor = x_rc * z_r;   // fraction spills below
        pixel_info++->weight_dia_neighbor = x_r * z_r;  // fraction spills diagonally to the right and below (and move pointer to next in array)
    }
    return 1;
}



/*
 * Calculates the coordinates of where pixels will move in meters and then converts to rows and columns.
 * 
 * Parameters:
 *   r_ii: Pointer to the array of Point2D structures to store the calculated coordinates.
 *   geo: Pointer to the Geometry structure containing the geometric parameters.
 *   new_beam_center: Pointer to the Point2D structure to store the new new beam center.
 *   new_image_shape: Pointer to the Shape structure to store the dimensions of the transformed image.
 * 
 * Returns:
 *   1 if the calculation is successful, 0 otherwise.
 */
static int calc_r(struct Point3D* r_ii, struct Geometry* geo, struct Point2D* new_beam_center, struct Shape* new_image_shape) {
    //struct Point2D* optr = r_ii;
    double sec_2theta, cos_internal, sin_internal, internal_angle, tilt_cos, tilt_sin, x_pos, z_pos, conv_px_x, conv_px_z,
        alpha_scattered, phi_scattered, cos_alpha, cos_phi,
        q_xy, q_xy_sq, q_z, q_sq, q_scaler, r_xy, r_z, max_x, min_z;

    double* x = (double*)malloc(geo->columns * sizeof(double));
    double* z = (double*)malloc(geo->rows * sizeof(double));
    if (x == NULL || z == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory to arrays. This is likely due to not giving a proper data array.");
        return 0;
    }
    for (size_t cc = 0; cc < (size_t)geo->columns; ++cc) {
        x[cc] = (double)cc * geo->pixel_x - geo->beamcenter_x;
    }
    for (size_t rr = 0; rr < (size_t)geo->rows; ++rr) {
        z[rr] = geo->beamcenter_z - (double)rr * geo->pixel_z;
    }
    
    new_beam_center->x = DBL_MAX;
    new_beam_center->z = DBL_MIN;
    max_x = DBL_MIN;
    min_z = DBL_MAX;

	conv_px_x = 1. / geo->pixel_x;
	conv_px_z = 1. / geo->pixel_z;
    internal_angle = geo->incident_angle - geo->refraction_angle;
    cos_internal = cos(internal_angle);
	sin_internal = sin(internal_angle);
    tilt_cos = cos(geo->tilt_angle);
    tilt_sin = sin(geo->tilt_angle);

    for (size_t rr = 0; rr < (size_t)geo->rows; ++rr) {
        for (size_t cc = 0; cc < (size_t)geo->columns; ++cc) {
            x_pos = x[cc] * tilt_cos - z[rr] * tilt_sin;
            z_pos = z[rr] * tilt_cos + x[cc] * tilt_sin;
			sec_2theta = sqrt(x_pos * x_pos + z_pos * z_pos + geo->det_dist * geo->det_dist) / geo->det_dist;
			r_ii->solid_angle = sec_2theta * sec_2theta * sec_2theta;
            alpha_scattered = atan2(z_pos, geo->det_dist);
            phi_scattered = atan2(x_pos * cos(alpha_scattered), geo->det_dist);
			cos_phi = cos(phi_scattered);
            alpha_scattered -= geo->incident_angle;
            cos_alpha = cos(alpha_scattered);
            q_xy_sq = cos_alpha * cos_alpha + cos_internal * cos_internal - 2.0 * cos_internal * cos_alpha * cos_phi;
            q_xy = sqrt(q_xy_sq) * sign(x_pos);
            q_z = sin(alpha_scattered) + sin_internal;
            q_sq = q_xy_sq + q_z * q_z;
			q_scaler = geo->det_dist * sqrt(4 - q_sq) / (2 - q_sq);
            r_xy = q_xy * q_scaler * conv_px_x;
            r_z = q_z * q_scaler * conv_px_z;
            if (r_xy < new_beam_center->x) { new_beam_center->x = r_xy; }
            if (r_z > new_beam_center->z) { new_beam_center->z = r_z; }
            if (r_xy > max_x) { max_x = r_xy; }
            if (r_z < min_z) { min_z = r_z; }
            r_ii->x = r_xy;
            r_ii++->z = r_z;
        }
    }
    new_beam_center->x *= -1;
    new_image_shape->cols = (npy_intp)ceil(new_beam_center->x + max_x) + 1;
    new_image_shape->rows = (npy_intp)ceil(new_beam_center->z - min_z) + 1;
    free(x);
    free(z);
    return 1;
}

static PyObject* transform(PyObject* self, PyObject* args) {
    PyArrayObject* input_data_obj;
    PyArrayObject* input_flat_field;
    double incident_angle, refraction_angle, pixel_z, pixel_x, poni_z, poni_x, det_dist, tilt_angle;

    if (!PyArg_ParseTuple(args, "OOdddddddddd", &input_data_obj, &input_flat_field, &incident_angle, &refraction_angle,
        &pixel_z, &pixel_x, &poni_z, &poni_x, &det_dist, &tilt_angle)) {
        PyErr_SetString(PyExc_ValueError, "The inputs were not parsed.");
        return NULL;
    }

    int64_t rows, columns;
    if (PyArray_Check(input_data_obj) && PyArray_Check(input_flat_field)) {
        if (PyArray_TYPE(input_data_obj) != NPY_DOUBLE || PyArray_TYPE(input_flat_field) != NPY_DOUBLE) {
            PyErr_SetString(PyExc_ValueError, "The data input must be a NumPy array of dtype=np.float64.");
            return NULL;
        }
        if (!PyArray_IS_C_CONTIGUOUS(input_data_obj) || !PyArray_IS_C_CONTIGUOUS(input_flat_field)) {
            PyErr_SetString(PyExc_ValueError, "Input data is not C-contiguous.");
            return NULL;
        }
        if (PyArray_NDIM(input_data_obj) != 2 || PyArray_NDIM(input_flat_field) != 2) {
            PyErr_SetString(PyExc_ValueError, "Input data must be a 2D NumPy array.");
            return NULL;
        }
        size_t* data_shape = PyArray_SHAPE(input_data_obj);
        size_t* flat_field_shape = PyArray_SHAPE(input_flat_field);
        rows = data_shape[0];
        columns = data_shape[1];
        if (rows != flat_field_shape[0] || columns != flat_field_shape[1]) {
            PyErr_SetString(PyExc_ValueError, "Data and flat field must have the same shape.");
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Input must be a NumPy array or a tuple of numpy arrays");
        return NULL;
    }
    int64_t im_size = rows * columns;

    // Points to numpy data
    double* data_array = PyArray_DATA(input_data_obj);
    double* flat_field_array = PyArray_DATA(input_flat_field);

    PySys_WriteStdout("Loaded images.\n");

    // Checks and set up done, now do the calculations

    //d_tan_yaw = det_dist * tan(yaw_angle);
    //d_tan_pitch_sec_yaw = det_dist * tan(pitch_angle) / cos(yaw_angle);

    struct Geometry geometry = {
        .pixel_x = pixel_x,
        .pixel_z = pixel_z,
        .beamcenter_x = poni_x - 0.5 * pixel_x,
        .beamcenter_z = ((double)rows - 0.5) * pixel_z - poni_z,
        .det_dist = det_dist,
        .incident_angle = incident_angle,
        .refraction_angle = refraction_angle,
        .tilt_angle = tilt_angle,
        .rows = rows,
        .columns = columns,
    };

    struct Point3D* r_arr = (struct Point3D*)malloc((im_size) * sizeof(struct Point3D));
    if (r_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for distances for transform.");
        free(data_array);
        free(flat_field_array);
        return NULL;
    }

    struct Point2D beam_center_t;
    struct Shape new_im_shape;
    if (!calc_r(r_arr, &geometry, &beam_center_t, &new_im_shape)) {
        PyErr_SetString(PyExc_ValueError, "Failed to calculate distances for transform\n");
        free(data_array);
        free(flat_field_array);
        free(r_arr);
        return NULL;
    }
    PySys_WriteStdout("Found new locations for pixels.\nOutput images will have shape (%d, %d)\n", new_im_shape.rows, new_im_shape.cols);
    struct Point2D poni_t = {
        .x = (beam_center_t.x + 0.5) * pixel_x,
        .z = (new_im_shape.rows - beam_center_t.z - 0.5) * pixel_z,
    };
    PySys_WriteStdout("Transformed image has PONI (%.6f, %.6f)\n", poni_t.z, poni_t.x);

    struct PixelInfo* pixel_info = malloc((im_size) * sizeof(struct PixelInfo));
    if (pixel_info == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for pixel information.");
        free(data_array);
        free(flat_field_array);
        free(r_arr);
        return NULL;
    }
    
    calc_pixel_info(r_arr, pixel_info, &beam_center_t, &new_im_shape, im_size);

    npy_intp dim[2] = { new_im_shape.rows, new_im_shape.cols };
    PyObject* transformed_array_obj = PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
	double* transformed_data = PyArray_DATA(transformed_array_obj);
    PyObject* transformed_flat_obj = PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
    double* transformed_flat = PyArray_DATA(transformed_flat_obj);
	if (transformed_array_obj == NULL || transformed_flat_obj == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to create output arrays. Likely due to data input being incorrect shape.");
        free(transformed_data);
		free(transformed_flat);
        free(data_array);
        free(flat_field_array);
        free(r_arr);
        free(pixel_info);
        return NULL;
    }

    move_pixels(data_array, flat_field_array, transformed_data, transformed_flat, pixel_info, &beam_center_t, &new_im_shape, im_size);

    PyObject* poni_tuple = PyTuple_Pack(2, PyFloat_FromDouble(poni_t.z), PyFloat_FromDouble(poni_t.x));

    free(r_arr);
    free(pixel_info);
    
    return Py_BuildValue("OOO", transformed_array_obj, transformed_flat_obj, poni_tuple);
}

/*
 * List of functions to add to gixpy in exec_gixpy().
 */
static PyMethodDef gixpy_functions[] = {
    { "transform", (PyCFunction)transform, METH_VARARGS | METH_KEYWORDS, gixpy_transform_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize gixpy. May be called multiple times, so avoid
 * using static state.
 */
int exec_gixpy(PyObject *module) {
    PyModule_AddFunctions(module, gixpy_functions);

    PyModule_AddStringConstant(module, "__author__", "Teddy Tortorici");
    PyModule_AddStringConstant(module, "__version__", "5.6");
    PyModule_AddIntConstant(module, "year", 2024);

    return 0; /* success */
}

/*
 * Documentation for gixpy.
 */
PyDoc_STRVAR(gixpy_doc, 
    "For transforming GIWAXS images to rotate reciprocal space vectors into the detector plane.\n"
    "This introduces a missing/forbidden wedge. The transformation preserves pixel size and detector distance."
);


static PyModuleDef_Slot gixpy_slots[] = {
    { Py_mod_exec, exec_gixpy },
    { 0, NULL }
};

static PyModuleDef gixpy_def = {
    PyModuleDef_HEAD_INIT,
    "gixpy",
    gixpy_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    gixpy_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_transform() {
    PyObject *module = PyModuleDef_Init(&gixpy_def);
    import_array();
    return module;
}

