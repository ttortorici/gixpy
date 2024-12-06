#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>

// Return sign of input
static double sign(double x) {
    return (signbit(x) == 0) ? 1 : -1;
}

/*
 * Implements an example function.
 */
PyDoc_STRVAR(_gixpy_transform_doc, 
    "transform(data, flat_field, pixel_z, pixel_x, poni_z, poni_x, detector_distance, incident_angle, tilt_angle, critical_angle)\n"
    "\n"
    "Transform GIWAXS/GISAXS images.\n"
    "\n"
    ":param data: numpy.ndarray - NumPy array of X-ray image data.\n"
    ":param flat_field: numpy.ndarray - An array to keep track of the pixel movement. Give the image's flat-field image data, if it exists, or an array of ones, if not.\n"
    ":param pixel_z: float - The size of a pixel in the z-direction (in meters).\n"
    ":param pixel_x: float - The size of a pixel in the x-direction (in meters).\n"
    ":param poni_z: float - The distance from the bottom edge of the detector to the point of normal incidence (PONI) in the z-direction (in meters).\n"
    ":param poni_x: float - The distance from the left edge of the detector to the point of normal incidence (PONI) in the x-direction (in meters).\n"
    ":param detector_distance: float - The distance from the sample to the detector (in meters).\n"
    ":param incident_angle: float - The angle of incidence on the sample (in radians).\n"
    ":param tilt_angle: float - The angle the detector is tilted relative to the sample normal (in radians).\n"
    ":param critical_angle: float - The critical angle for total external reflection (in radians).\n"
    ":return: (transformed_data : numpy.ndarray, transformed_flat_field : numpy.ndarray, beam_center : tuple)\n"
);

struct Point2D {
    double x;
    double z;
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
    double poni_x;
    double poni_z;
    double det_dist;
    double incident_angle;
    double critical_angle;
    double tilt_angle;
    size_t rows;
    size_t columns;
};

static int move_pixels(double** data_ptr_ii, double** flat_ptr_ii, double** data_t_ptr_ii, double** flat_t_ptr_ii,
    struct PixelInfo* pixel_info, double* solid_angle, struct Point2D* beam_center_t, struct Shape* shape_t, int64_t im_size) {
    double current_pixel_intensity, current_flatfield_value;
    int64_t index_t_prev;
    int64_t index_t_curr = 0;

    for (int64_t px_index = 0; px_index < im_size; ++px_index) {
        // move pointer to new location for this pixel
        index_t_prev = index_t_curr;
        index_t_curr = pixel_info->row * shape_t->cols + pixel_info->col;
        *data_t_ptr_ii += index_t_curr - index_t_prev;
    	*flat_t_ptr_ii += index_t_curr - index_t_prev;

        // value of counts (with solid angle correction) and flat-field at this pixel
        current_pixel_intensity = **data_ptr_ii * (*solid_angle);
		current_flatfield_value = **flat_ptr_ii;

        // Split pixel across neighbors
        **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_curr;
		**flat_t_ptr_ii += current_flatfield_value * pixel_info->weight_curr;
        *data_t_ptr_ii = *data_t_ptr_ii + 1;    // move to column neighbor in transformed image
		*flat_t_ptr_ii = *flat_t_ptr_ii + 1;    // move to column neighbor in transformed image

        **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_col_neighbor;
		**flat_t_ptr_ii += current_flatfield_value * pixel_info->weight_col_neighbor;
        *data_t_ptr_ii += shape_t->cols;    // move to diagonal neighbor in transformed image
		*flat_t_ptr_ii += shape_t->cols;    // move to diagonal neighbor in transformed image

        **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_dia_neighbor;
		**flat_t_ptr_ii += current_flatfield_value * pixel_info->weight_dia_neighbor;
        *data_t_ptr_ii -= 1;                // move to row neighbor in transformed image
		*flat_t_ptr_ii -= 1;                // move to row neighbor in transformed image

        **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_row_neighbor;
		**flat_t_ptr_ii += current_flatfield_value * pixel_info->weight_row_neighbor;
        *data_t_ptr_ii -= shape_t->cols;    // move back to current pixel in transformed image
		*flat_t_ptr_ii -= shape_t->cols;    // move back to current pixel in transformed image

        // move to next pixel in image
        *data_ptr_ii += 1;
		*flat_ptr_ii += 1;
		solid_angle++;
        pixel_info++;
    }
    return 1;
}

static int calc_pixel_info(struct Point2D* r_ii, struct PixelInfo* pixel_info,
    struct Point2D* beam_center, struct Shape* new_im_shape, int64_t image_size) {
    double x_deto, y_deto, x_floor, y_floor, x_r, x_rc, y_r, y_rc;
    for (int64_t ii = 0; ii < image_size; ++ii) {
        // shift from beamcenter as origin to top-left of detector (detector origin)
        x_deto = beam_center->x - r_ii->x;    // shift origin to left of detector
        y_deto = beam_center->z - r_ii++->z;  // shift origin to top of detector (and move pointer to next in array)
        
        // determine anchor pixels by flooring the distance from the detector origin
        x_floor = floor(x_deto);
        y_floor = floor(y_deto);

        // determine how much pixel intensity spills over into neighbors relative to the anchor pixel
        x_r = x_deto - x_floor;     // Determine how much the intensity spills over to the neighbor to the right
        x_rc = 1. - x_r;            // Determine how much of the intensity stays at this pixel
        y_r = y_deto - y_floor;     // Determine how much the intensity spills over to the neighbor below
        y_rc = 1. - y_r;            // Determine how much of the intensity stays at this pixel
        pixel_info->row = (int64_t)y_floor;  // This is the new row for pixel at (rr, cc)
        pixel_info->col = (int64_t)x_floor;  // This is the new column for pixel at (rr, cc)
        pixel_info->weight_curr = x_rc * y_rc;          // fraction that stays
        pixel_info->weight_col_neighbor = x_r * y_rc;   // fraction spills to the right
        pixel_info->weight_row_neighbor = x_rc * y_r;   // fraction spills below
        pixel_info++->weight_dia_neighbor = x_r * y_r;  // fraction spills diagonally to the right and below (and move pointer to next in array)
    }
    return 1;
}

static int calc_r(struct Point2D* r_ii, double* solid_angle, struct Geometry* geo, struct Point2D* new_beam_center, struct Shape* new_image_shape) {
    double beamcenter_x, beamcenter_z, sec_2theta, cos_internal, sin_internal, critical_angle_sq,
        internal_angle, tilt_cos, tilt_sin, x_pos, z_pos, conv_px_x, conv_px_z, alpha, cos_alpha,
        cos_phi, dist_sq, q_y, vert_dist_sq, ray_dist_sq, cos_phi_sq, sin_phi_sq, x_sq,
        cos_int_sq, q_xy, q_xy_sq, q_z, q_sq, r_over_q, r_xy, r_z, min_x, min_z;
    // pyFAI defines the PONI distance as from the bottom-left corner of the bottom left pixel
    // This program defines it as the center of the top-left pixel (the row=0, column=0 pixel)
    // no detector rotations, so beamcenter=PONI
    beamcenter_x = geo->poni_x - 0.5 * geo->pixel_x;
    beamcenter_z = ((double)geo->rows - 0.5) * geo->pixel_z - geo->poni_z;

    // create dynamic arrays for horizontal and vertical directions.
    // x and z are in the lab frame's minus x-direction and minus z-directions respectively
    // origin is the beamcenter/PONI
    double* x = (double*)malloc(geo->columns * sizeof(double));
    double* z = (double*)malloc(geo->rows * sizeof(double));
    if (x == NULL || z == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory to arrays. This is likely due to not giving a proper data array.");
        return 0;
    }
    for (double cc = 0; cc < (double)geo->columns; ++cc) {
        *(x++) = beamcenter_x - cc * geo->pixel_x;
    }
    for (double rr = 0; rr < (double)geo->rows; ++rr) {
        *(z++) = beamcenter_z - (double)rr * geo->pixel_z;
    }
    x -= geo->columns;  // reset pointer position to beginning of array
	z -= geo->rows;     // reset pointer position to beginning of array

    // This seeds the search for locations of minimum and maximum that will be conducted while looping
    new_beam_center->x = DBL_MIN;   // equivalent to max_x
    new_beam_center->z = DBL_MIN;   // equivalent to max_z
    min_x = DBL_MAX;
    min_z = DBL_MAX;

    // multiplying these will convert distances from meters to number of pixels
    conv_px_x = 1. / geo->pixel_x;
    conv_px_z = 1. / geo->pixel_z;

    // determine the angle the beam takes within the sample
    critical_angle_sq = geo->critical_angle * geo->critical_angle;
    internal_angle = sqrt((geo->incident_angle * geo->incident_angle - critical_angle_sq) / (1 - 0.5 * critical_angle_sq));
    cos_internal = cos(internal_angle);
	cos_int_sq = cos_internal * cos_internal;
    sin_internal = sin(internal_angle);

    tilt_cos = cos(geo->tilt_angle);
    tilt_sin = sin(geo->tilt_angle);

    dist_sq = geo->det_dist * geo->det_dist;

    for (size_t rr = 0; rr < geo->rows; ++rr) {
        for (size_t cc = 0; cc < geo->columns; ++cc) {
            // rotate positions based on the tilt angle
            x_pos = x[cc] * tilt_cos - z[rr] * tilt_sin;
            z_pos = z[rr] * tilt_cos + x[cc] * tilt_sin;

            // determine solid angle correction
            sec_2theta = sqrt(x_pos * x_pos + z_pos * z_pos + dist_sq) / geo->det_dist;
            *(solid_angle++) = sec_2theta * sec_2theta * sec_2theta;
			
            // determine scattering angles alpha (polar) and phi (azimuthal)
            alpha = atan2(z_pos, geo->det_dist) - geo->incident_angle;
            x_sq = x_pos * x_pos;
			vert_dist_sq = z_pos * z_pos + dist_sq;
			ray_dist_sq = vert_dist_sq + x_sq;
            cos_phi_sq = vert_dist_sq / ray_dist_sq;
            cos_phi = sqrt(cos_phi_sq);
            sin_phi_sq = x_sq / ray_dist_sq;
            cos_alpha = cos(alpha);
            
            // The following "q" are unitless q' = wavelength * q / (2 * pi)
            q_y = cos_alpha * cos_phi - cos_internal;
            q_z = sin(alpha) * cos_phi + sin_internal;
            q_xy_sq = sin_phi_sq + q_y * q_y;
            q_xy = sqrt(q_xy_sq) * sign(x_pos);
            q_sq = q_xy_sq + q_z * q_z;
            
            // using Bragg condition: r = d * tan(2 * asin(q' / 2))
            // and using trig identity: tan(2 * asin(x / 2)) = x * sqrt(4 - x^2) / (2 - x^2)
            r_over_q = geo->det_dist * sqrt(4 - q_sq) / (2 - q_sq);
            
            // (q_xy / q) = (r_xy / r) = cos(psi)
            // (q_z / q) = (r_z / r) = sin(psi)
            // convert lengths to units of pixels
            r_xy = q_xy * r_over_q * conv_px_x;
            r_z = q_z * r_over_q * conv_px_z;

            // evaluate if this position is a minimum or a maximum
            if (r_xy > new_beam_center->x) { new_beam_center->x = r_xy; }
            if (r_xy < min_x) { min_x = r_xy; }
            if (r_z > new_beam_center->z) { new_beam_center->z = r_z; }
            if (r_z < min_z) { min_z = r_z; }
            r_ii->x = r_xy;
            r_ii++->z = r_z;
        }
    }

    new_image_shape->cols = (npy_intp)ceil(new_beam_center->x - min_x) + 1;
    new_image_shape->rows = (npy_intp)ceil(new_beam_center->z - min_z) + 1;
    PySys_WriteStdout("Transformed beam center: (%.2f, %.2f) pixels\n", new_beam_center->z, new_beam_center->x);
    free(x);
    free(z);
    return 1;
}



static PyObject* transform(PyObject* self, PyObject* args) {
    PyObject* input_data_obj;
    PyObject* input_flat_field_obj;
    PyArrayObject** data_array_obj_ptr;
    PyArrayObject** flat_array_obj_ptr; // pointer to pointers
    double incident_angle, pixel_z, pixel_x, poni_z, poni_x, det_dist, tilt_angle, critical_angle;
    int8_t solid_angle_correction = 1;
    
    if (!PyArg_ParseTuple(args, "OOdddddddd|p", &input_data_obj, &input_flat_field_obj, &pixel_z, &pixel_x,
        &poni_z, &poni_x, &det_dist, &incident_angle, &tilt_angle, &critical_angle, &solid_angle_correction)) {
        PyErr_SetString(PyExc_ValueError, "The inputs were not parsed.");
        return NULL;
    }

	// Check if input is a numpy array
    if (!(PyArray_Check(input_data_obj) && PyArray_Check(input_flat_field_obj))) {
        PyErr_SetString(PyExc_ValueError, "Input must be a NumPy array or a tuple of numpy arrays");
        return NULL;
    }

    // Pull data from numpy arrays
    data_array_obj_ptr = malloc(sizeof(PyArrayObject*));
    if (data_array_obj_ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for data array pointers.");
        return NULL;
    }
    flat_array_obj_ptr = malloc(sizeof(PyArrayObject*));
    if (flat_array_obj_ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for flat field array pointers.");
        free(data_array_obj_ptr);
        return NULL;
    }
    data_array_obj_ptr[0] = (PyArrayObject*)input_data_obj;
    if (PyArray_TYPE(data_array_obj_ptr[0]) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "The data input must be a NumPy array of dtype=np.float64.");
        free(data_array_obj_ptr);
        free(flat_array_obj_ptr);
        return NULL;
    }
    flat_array_obj_ptr[0] = (PyArrayObject*)input_flat_field_obj;
    if (PyArray_TYPE(flat_array_obj_ptr[0]) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "The flat field input must be a NumPy array of dtype=np.float64.");
        free(data_array_obj_ptr);
        free(flat_array_obj_ptr);
        return NULL;
    }

    // Check if data is C-contiguous
    if (!(PyArray_IS_C_CONTIGUOUS(data_array_obj_ptr[0])) && PyArray_IS_C_CONTIGUOUS(flat_array_obj_ptr[0])) {
        PyErr_SetString(PyExc_ValueError, "Input data is not C-contiguous.");
        free(data_array_obj_ptr);
        return NULL;
    }

    // Check size
    size_t rows, columns, im_size;
    size_t ndim = PyArray_NDIM(data_array_obj_ptr[0]);
    size_t* data_shape = PyArray_SHAPE(data_array_obj_ptr[0]);
    if (ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "An element of the data input tuple is not a 2D NumPy Array and should be.");
        free(data_array_obj_ptr);
        free(flat_array_obj_ptr);
        return NULL;
    }
    rows = data_shape[0];
    columns = data_shape[1];
	ndim = PyArray_NDIM(flat_array_obj_ptr[0]);
	data_shape = PyArray_SHAPE(flat_array_obj_ptr[0]);
	if (ndim != 2) {
		PyErr_SetString(PyExc_ValueError, "An element of the flat field input tuple is not a 2D NumPy Array and should be.");
		free(data_array_obj_ptr);
		free(flat_array_obj_ptr);
		return NULL;
	}
	if (data_shape[0] != rows || data_shape[1] != columns) {
		PyErr_SetString(PyExc_ValueError, "The data and flat field inputs must have the same shape.");
		free(data_array_obj_ptr);
		free(flat_array_obj_ptr);
		return NULL;
	}
    im_size = rows * columns;
    
    // Pointer to array of pointers which will point to numpy data
    double** data_array_ptr = (double**)malloc(sizeof(double*));
	if (data_array_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for data array pointers.");
		free(data_array_obj_ptr);
		free(flat_array_obj_ptr);
		return NULL;
	}
	double** flat_array_ptr = (double**)malloc(sizeof(double*));
	if (flat_array_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for flat field array pointers.");
		free(data_array_obj_ptr);
		free(flat_array_obj_ptr);
		free(data_array_ptr);
		return NULL;
	}

    data_array_ptr[0] = PyArray_DATA(data_array_obj_ptr[0]);
	flat_array_ptr[0] = PyArray_DATA(flat_array_obj_ptr[0]);

    PySys_WriteStdout("Loaded image and flat field with shape (%d, %d).\n", rows, columns);
    
    struct Geometry geometry = {
        .poni_x = poni_x,
        .poni_z = poni_z,
		.pixel_x = pixel_x,
		.pixel_z = pixel_z,
        .det_dist = det_dist,
        .incident_angle = incident_angle,
        .tilt_angle = tilt_angle,
        .critical_angle = critical_angle,
        .rows = rows,
        .columns = columns
    };

    struct Point2D* r_arr = (struct Point2D*)malloc(im_size * sizeof(struct Point2D));
    if (r_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for distances for transform.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
        return NULL;
    }
    double* solid_angles = (double*)malloc(im_size * sizeof(double));
    if (solid_angles == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for solid angles for transform.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
		free(r_arr);
		return NULL;
    }

    struct Point2D beam_center_t;
    struct Shape new_im_shape;
    if (!calc_r(r_arr, solid_angles, &geometry, &beam_center_t, &new_im_shape)) {
        PyErr_SetString(PyExc_ValueError, "Failed to calculate distances for transform\n");
        free(data_array_obj_ptr);
        free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
        free(r_arr);
		free(solid_angles);
        return NULL;
    }
    PySys_WriteStdout("Found new locations for pixels.\nOutput images will have shape (%d, %d)\n", new_im_shape.rows, new_im_shape.cols);

    if (!solid_angle_correction) {
        for (size_t ii = 0; ii < im_size; ii++) {
            solid_angles[ii] = 1.;
        }
    }

    struct PixelInfo* pixel_info = (struct PixelInfo*)malloc((im_size) * sizeof(struct PixelInfo));
    if (pixel_info == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for pixel information.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
        free(r_arr);
        free(solid_angles);
        return NULL;
    }
    
    calc_pixel_info(r_arr, pixel_info, &beam_center_t, &new_im_shape, im_size);
    free(r_arr);

    // create pointer to pointers to output arrays
    PyArrayObject** transformed_array_obj_ptr;
    PyArrayObject** transformed_flatf_obj_ptr;
    double** transformed_data_ptr;
	double** transformed_flat_ptr;

    transformed_array_obj_ptr = (PyArrayObject**)malloc(sizeof(PyArrayObject*));
	if (transformed_array_obj_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for transformed array pointers.");
		free(data_array_obj_ptr);
		free(data_array_ptr);
        free(flat_array_obj_ptr);
        free(flat_array_ptr);
		free(pixel_info);
        free(solid_angles);
		return NULL;
	}
	transformed_flatf_obj_ptr = (PyArrayObject**)malloc(sizeof(PyArrayObject*));
	if (transformed_flatf_obj_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for transformed flat field pointers.");
		free(data_array_obj_ptr);
		free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
		free(pixel_info);
		free(solid_angles);
		free(transformed_array_obj_ptr);
		return NULL;
	}

	// initiate transformed data pointers
    transformed_data_ptr = (double**)malloc(sizeof(double*));
	if (transformed_data_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for transformed data pointers.");
		free(data_array_obj_ptr);
		free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
		free(pixel_info);
		free(transformed_array_obj_ptr);
        free(solid_angles);
		free(transformed_flatf_obj_ptr);
		return NULL;
	}
	transformed_flat_ptr = (double**)malloc(sizeof(double*));
	if (transformed_flat_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for transformed flat field pointers.");
		free(data_array_obj_ptr);
		free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
		free(pixel_info);
		free(transformed_array_obj_ptr);
		free(transformed_data_ptr);
		free(solid_angles);
		free(transformed_flatf_obj_ptr);
		return NULL;
	}

	// Create NumPy arrays to store transfomed data
    npy_intp dim[2] = { new_im_shape.rows, new_im_shape.cols };

    PySys_WriteStdout("Initializing transformed image with dimension 2: (%d, %d)\n", new_im_shape.rows, new_im_shape.cols);
	*transformed_array_obj_ptr = (PyArrayObject*)PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);  // transformed image is a pointer derefrenced from the pointer to the pointer
    if (*transformed_array_obj_ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialize NumPy array for the data array. Likely due to data input being incorrect shape.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
        free(pixel_info);
        free(solid_angles);
		free(transformed_array_obj_ptr);
		free(transformed_data_ptr);
		free(transformed_flatf_obj_ptr);
		free(transformed_flat_ptr);
        return NULL;
    }
    *transformed_flatf_obj_ptr = (PyArrayObject*)PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
	if (*transformed_flatf_obj_ptr == NULL) {
		PyErr_SetString(PyExc_ValueError, "Failed to initialize NumPy array for the flat field. Likely due to data input being incorrect shape.");
		free(data_array_obj_ptr);
		free(data_array_ptr);
		free(flat_array_obj_ptr);
		free(flat_array_ptr);
		free(pixel_info);
		free(solid_angles);
		free(transformed_array_obj_ptr);
		free(transformed_data_ptr);
		free(transformed_flatf_obj_ptr);
		free(transformed_flat_ptr);
		return NULL;
	}

	// link data pointers to the data in the NumPy arrays
    *transformed_data_ptr = (double*)PyArray_DATA(*transformed_array_obj_ptr);
	*transformed_flat_ptr = (double*)PyArray_DATA(*transformed_flatf_obj_ptr);

    move_pixels(data_array_ptr, flat_array_ptr, transformed_data_ptr, transformed_flat_ptr, pixel_info, solid_angles, &beam_center_t, &new_im_shape, im_size);
    free(solid_angles);

    PyArrayObject* return_data_array;
	PyArrayObject* return_flat_array;
    return_data_array = (PyArrayObject*)transformed_array_obj_ptr[0];
	return_flat_array = (PyArrayObject*)transformed_flatf_obj_ptr[0];

	poni_x = (beam_center_t.x + 0.5) * pixel_x;
	poni_z = (new_im_shape.rows - 0.5 - beam_center_t.z) * pixel_z;
    
    PyObject* beam_center_tuple = PyTuple_Pack(2, PyFloat_FromDouble(poni_z), PyFloat_FromDouble(poni_x));

    free(pixel_info);
    free(transformed_array_obj_ptr);
    free(transformed_data_ptr);
	free(flat_array_obj_ptr);
	free(flat_array_ptr);
    free(data_array_obj_ptr);
    free(data_array_ptr);
    
    return Py_BuildValue("OOO", return_data_array, return_flat_array, beam_center_tuple);
}


/*
 * List of functions to add to _gixpy in exec__gixpy().
 */
static PyMethodDef _gixpy_functions[] = {
    { "transform", (PyCFunction)transform, METH_VARARGS | METH_KEYWORDS, _gixpy_transform_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};


/*
 * Initialize _gixpy. May be called multiple times, so avoid
 * using static state.
 */
int exec__gixpy(PyObject *module) {
    PyModule_AddFunctions(module, _gixpy_functions);

    PyModule_AddStringConstant(module, "__author__", "Teddy Tortorici");
    PyModule_AddStringConstant(module, "__version__", "2.0");
    PyModule_AddIntConstant(module, "year", 2024);

    return 0; /* success */
}


/*
 * Documentation for _gixpy.
 */
PyDoc_STRVAR(_gixpy_doc, 
    "For transforming GIWAXS images to rotate reciprocal space vectors into the detector plane.\n"
    "This introduces a missing/forbidden wedge. The transformation preserves pixel size and detector distance."
);


static PyModuleDef_Slot _gixpy_slots[] = {
    { Py_mod_exec, exec__gixpy },
    { 0, NULL }
};


static PyModuleDef _gixpy_def = {
    PyModuleDef_HEAD_INIT,
    "_gixpy",
    _gixpy_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    _gixpy_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};


PyMODINIT_FUNC PyInit__gixpy() {
    PyObject *module = PyModuleDef_Init(&_gixpy_def);
    import_array();
    return module;
}

