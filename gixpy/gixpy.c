#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>

const float DEG2RAD = 0.0174532925199432957692369076848861271344287188854;

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
    "incident_angle_deg : float\n"
    "    Angle of incidence on the sample (in degrees).\n"
    "pixel_size_m : float\n"
    "    Size of a pixel (in meters).\n"
    "beam_center_y_m : float\n"
    "    Distance from the bottom of the detector to the beam center (in meters).\n"
    "beam_center_x_m : float\n"
    "    Distance from the left of the detector to the beam center (looking at the detector from the sample) (in meters).\n"
    "det_dist_m : float\n"
    "    Distance of the detector from the sample (in meters).\n"
    "tilt_angle_deg : float\n"
    "    Angle the detector is rotated relative to the sample normal (in degrees).\n"
    "Returns:\n"
    "transformed_array : ndarray\n"
    "    Resulting array with same dimensionality as input. The rows and columns of the image(s) will change to preserve pixel size.\n"
    "transformed_beam_center: tuple(float, float).\n"
    "    (y-direction, x-direction) beam center (in pixels) from top-left corner (facing detector from sample)."
);

static struct Point2D {
    double x;
    double y;
};

static struct Shape {
    npy_intp rows;
    npy_intp cols;
};

static struct Geometry {
    double beam_center_x;
    double beam_center_y;
    double det_dist;
    double incident_angle;
    double tilt_angle;
    int64_t rows;
    int64_t columns;
};

static PyObject* transform(PyObject* self, PyObject* args) {
    // PyObject* data_obj;
    PyArrayObject* data_array_obj;
    double incident_angle, pixel_size, beam_center_y, beam_center_x, det_dist, tilt_angle, to_pixel;
    
    if (!PyArg_ParseTuple(args, "Odddddd", &data_array_obj, &incident_angle, &pixel_size,
        &beam_center_y, &beam_center_x, &det_dist, &tilt_angle)) {
        return NULL;
    }

    double* data_array = PyArray_DATA(data_array_obj);
    int64_t ndim = PyArray_NDIM(data_array_obj);
    size_t* data_shape = PyArray_SHAPE(data_array_obj);

    size_t im_num, rows, columns;

    if (ndim == 2) {
        im_num = 1;
        rows = data_shape[0];
        columns = data_shape[1];
    }
    else if (ndim == 3) {
        im_num = data_shape[0];
        rows = data_shape[1];
        columns = data_shape[2];
    }
    else {
        im_num = 0;
        rows = 0;
        columns = 0;
        PyErr_SetString(PyExc_ValueError, "The data input must be a 2 or 3 dimensional array\n");
        Py_DECREF(data_array_obj);
        return NULL;
    }

    PySys_WriteStdout("Loaded %d-D array of shape (%d, %d, %d).\n", ndim, im_num, rows, columns);

    to_pixel = 1. / pixel_size;
    
    struct Geometry geometry = {
        .beam_center_x = beam_center_x * to_pixel,
        .beam_center_y = (double)rows - beam_center_y * to_pixel,
        .det_dist = det_dist * to_pixel,
        .incident_angle = incident_angle * DEG2RAD,
        .tilt_angle = incident_angle * DEG2RAD,
        .rows = rows,
        .columns = columns
    };

    double* r_arr = (double*)malloc((rows * columns) * sizeof(struct Point2D));
    if (r_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for distances for transform.");
        Py_DECREF(data_array_obj);
        return NULL;
    }

    struct Point2D beam_center_t;
    struct Shape new_im_shape;
    if (!calc_r(r_arr, &geometry, &beam_center_t, &new_im_shape)) {
        PyErr_SetString(PyExc_ValueError, "Failed to calculate distances for transform\n");
        Py_DECREF(data_array_obj);
        free(data_array);
        return NULL;
    }
    PySys_WriteStdout("Found new locations for pixels.\nOutput will be %d-D array of shape (%d, %d, %d).\n", ndim, im_num, new_im_shape.rows, new_im_shape.cols);

    PyArrayObject* transformed_array_obj;
    if (im_num == 1) {
        npy_intp dim[2] = { new_im_shape.rows, new_im_shape.cols };
        transformed_array_obj = PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
    } else {
        npy_intp dim[3] = { im_num, new_im_shape.rows, new_im_shape.cols };
        transformed_array_obj = PyArray_ZEROS(3, dim, NPY_DOUBLE, 0);
    }
    if (transformed_array_obj == NULL) {
        PyErr_Print("Failed to create internal arrays. Likely due to data input being incorrect shape.");
        Py_DECREF(data_array_obj);
        Py_DECREF(transformed_array_obj);
        return NULL;
    }
        
    double* transformed_data = PyArray_DATA(transformed_array_obj);

    move_pixels(data_array, transformed_data, r_arr, &beam_center_t, &new_im_shape, im_num, rows, columns);

    PyObject* return_tuple = PyTuple_New(2);
    PyObject* beam_center_tuple = PyTuple_New(2);
    if (return_tuple == NULL || beam_center_tuple == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python tuple to return values.");
        Py_DECREF(data_array_obj);
        Py_DECREF(transformed_array_obj);
        free(r_arr);
        return NULL;
    }
    PyTuple_SET_ITEM(beam_center_tuple, 0, PyFloat_FromDouble(beam_center_t.y));
    PyTuple_SET_ITEM(beam_center_tuple, 1, PyFloat_FromDouble(beam_center_t.x));
    PyTuple_SET_ITEM(return_tuple, 0, transformed_array_obj);
    PyTuple_SET_ITEM(return_tuple, 1, beam_center_tuple);

    Py_DECREF(data_array_obj);
    free(r_arr);
    return return_tuple;    
}

static int move_pixels(double* data_ii, double* data_t_ii, struct Point2D* r_ii,
                        struct Point2D* beam_center_t, struct Shape* shape_t, size_t im_num, size_t rows, size_t columns) {
    double x_deto, y_deto, x_floor, y_floor, x_r, y_r, x_rc, y_rc, data_at_px;
    double current_pixel_weight, x_neighbor_weight, y_neighbor_weight, diag_neighbor_weight;
    int64_t x_px, y_px, index_t_prev;
    int64_t index_t_curr = 0;
    int64_t px_num = rows * columns;

    for (size_t rr = 0; rr < rows; ++rr) {
        for (size_t cc = 0; cc < columns; ++cc) {
            x_deto = beam_center_t->x - r_ii->x;
            y_deto = beam_center_t->y - r_ii++->y;
            x_floor = floor(x_deto);
            y_floor = floor(y_deto);
            x_px = x_floor;
            y_px = y_floor;
            x_r = x_deto - x_floor;
            x_rc = 1. - x_r;
            y_r = y_deto - y_floor;
            y_rc = 1. - y_r;
            current_pixel_weight = x_rc * y_rc;
            x_neighbor_weight = x_r * y_rc;
            y_neighbor_weight = x_rc * y_r;
            diag_neighbor_weight = x_r * y_r;

            index_t_prev = index_t_curr;
            index_t_curr = y_px * shape_t->cols + x_px;
            data_t_ii += index_t_curr - index_t_prev;       // move t pointer based on new pixel location
            data_at_px = *data_ii++;                        // get data at current pixel and move pointer forward
            if (data_at_px > 0.1) {
                int64_t ii;
                for (ii = 0; ii < im_num; ++ii) {
                    *data_t_ii++ += data_at_px * current_pixel_weight;
                    *data_t_ii += data_at_px * x_neighbor_weight;
                    data_t_ii += shape_t->cols;
                    *data_t_ii-- += data_at_px * diag_neighbor_weight;
                    *data_t_ii += data_at_px * y_neighbor_weight;
                    data_t_ii -= shape_t->cols;
                    data_t_ii += px_num;
                }
                data_t_ii -= px_num * ii;
            }
        }
    }
    return 1;
}

static int calc_r(struct Point2D* r_ii, struct Geometry* geo, struct Point2D* new_beam_center, struct Shape* new_image_shape) {
    struct Point2D* optr = r_ii;
    double cos_incident, sin_incident;

    cos_incident = cos(geo->incident_angle);
    sin_incident = sin(geo->incident_angle);

    double* x = (double*)malloc(geo->columns * sizeof(double));
    double* y = (double*)malloc(geo->rows * sizeof(double));
    if (x== NULL || y == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory to arrays. This is likely due to not giving a proper data array.");
        return 0;
    }

    for (size_t cc = 0; cc < geo->columns; ++cc) {
        x[cc] = geo->beam_center_x - (double)cc;
    }
    for (size_t rr = 0; rr < geo->rows; ++rr) {
        y[rr] = geo->beam_center_y - (double)rr;
    }

    double tilt_cos, tilt_sin, x_pos, y_pos, hori_travel_sq, alpha_scattered, sin_phi, cos_alpha, cos_phi;
    double q_xy, q_xy_sq, q_z, q_z_sq, q_sq, q_scaler, r_xy, r_z;
    new_beam_center->x = DBL_MIN;
    new_beam_center->y = DBL_MIN;
    double min_x = DBL_MAX;
    double min_y = DBL_MAX;
    double det_dist_sq = geo->det_dist * geo->det_dist;
    //int64_t last_index;
    //int64_t index = 0;

    if (geo->tilt_angle != 0.0) {
        tilt_cos = cos(geo->tilt_angle);
        tilt_sin = sin(geo->tilt_angle);
        for (size_t rr = 0; rr < geo->rows; ++rr) {
            for (size_t cc = 0; cc < geo->columns; ++cc) {
                x_pos = x[cc] * tilt_cos - y[rr] * tilt_sin;
                y_pos = y[rr] * tilt_cos + x[cc] * tilt_sin;
                hori_travel_sq = det_dist_sq + x_pos * x_pos;
                sin_phi = x_pos / sqrt(hori_travel_sq);
                cos_phi = sqrt(1 - sin_phi * sin_phi);
                alpha_scattered = asin(y_pos / sqrt(hori_travel_sq + y_pos * y_pos)) - geo->incident_angle;
                cos_alpha = cos(alpha_scattered);
                q_xy_sq = cos_alpha * cos_alpha + cos_incident * cos_incident - 2.0 * cos_incident * cos_alpha * cos_phi;
                q_xy = sqrt(q_xy_sq) * sign(x_pos);
                q_z = sin(alpha_scattered) + sin(geo->incident_angle);
                q_z_sq = q_z * q_z;
                q_sq = q_xy_sq + q_z_sq;
                q_scaler = geo->det_dist * sqrt(0.5 + 1.0 / (2.0 - q_sq));
                r_xy = q_xy * q_scaler;
                r_z = q_z * q_scaler;
                if (r_xy > new_beam_center->x) { new_beam_center->x = r_xy; }
                if (r_z > new_beam_center->y) { new_beam_center->y = r_z; }
                if (r_xy < min_x) { min_x = r_xy; }
                if (r_z < min_y) { min_y = r_z; }
                r_ii->x = r_xy;
                r_ii++->y = r_z;
            }
        }
    } else {
        for (size_t cc = 0; cc < geo->columns; ++cc) {
            x_pos = x[cc];
            hori_travel_sq = det_dist_sq + x_pos * x_pos;
            sin_phi = x_pos / sqrt(hori_travel_sq);
            cos_phi = sqrt(1 - sin_phi * sin_phi);
            for (size_t rr = 0; rr < geo->rows; ++rr) {
                y_pos = y[rr];
                alpha_scattered = asin(y_pos / sqrt(hori_travel_sq + y_pos * y_pos)) - geo->incident_angle;
                cos_alpha = cos(alpha_scattered);
                q_xy_sq = cos_alpha * cos_alpha + cos_incident * cos_incident - 2.0 * cos_incident * cos_alpha * cos_phi;
                q_xy = sqrt(q_xy_sq) * sign(x_pos);
                q_z = sin(alpha_scattered) + sin(geo->incident_angle);
                q_z_sq = q_z * q_z;
                q_sq = q_xy_sq + q_z_sq;
                q_scaler = geo->det_dist * sqrt(0.5 + 1.0 / (2.0 - q_sq));
                r_xy = q_xy * q_scaler;
                r_z = q_z * q_scaler;
                if (r_xy > new_beam_center->x) { new_beam_center->x = r_xy; }
                if (r_z > new_beam_center->y) { new_beam_center->y = r_z; }
                if (r_xy < min_x) { min_x = r_xy; }
                if (r_z < min_y) { min_y = r_z; }
                r_ii->x = r_xy;
                r_ii->y = r_z;
                r_ii += geo->columns;
            }
            r_ii += 1 - geo->columns;
        }
    }
    new_image_shape->cols = ceil(new_beam_center->x - min_x) + 1;
    new_image_shape->rows = ceil(new_beam_center->y - min_y) + 1;
    free(x);
    free(y);
    return 1;
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
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
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

PyMODINIT_FUNC PyInit_gixpy() {
    PyObject *module = PyModuleDef_Init(&gixpy_def);
    import_array();
    return module;
}

