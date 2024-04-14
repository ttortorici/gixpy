#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

static struct PixelInfo {
    int64_t row;
    int64_t col;
    double weight_curr;
    double weight_col_neighbor;
    double weight_row_neighbor;
    double weight_dia_neighbor;
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
    PyObject* input_data_obj;
    PyArrayObject** data_array_obj_ptr; // pointer to pointers
    double incident_angle, pixel_size, beam_center_y, beam_center_x, det_dist, tilt_angle, to_pixel;
    
    if (!PyArg_ParseTuple(args, "Odddddd", &input_data_obj, &incident_angle, &pixel_size,
        &beam_center_y, &beam_center_x, &det_dist, &tilt_angle)) {
        PyErr_SetString(PyExc_ValueError, "The inputs were not parsed.");
        return NULL;
    }

    int64_t im_num, rows, columns;
    if (PyTuple_Check(input_data_obj) || PyList_Check(input_data_obj)) {
        if (PyTuple_Check(input_data_obj)) { im_num = PyTuple_Size(input_data_obj); }
        else { im_num = PyList_Size(input_data_obj); }

        data_array_obj_ptr = malloc(im_num * sizeof(PyArrayObject*));

        if (PyTuple_Check(input_data_obj)) {
            data_array_obj_ptr[0] = (PyArrayObject*)PyTuple_GetItem(input_data_obj, 0);
        }
        else {
            data_array_obj_ptr[0] = (PyArrayObject*)PyList_GetItem(input_data_obj, 0);
        }
        
        int64_t ndim = PyArray_NDIM(data_array_obj_ptr[0]);
        size_t* data_shape = PyArray_SHAPE(data_array_obj_ptr[0]);
        if (PyArray_TYPE(data_array_obj_ptr[0]) != NPY_DOUBLE) {
            PyErr_SetString(PyExc_ValueError, "The first element of the data input list/tuple is not a NumPy Array and should be.");
            free(data_array_obj_ptr);
            return NULL;
        }
        if (ndim != 2) {
            PyErr_SetString(PyExc_ValueError, "The first element of the data input list/tuple is not a 2D NumPy Array and should be.");
            free(data_array_obj_ptr);
            return NULL;
        }
        rows = data_shape[0];
        columns = data_shape[1];
        for (Py_ssize_t ii = 1; ii < im_num; ++ii) {
            if (PyTuple_Check(input_data_obj)) {
                data_array_obj_ptr[ii] = (PyArrayObject*)PyTuple_GetItem(input_data_obj, ii);
            }
            else {
                data_array_obj_ptr[ii] = (PyArrayObject*)PyList_GetItem(input_data_obj, ii);
            }
            int64_t ndim = PyArray_NDIM(data_array_obj_ptr[ii]);
            size_t* data_shape = PyArray_SHAPE(data_array_obj_ptr[ii]);
            if (PyArray_TYPE(data_array_obj_ptr[ii]) != NPY_DOUBLE) {
                PyErr_SetString(PyExc_ValueError, "Element %d was not dtype=np.float64 and should be.", ii);
                free(data_array_obj_ptr);
                return NULL;
            }
            if (ndim != 2) {
                PyErr_SetString(PyExc_ValueError, "Element %d of the data input list/tuple is not a 2D NumPy Array and should be.", ii);
                free(data_array_obj_ptr);
                return NULL;
            }
            if (data_shape[0] != rows || data_shape[1] != columns) {
                PyErr_SetString(PyExc_ValueError, "All input data in the list/tuple must be the same size.");
                free(data_array_obj_ptr);
                return NULL;
            }
            if (!PyArray_IS_C_CONTIGUOUS(data_array_obj_ptr[ii])) {
                PyErr_SetString(PyExc_ValueError, "Element %d is not C-contiguous.", ii);
                free(data_array_obj_ptr);
                return NULL;
            }
        }
    }
    else if (PyArray_Check(input_data_obj)) {
        im_num = 1;
        data_array_obj_ptr = malloc(sizeof(PyArrayObject*));
        data_array_obj_ptr[0] = (PyArrayObject*)input_data_obj;
        if (PyArray_TYPE(data_array_obj_ptr[0]) != NPY_DOUBLE) {
            PyErr_SetString(PyExc_ValueError, "The data input must be a NumPy array of dtype=np.float64.");
            free(data_array_obj_ptr);
            return NULL;
        }
        if (!PyArray_IS_C_CONTIGUOUS(data_array_obj_ptr[0])) {
            PyErr_SetString(PyExc_ValueError, "Input data is not C-contiguous.");
            free(data_array_obj_ptr);
            return NULL;
        }
        int64_t ndim = PyArray_NDIM(data_array_obj_ptr[0]);
        size_t* data_shape = PyArray_SHAPE(data_array_obj_ptr[0]);
        if (ndim != 2) {
            PyErr_SetString(PyExc_ValueError, "Element %d of the data input tuple is not a 2D NumPy Array and should be.", 0);
            free(data_array_obj_ptr);
            return NULL;
        }
        rows = data_shape[0];
        columns = data_shape[1];
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Input must be a NumPy array or a tuple of numpy arrays");
        return NULL;
    }
    int64_t im_size = rows * columns;
    
    // Pointer to array of pointers which will point to numpy data
    double** data_array_ptr = (double**)malloc(im_num * sizeof(double*));

    for (size_t ii = 0; ii < im_num; ++ii) {
        data_array_ptr[ii] = PyArray_DATA(data_array_obj_ptr[ii]);
    }

    PySys_WriteStdout("Loaded %d images of shape (%d, %d).\n", im_num, rows, columns);

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

    struct Point2D* r_arr = (struct Point2D*)malloc((im_size) * sizeof(struct Point2D));
    if (r_arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for distances for transform.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
        return NULL;
    }

    struct Point2D beam_center_t;
    struct Shape new_im_shape;
    if (!calc_r(r_arr, &geometry, &beam_center_t, &new_im_shape)) {
        PyErr_SetString(PyExc_ValueError, "Failed to calculate distances for transform\n");
        free(data_array_obj_ptr);
        free(data_array_ptr);
        free(r_arr);
        return NULL;
    }
    PySys_WriteStdout("Found new locations for pixels.\nOutput images will have shape (%d, %d)\n", new_im_shape.rows, new_im_shape.cols);

    struct PixelInfo* pixel_info = (struct PixelInfo*)malloc((im_size) * sizeof(struct PixelInfo));
    if (pixel_info == NULL) {
        PyErr_SetString(PyExc_ValueError, "Failed to allocate memory for pixel information.");
        free(data_array_obj_ptr);
        free(data_array_ptr);
        free(r_arr);
        return NULL;
    }
    
    calc_pixel_info(r_arr, pixel_info, &beam_center_t, &new_im_shape, im_size);
    //printf("calculated info\n");

    //printf("(%d, %d)", pixel_info->row, pixel_info->col);

    // create pointer to pointers to output arrays
    PyArrayObject** transformed_array_obj_ptr;
    double** transformed_data_ptr;

    transformed_array_obj_ptr = (PyArrayObject**)malloc(im_num * sizeof(PyArrayObject*));
    transformed_data_ptr = (double**)malloc(im_num * sizeof(double*));

    npy_intp dim[2] = { new_im_shape.rows, new_im_shape.cols };
    for (size_t ii = 0; ii < im_num; ++ii) {
        //printf("%d\n", ii);
        *transformed_array_obj_ptr = (PyArrayObject*)PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
        if (*transformed_array_obj_ptr == NULL) {
            PyErr_Print("Failed to create internal arrays. Likely due to data input being incorrect shape.");
            for (size_t jj = 0; jj < ii; ++jj) {
                Py_DECREF(*(--transformed_array_obj_ptr));
                free(*(--transformed_data_ptr));
            }
            free(data_array_obj_ptr);
            free(data_array_ptr);
            free(r_arr);
            free(pixel_info);
            free(transformed_array_obj_ptr);
            free(transformed_data_ptr);
            return NULL;
        }
        *transformed_data_ptr = (double*)PyArray_DATA(*transformed_array_obj_ptr);
        transformed_data_ptr++;
        transformed_array_obj_ptr++;
    }
    // return pointer to pointers to start position
    transformed_data_ptr -= im_num;
    transformed_array_obj_ptr -= im_num;

    move_pixels(data_array_ptr, transformed_data_ptr, pixel_info, &beam_center_t, &new_im_shape, im_num, im_size);

    PyObject* return_array;
    if (im_num == 1) {
        return_array = (PyArrayObject*)transformed_array_obj_ptr[0];
    }
    else {
        Py_ssize_t num_images = (Py_ssize_t)im_num;
        PyObject* return_list = PyList_New(num_images);
        for (Py_ssize_t ii = 0; ii < num_images; ++ii) {
            PyList_SET_ITEM(return_list, ii, transformed_array_obj_ptr[ii]);
        }
        return_array = PyList_AsTuple(return_list);
    }
    
    PyObject* beam_center_tuple = PyTuple_Pack(2, PyFloat_FromDouble(beam_center_t.y), PyFloat_FromDouble(beam_center_t.x));

    free(r_arr);
    free(pixel_info);
    free(transformed_array_obj_ptr);
    free(transformed_data_ptr);
    free(data_array_obj_ptr);
    free(data_array_ptr);
    
    return Py_BuildValue("OO", return_array, beam_center_tuple);
}

static int move_pixels(double** data_ptr_ii, double** data_t_ptr_ii, struct PixelInfo* pixel_info,
                        struct Point2D* beam_center_t, struct Shape* shape_t, int64_t im_num, int64_t im_size) {
    double current_pixel_intensity;
    int64_t index_t_prev;
    int64_t index_t_curr = 0;

    for (int64_t im_index = 0; im_index < im_num; ++im_index) {
        for (int64_t px_index = 0; px_index < im_size; ++px_index) {
            //printf("%d\n", px_index);
            // move pointer to new location
            //printf("%p\n", *data_t_ptr_ii);
            index_t_prev = index_t_curr;
            index_t_curr = pixel_info->row * shape_t->cols + pixel_info->col;
            //printf("(%d, %d)\n", pixel_info->row, pixel_info->col);
            //printf("%d\n", index_t_curr - index_t_prev);
            *data_t_ptr_ii += index_t_curr - index_t_prev;
            ////printf("%p\n", *data_t_ptr_ii);

            current_pixel_intensity = **data_ptr_ii;
            // move intensity at current pixel
            **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_curr;
            *data_t_ptr_ii = *data_t_ptr_ii + 1;                // move to column neighbor
            
            **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_col_neighbor;
            *data_t_ptr_ii += shape_t->cols;    // move to diagonal neighbor

            **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_dia_neighbor;
            *data_t_ptr_ii -= 1;                // move to row neighbor

            **data_t_ptr_ii += current_pixel_intensity * pixel_info->weight_row_neighbor;
            *data_t_ptr_ii -= shape_t->cols;    // move back to current pixel

            // move to next pixel in image
            *data_ptr_ii += 1;
            pixel_info++;
        }
        pixel_info -= im_size;  // move back to start of image
        data_ptr_ii++;          // move to next image
    }
    return 1;
}

static int calc_pixel_info(struct Point2D* r_ii, struct PixelInfo* pixel_info,
                           struct Point2D* beam_center_t, struct Shape* new_im_shape, int64_t image_size) {
    double x_deto, y_deto, x_floor, y_floor, x_r, x_rc, y_r, y_rc;
    for (int64_t ii = 0; ii < image_size; ++ii) {
        x_deto = beam_center_t->x - r_ii->x;    // shift origin to left of detector
        y_deto = beam_center_t->y - r_ii++->y;  // shift origin to top of detector (and move pointer to next in array)
        x_floor = floor(x_deto);
        y_floor = floor(y_deto);
        x_r = x_deto - x_floor;     // Determine how much the intensity spills over to the neighbor to the right
        x_rc = 1. - x_r;            // Determine how much of the intensity stays at this pixel
        y_r = y_deto - y_floor;     // Determine how much the intensity spills over to the neighbor below
        y_rc = 1. - y_r;            // Determine how much of the intensity stays at this pixel
        //if (ii == 0) { printf("(%.1f,%.1f) --- ", y_floor, x_floor); }
        pixel_info->row = (int64_t)y_floor;  // This is the new row for pixel at (rr, cc)
        pixel_info->col = (int64_t)x_floor;  // This is the new column for pixel at (rr, cc)
        //if (ii == 0) { printf("(%d, %d)\n", pixel_info->row, pixel_info->col); }
        pixel_info->weight_curr = x_rc * y_rc;          // fraction that stays
        pixel_info->weight_col_neighbor = x_r * y_rc;   // fraction spills to the right
        pixel_info->weight_row_neighbor = x_rc * y_r;   // fraction spills below
        pixel_info++->weight_dia_neighbor = x_r * y_r;  // fraction spills diagonally to the right and below (and move pointer to next in array)
    }
    return 1;
}

static int calc_r(struct Point2D* r_ii, struct Geometry* geo, struct Point2D* new_beam_center, struct Shape* new_image_shape) {
    //struct Point2D* optr = r_ii;
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
    PyModule_AddStringConstant(module, "__version__", "1.6");
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

