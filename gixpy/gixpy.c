#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

const float DEG2RAD = 0.0174532925199432957692369076848861271344287188854;

/*
 * Implements an example function.
 */
PyDoc_STRVAR(gixpy_example_doc, "example(obj, number)\
\
Example function");

static PyObject* transform(PyObject* self, PyObject* args) {
    PyObject* data_obj;
    PyObject* weight_obj;
    float incident_angle;
    float pixel_size;
    float beam_center_y;
    float beam_center_x;
    float det_dist;
    float tilt_angle;

    if (!PyArg_ParseTuple(args, "OOffffff", &data_obj, &weight_obj, &incident_angle, &pixel_size,
        &beam_center_y, &beam_center_x, &det_dist, &tilt_angle)) {
        return NULL;
    }

    PyArrayObject* data_array = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* weight_array = (PyArrayObject*)PyArray_FROM_OTF(weight_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (data_array == NULL || weight_array == NULL) {
        Py_XDECREF(data_array);
        Py_XDECREF(weight_array);
        return NULL;
    }

    // Get array dimensions
    npy_intp* data_shape = PyArray_SHAPE(data_array);
    npy_intp* weight_shape = PyArray_SHAPE(weight_array);
    size_t rows = data_shape[0];
    size_t columns = data_shape[1];

    // Perform transformations and calculations here
    // ...

    Py_DECREF(data_array);
    Py_DECREF(weight_array);

    // Return transformed data, weight, and beam_center_t
    // ...

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef module_methods[] = {
    {"transform", transform, METH_VARARGS, "Transform data and weight arrays."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "transform_module",
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_transform_module(void) {
    PyObject* module = PyModule_Create(&moduledef);
    import_array();  // Initialize NumPy
    return module;
}

/*
 * List of functions to add to gixpy in exec_gixpy().
 */
static PyMethodDef gixpy_functions[] = {
    { "example", (PyCFunction)gixpy_example, METH_VARARGS | METH_KEYWORDS, gixpy_example_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize gixpy. May be called multiple times, so avoid
 * using static state.
 */
int exec_gixpy(PyObject *module) {
    PyModule_AddFunctions(module, gixpy_functions);

    PyModule_AddStringConstant(module, "__author__", "Teddy");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddIntConstant(module, "year", 2024);

    return 0; /* success */
}

/*
 * Documentation for gixpy.
 */
PyDoc_STRVAR(gixpy_doc, "The gixpy module");


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
    return PyModuleDef_Init(&gixpy_def);
}
