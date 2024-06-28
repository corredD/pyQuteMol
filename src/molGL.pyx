cdef extern from "Python.h":
    ctypedef int Py_intptr_t

cdef extern from "windows.h"
    pass
    
cdef extern from "numpy/arrayobject.h":
    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef Py_intptr_t *dimensions
        cdef Py_intptr_t *strides
        cdef object base
        cdef int flags
        cdef object weakreflist
        # descr not implemented yet here...

ctypedef unsigned int GLenum
ctypedef float GLfloat

cdef extern from "windows.h":
    pass
    
cdef extern from "myrand.h":
    float myrand()

cdef extern from "GL/glew.h":
    void glMultiTexCoord2fARB(GLenum target, GLfloat s, GLfloat t)
    void glMultiTexCoord4fARB(GLenum target, GLfloat s, GLfloat t, GLfloat u, GLfloat v)

cdef extern from "GL/gl.h":
    void glVertex2f(GLfloat x, GLfloat y)
    void glVertex3f(GLfloat x, GLfloat y, GLfloat z)
    void glNormal3f(GLfloat nx, GLfloat ny, GLfloat nz)
    void glTexCoord2f(GLfloat s, GLfloat t)
    void glColor3f(GLfloat r, GLfloat g, GLfloat b)


GL_TEXTURE0_ARB = 0x84C0
GL_TEXTURE1_ARB = 0x84C1

import numpy as np
cimport numpy as np
from libc.math cimport pow
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Exception messages
cdef char *conf_error_msg = b"conf must be a sequence of 3 dimensional coordinates"

# Utility functions
cdef inline void raise_conf_error():
    raise ValueError(conf_error_msg.decode())

# Function prototypes
def MolDraw(np.ndarray[np.float32_t, ndim=2] coords,
            np.ndarray[np.float32_t, ndim=1] radii,
            np.ndarray[np.float32_t, ndim=2] textures,
            np.ndarray[np.float32_t, ndim=2] colors,
            np.ndarray[np.float32_t, ndim=1] clipplane,
            np.ndarray[np.int32_t, ndim=1] exclude,
            np.ndarray[np.int32_t, ndim=1] indices):
    cdef int i, index, numindices, numexclude, stride1, stride2, noindices, clipping
    cdef float x, y, z, r
    cdef float *vert, *rad, *col, *tex, *clip
    cdef int *idx, *excl

    vert = <float *> coords.data
    col = <float *> colors.data
    rad = <float *> radii.data
    tex = <float *> textures.data

    if coords.ndim != 2:
        raise_conf_error()

    if coords.shape[1] == 3:
        stride1 = coords.strides[0] // sizeof(float)
        stride2 = coords.strides[1] // sizeof(float)
    else:
        stride1 = coords.strides[1] // sizeof(float)
        stride2 = coords.strides[0] // sizeof(float)

    if indices is None:
        if coords.shape[1] == 3:
            numindices = coords.shape[0]
        else:
            numindices = coords.shape[1]
        noindices = 0
    else:
        numindices = indices.shape[0]
        idx = <int *> indices.data
        noindices = 1

    if not np.allclose(clipplane, 0):
        clipping = 1
        clip = <float *> clipplane.data
        excl = <int *> exclude.data
        numexclude = exclude.shape[0]
    else:
        clipping = 0

    for i in range(numindices):
        if noindices == 0:
            index = i
        else:
            index = idx[i]

        x = vert[(index * stride1) + (stride2 * 0)]
        y = vert[(index * stride1) + (stride2 * 1)]
        z = vert[(index * stride1) + (stride2 * 2)]
        r = rad[index]

        if clipping == 1 and (x * clip[0] + y * clip[1] + z * clip[2] + clip[3]) < 0:
            continue

        glColor3f(col[(index * 3) + 0], col[(index * 3) + 1], col[(index * 3) + 2])
        glTexCoord2f(tex[(index * 2) + 0], tex[(index * 2) + 1])
        glNormal3f(1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, -1, r)
        glVertex3f(x, y, z)
        glNormal3f(1, -1, r)
        glVertex3f(x, y, z)

    if clipping == 1:
        for i in range(numexclude):
            index = excl[i]
            x = vert[(index * stride1) + (stride2 * 0)]
            y = vert[(index * stride1) + (stride2 * 1)]
            z = vert[(index * stride1) + (stride2 * 2)]
            r = rad[index]
            glColor3f(col[(index * 3) + 0], col[(index * 3) + 1], col[(index * 3) + 2])
            glTexCoord2f(tex[(index * 2) + 0], tex[(index * 2) + 1])
            glNormal3f(1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, -1, r)
            glVertex3f(x, y, z)
            glNormal3f(1, -1, r)
            glVertex3f(x, y, z)

def MolDrawShadow(np.ndarray[np.float32_t, ndim=2] coords,
                  np.ndarray[np.float32_t, ndim=1] radii,
                  np.ndarray[np.float32_t, ndim=1] clipplane,
                  np.ndarray[np.int32_t, ndim=1] exclude,
                  np.ndarray[np.int32_t, ndim=1] indices):
    cdef int i, index, numindices, numexclude, stride1, stride2, noindices, clipping
    cdef float x, y, z, r
    cdef float *vert, *rad, *clip
    cdef int *idx, *excl

    vert = <float *> coords.data
    rad = <float *> radii.data

    if coords.ndim != 2:
        raise_conf_error()

    if coords.shape[1] == 3:
        stride1 = coords.strides[0] // sizeof(float)
        stride2 = coords.strides[1] // sizeof(float)
    else:
        stride1 = coords.strides[1] // sizeof(float)
        stride2 = coords.strides[0] // sizeof(float)

    if indices is None:
        numindices = coords.shape[0]
        noindices = 0
    else:
        numindices = indices.shape[0]
        idx = <int *> indices.data
        noindices = 1

    if not np.allclose(clipplane, 0):
        clipping = 1
        clip = <float *> clipplane.data
        excl = <int *> exclude.data
        numexclude = exclude.shape[0]
    else:
        clipping = 0

    for i in range(numindices):
        if noindices == 0:
            index = i
        else:
            index = idx[i]

        x = vert[(index * stride1) + (stride2 * 0)]
        y = vert[(index * stride1) + (stride2 * 1)]
        z = vert[(index * stride1) + (stride2 * 2)]
        r = rad[index]

        if clipping == 1 and (x * clip[0] + y * clip[1] + z * clip[2] + clip[3]) < 0:
            continue

        glNormal3f(1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, -1, r)
        glVertex3f(x, y, z)
        glNormal3f(1, -1, r)
        glVertex3f(x, y, z)

    if clipping == 1:
        for i in range(numexclude):
            index = excl[i]
            x = vert[(index * stride1) + (stride2 * 0)]
            y = vert[(index * stride1) + (stride2 * 1)]
            z = vert[(index * stride1) + (stride2 * 2)]
            r = rad[index]
            glNormal3f(1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, -1, r)
            glVertex3f(x, y, z)
            glNormal3f(1, -1, r)
            glVertex3f(x, y, z)

def MolDrawHalo(np.ndarray[np.float32_t, ndim=2] coords,
                np.ndarray[np.float32_t, ndim=1] radii,
                float halo_size,
                np.ndarray[np.float32_t, ndim=1] clipplane,
                np.ndarray[np.int32_t, ndim=1] exclude,
                np.ndarray[np.int32_t, ndim=1] indices):
    cdef int i, index, numindices, numexclude, stride1, stride2, noindices, clipping
    cdef float x, y, z, r, s
    cdef float *vert, *rad, *clip
    cdef int *idx, *excl

    vert = <float *> coords.data
    rad = <float *> radii.data
    s = halo_size * 2.5

    if coords.ndim != 2:
        raise_conf_error()

    if coords.shape[1] == 3:
        stride1 = coords.strides[0] // sizeof(float)
        stride2 = coords.strides[1] // sizeof(float)
    else:
        stride1 = coords.strides[1] // sizeof(float)
        stride2 = coords.strides[0] // sizeof(float)

    if indices is None:
        if coords.shape[1] == 3:
            numindices = coords.shape[0]
        else:
            numindices = coords.shape[1]
        noindices = 0
    else:
        numindices = indices.shape[0]
        idx = <int *> indices.data
        noindices = 1

    if not np.allclose(clipplane, 0):
        clipping = 1
        clip = <float *> clipplane.data
        excl = <int *> exclude.data
        numexclude = exclude.shape[0]
    else:
        clipping = 0

    for i in range(numindices):
        if noindices == 0:
            index = i
        else:
            index = idx[i]

        x = vert[(index * stride1) + (stride2 * 0)]
        y = vert[(index * stride1) + (stride2 * 1)]
        z = vert[(index * stride1) + (stride2 * 2)]
        r = rad[index]

        if clipping == 1 and (x * clip[0] + y * clip[1] + z * clip[2] + clip[3]) < 0:
            continue

        glNormal3f(1, 1, r + s)
        glVertex3f(x, y, z)
        glNormal3f(-1, 1, r + s)
        glVertex3f(x, y, z)
        glNormal3f(-1, -1, r + s)
        glVertex3f(x, y, z)
        glNormal3f(1, -1, r + s)
        glVertex3f(x, y, z)

    if clipping == 1:
        for i in range(numexclude):
            index = excl[i]
            x = vert[(index * stride1) + (stride2 * 0)]
            y = vert[(index * stride1) + (stride2 * 1)]
            z = vert[(index * stride1) + (stride2 * 2)]
            r = rad[index]
            glNormal3f(1, 1, r + s)
            glVertex3f(x, y, z)
            glNormal3f(-1, 1, r + s)
            glVertex3f(x, y, z)
            glNormal3f(-1, -1, r + s)
            glVertex3f(x, y, z)
            glNormal3f(1, -1, r + s)
            glVertex3f(x, y, z)

def MolDrawOnTexture(np.ndarray[np.float32_t, ndim=2] coords,
                     np.ndarray[np.float32_t, ndim=1] radii,
                     np.ndarray[np.float32_t, ndim=2] textures,
                     np.ndarray[np.float32_t, ndim=2] colors,
                     np.ndarray[np.float32_t, ndim=1] clipplane,
                     np.ndarray[np.int32_t, ndim=1] exclude,
                     np.ndarray[np.int32_t, ndim=1] indices):
    cdef int i, index, numindices, numexclude, stride1, stride2, noindices, clipping
    cdef float x, y, z, r
    cdef float *vert, *rad, *col, *tex, *clip
    cdef int *idx, *excl

    vert = <float *> coords.data
    col = <float *> colors.data
    rad = <float *> radii.data
    tex = <float *> textures.data

    if coords.ndim != 2:
        raise_conf_error()

    if coords.shape[1] == 3:
        stride1 = coords.strides[0] // sizeof(float)
        stride2 = coords.strides[1] // sizeof(float)
    else:
        stride1 = coords.strides[1] // sizeof(float)
        stride2 = coords.strides[0] // sizeof(float)

    if indices is None:
        if coords.shape[1] == 3:
            numindices = coords.shape[0]
        else:
            numindices = coords.shape[1]
        noindices = 0
    else:
        numindices = indices.shape[0]
        idx = <int *> indices.data
        noindices = 1

    if not np.allclose(clipplane, 0):
        clipping = 1
        clip = <float *> clipplane.data
        excl = <int *> exclude.data
        numexclude = exclude.shape[0]
    else:
        clipping = 0

    for i in range(numindices):
        if noindices == 0:
            index = i
        else:
            index = idx[i]

        x = vert[(index * stride1) + (stride2 * 0)]
        y = vert[(index * stride1) + (stride2 * 1)]
        z = vert[(index * stride1) + (stride2 * 2)]
        r = rad[index]

        if clipping == 1 and (x * clip[0] + y * clip[1] + z * clip[2] + clip[3]) < 0:
            continue

        glColor3f(col[(index * 3) + 0], col[(index * 3) + 1], col[(index * 3) + 2])
        glMultiTexCoord2fARB(GL_TEXTURE0_ARB, tex[(index * 2) + 0], tex[(index * 2) + 1])
        glMultiTexCoord2fARB(GL_TEXTURE1_ARB, tex[(index * 2) + 0], tex[(index * 2) + 1])
        glNormal3f(1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, 1, r)
        glVertex3f(x, y, z)
        glNormal3f(-1, -1, r)
        glVertex3f(x, y, z)
        glNormal3f(1, -1, r)
        glVertex3f(x, y, z)

    if clipping == 1:
        for i in range(numexclude):
            index = excl[i]
            x = vert[(index * stride1) + (stride2 * 0)]
            y = vert[(index * stride1) + (stride2 * 1)]
            z = vert[(index * stride1) + (stride2 * 2)]
            r = rad[index]
            glColor3f(col[(index * 3) + 0], col[(index * 3) + 1], col[(index * 3) + 2])
            glMultiTexCoord2fARB(GL_TEXTURE0_ARB, tex[(index * 2) + 0], tex[(index * 2) + 1])
            glMultiTexCoord2fARB(GL_TEXTURE1_ARB, tex[(index * 2) + 0], tex[(index * 2) + 1])
            glNormal3f(1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, 1, r)
            glVertex3f(x, y, z)
            glNormal3f(-1, -1, r)
            glVertex3f(x, y, z)
            glNormal3f(1, -1, r)
            glVertex3f(x, y, z)

def molDrawSticks(np.ndarray[np.float32_t, ndim=2] coords,
                  np.ndarray[np.float32_t, ndim=1] radii,
                  np.ndarray[np.float32_t, ndim=2] colors,
                  np.ndarray[np.float32_t, ndim=1] clipplane,
                  np.ndarray[np.int32_t, ndim=1] exclude,
                  np.ndarray[np.int32_t, ndim=1] indices):
    cdef int i, j, index1, index2, numindices, numexclude, stride1, stride2, noindices, clipping
    cdef float x1, y1, z1, r1
    cdef float x2, y2, z2, r2
    cdef float *vert, *rad, *col, *clip
    cdef int *idx, *excl

    vert = <float *> coords.data
    col = <float *> colors.data
    rad = <float *> radii.data

    if coords.ndim != 2:
        raise_conf_error()

    if coords.shape[1] == 3:
        stride1 = coords.strides[0] // sizeof(float)
        stride2 = coords.strides[1] // sizeof(float)
    else:
        stride1 = coords.strides[1] // sizeof(float)
        stride2 = coords.strides[0] // sizeof(float)

    if indices is None:
        numindices = coords.shape[0] // 2
        noindices = 0
    else:
        numindices = indices.shape[0] // 2
        idx = <int *> indices.data
        noindices = 1

    if not np.allclose(clipplane, 0):
        clipping = 1
        clip = <float *> clipplane.data
        excl = <int *> exclude.data
        numexclude = exclude.shape[0]
    else:
        clipping = 0

    for i in range(numindices):
        if noindices == 0:
            index1 = 2 * i
            index2 = 2 * i + 1
        else:
            index1 = idx[2 * i]
            index2 = idx[2 * i + 1]

        x1 = vert[(index1 * stride1) + (stride2 * 0)]
        y1 = vert[(index1 * stride1) + (stride2 * 1)]
        z1 = vert[(index1 * stride1) + (stride2 * 2)]
        r1 = rad[index1]

        x2 = vert[(index2 * stride1) + (stride2 * 0)]
        y2 = vert[(index2 * stride1) + (stride2 * 1)]
        z2 = vert[(index2 * stride1) + (stride2 * 2)]
        r2 = rad[index2]

        if clipping == 1 and ((x1 * clip[0] + y1 * clip[1] + z1 * clip[2] + clip[3]) < 0 or
                              (x2 * clip[0] + y2 * clip[1] + z2 * clip[2] + clip[3]) < 0):
            continue

        glColor3f(col[(index1 * 3) + 0], col[(index1 * 3) + 1], col[(index1 * 3) + 2])
        glVertex3f(x1, y1, z1)
        glColor3f(col[(index2 * 3) + 0], col[(index2 * 3) + 1], col[(index2 * 3) + 2])
        glVertex3f(x2, y2, z2)

    if clipping == 1:
        for i in range(numexclude):
            index1 = 2 * excl[i]
            index2 = 2 * excl[i] + 1

            x1 = vert[(index1 * stride1) + (stride2 * 0)]
            y1 = vert[(index1 * stride1) + (stride2 * 1)]
            z1 = vert[(index1 * stride1) + (stride2 * 2)]
            r1 = rad[index1]

            x2 = vert[(index2 * stride1) + (stride2 * 0)]
            y2 = vert[(index2 * stride1) + (stride2 * 1)]
            z2 = vert[(index2 * stride1) + (stride2 * 2)]
            r2 = rad[index2]

            glColor3f(col[(index1 * 3) + 0], col[(index1 * 3) + 1], col[(index1 * 3) + 2])
            glVertex3f(x1, y1, z1)
            glColor3f(col[(index2 * 3) + 0], col[(index2 * 3) + 1], col[(index2 * 3) + 2])
            glVertex3f(x2, y2, z2)
