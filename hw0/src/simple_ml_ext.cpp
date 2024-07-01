#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    // Allocate memory for the logits and gradients
    int rounds = (m + batch - 1) / batch;
    float *logits = new float[batch * k];
    float *gradients = new float[n * k];

    for(int i = 0; i < rounds; i++)
    {
        memset(logits, 0, sizeof(float) * batch * k);
        memset(gradients, 0, sizeof(float) * n * k);

        int beginx = i * batch;
        int endx = std::min((i + 1) * batch, m);
        int size = endx - beginx;

        for(int j = beginx; j < endx; j++)
        {
            for(int l = 0; l < k; l++)
            {
                for(int p = 0; p < n; p++)
                {
                    logits[(j - beginx) * k + l] += X[j * n + p] * theta[p * k + l];
                }
            }
        }
        for(int idx = 0; idx < batch * k; idx++)
        {
            logits[idx] = exp(logits[idx]);
        }
        for(int idx = 0; idx < batch; idx++)
        {
            float sum = std::accumulate(logits + idx * k, logits + (idx + 1) * k, 0.0);
            for(int l = 0; l < k; l++)
            {
                logits[idx * k + l] /= sum;
            }
        }
        for(int j = beginx; j < endx; j++)
        {
            int pos = y[j];
            logits[(j - beginx) * k + pos] -= 1;
        }
        
        for(int j = beginx; j < endx; j++)
        {
            for(int l = 0; l < k; l++)
            {
                for(int p = 0; p < n; p++)
                {
                    gradients[p * k + l] += X[j * n + p] * logits[(j - beginx) * k + l];
                }
            }
        }
        for(int j = 0; j < n * k; j++)
        {
            theta[j] -= lr * gradients[j] / size;
        }
    }

    delete [] logits;
    delete [] gradients;
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
