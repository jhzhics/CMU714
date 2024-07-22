#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

/*EwiseFnOp Macro*/
#define EwiseFnOpApi(name)\
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);\
}\

#define EwiseFnOpKernel(name, op)\
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = a[gid] op b[gid];\
}\

#define EwiseFnOp(name, op)\
EwiseFnOpKernel(name, op)\
EwiseFnOpApi(name)

/*ScalarFnOp Macro*/
#define ScalarFnOpApi(name)\
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);\
}

#define ScalarFnOpKernel(name, op)\
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = a[gid] op val;\
}

#define ScalarFnOp(name, op)\
ScalarFnOpKernel(name, op)\
ScalarFnOpApi(name)

/*EwiseFn Macro*/

#define EwiseFnApi(name)\
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);\
}\

#define EwiseFnKernel(name, f)\
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = f(a[gid], b[gid]);\
}\

#define EwiseFn(name, f)\
EwiseFnKernel(name, f)\
EwiseFnApi(name)

/*ScalarFn Macro*/

#define ScalarFnApi(name)\
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);\
}

#define ScalarFnKernel(name, f)\
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = f(a[gid],val);\
}

#define ScalarFn(name, op)\
ScalarFnKernel(name, op)\
ScalarFnApi(name)

/*EwiseFnSingle Macro*/
#define EwiseFnSingleApi(name)\
void Ewise##name(const CudaArray& a, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);\
}\

#define EwiseFnSingleKernel(name, f)\
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size) {\
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
  if (gid < size) out[gid] = f(a[gid]);\
}\

#define EwiseFnSingle(name, f)\
EwiseFnSingleKernel(name, f)\
EwiseFnSingleApi(name)

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
constexpr size_t V = 4;
constexpr size_t L = 64;
constexpr size_t BLOCK_TILE = L / V;
constexpr size_t S_step = 4;
constexpr size_t S = 64;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};
struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    size_t idx = offset;
    size_t left = gid;
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t mod = left % shape.data[i];
      idx += mod * strides.data[i];
      left /= shape.data[i];
    }
    out[gid] = a[idx];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, CudaVec shape,
                  CudaVec strides, size_t offset, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t idx = offset;
    size_t left = gid;
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t mod = left % shape.data[i];
      idx += mod * strides.data[i];
      left /= shape.data[i];
    }
    out[idx] = a[gid];
  }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, VecToCuda(shape),
                                              VecToCuda(strides), offset, a.size);
  
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                   CudaVec strides, size_t offset) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t idx = offset;
    size_t left = gid;
    for (int i = shape.size - 1; i >= 0; i--) {
      size_t mod = left % shape.data[i];
      idx += mod * strides.data[i];
      left /= shape.data[i];
    }
    out[idx] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

EwiseFnOp(Add,+)
ScalarFnOp(Add,+)

EwiseFnOp(Mul,*)
ScalarFnOp(Mul,*)

EwiseFnOp(Div,/)
ScalarFnOp(Div,/)

EwiseFnOp(Eq,==)
ScalarFnOp(Eq,==)

EwiseFnOp(Ge,>=)
ScalarFnOp(Ge,>=)

ScalarFn(Power, powf)

__device__ float device_max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

EwiseFn(Maximum, device_max)
ScalarFn(Maximum, device_max)

EwiseFnSingle(Log, logf)
EwiseFnSingle(Exp, expf)
EwiseFnSingle(Tanh, tanhf)

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void NaiveMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
uint32_t N, uint32_t P)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < M && j < P) {
    scalar_t sum = 0;
    for (int k = 0; k < N; k++) {
      sum += a[i * N + k] * b[k * P + j];
    }
    out[i * P + j] = sum;
  }
}

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* out, uint32_t M,
uint32_t N, uint32_t P)
{
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * V;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * V;

  //assert(blockDim.x == L && blockDim.y == L) 
  /*Shared Memory Prefetch*/

  __shared__ scalar_t a_tile[L][S];
  __shared__ scalar_t b_tile[S][L];
  scalar_t a[V], b[V],c[V][V];
  
  for(int i=0;i<V;i++)
  for(int j=0;j<V;j++)
  {
    c[i][j]=0;
  }

  int Vx_limit = V<M - x?V:M-x;
  int Vy_limit = V<P - y?V:P-y;

  for(int k = 0; k < N; k += S){
    int limit = k + S < N ? k+S : N;
    __syncthreads();

    for(int ki = k + S_step * threadIdx.y
    ; ki < limit; ki++)
    {
      for(int i = 0; i < V; i++){
        int idx_x = (blockIdx.x * BLOCK_TILE + threadIdx.x) * V + i;
        int idx_y = (blockIdx.y * BLOCK_TILE + threadIdx.x) * V + i;
        if (idx_x < M)
        {
          a_tile[i +threadIdx.x*V][ki - k] = A[idx_x * N + ki];
        }
        if (idx_y < P)
        {
          b_tile[ki - k][i+threadIdx.x*V] = B[ki * P + idx_y];
        }
      }
    }
    __syncthreads();

    for(int ki = k; ki < limit; ki++){
      for(int i = 0; i < Vx_limit; i++){
        a[i] = a_tile[i+V*threadIdx.x][ki - k];
      }
      for(int i = 0; i < Vy_limit; i++){
        b[i] = b_tile[ki - k][i+V*threadIdx.y];
      }
      for(int i = 0; i < Vx_limit; i++){
        for(int j = 0; j < Vy_limit; j++){
          c[i][j] += a[i] * b[j];
        }
      }
    }
  }
  
  for(int i = 0; i < Vx_limit; i++){
    for(int j = 0; j < Vy_limit; j++){

      if(x + i < M && y + j < P)
        out[(x + i) * P + y + j] = c[i][j];
    }
  }
}
void SplitSetitem(CudaArray * array_handle, std::vector<CudaArray *>& outs_handle) {
  size_t size = outs_handle[0]->size;
  size_t num_array = outs_handle.size();
  for (int i = 0; i < num_array; i++) {
    cudaMemcpy(outs_handle[i]->ptr,
    array_handle->ptr + i * size,
    size * ELEM_SIZE,
    cudaMemcpyDeviceToDevice);
  }
}


void StackSetitem(const std::vector<CudaArray *>& array_handles, CudaArray* out_handle) {
  size_t size = array_handles[0]->size;
  size_t num_array = array_handles.size();
  for(int i = 0; i < num_array; i++){
    cudaMemcpy(out_handle->ptr + i * size,
    array_handles[i]->ptr,
    size * ELEM_SIZE,
    cudaMemcpyDeviceToDevice);
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 block(BLOCK_TILE, BLOCK_TILE, 1);
  dim3 grid((M + L - 1) / L, (P + L - 1) / L, 1);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  auto cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error in matmul: " << cudaGetErrorString(cuda_err);
    throw std::runtime_error(ss.str());
  }
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t max_val = a[gid * reduce_size];
    for (int i = gid * reduce_size + 1; i < (gid + 1) * reduce_size; i++) {
      max_val = device_max(max_val, a[i]);
    }
    out[gid] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  size_t size = a.size / reduce_size;
  CudaDims dim = CudaOneDim(size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t tot = 0;
    for (int i = gid * reduce_size; i < (gid + 1) * reduce_size; i++) {
      tot += a[i];
    }
    out[gid] = tot;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  size_t size = a.size / reduce_size;
  CudaDims dim = CudaOneDim(size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("stack_setitem", StackSetitem);
  m.def("split_setitem", SplitSetitem);
}
