/*******************************************************************************
* Copyright 2014-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
//#define HBW_MALLOC
#ifdef HBW_MALLOC
#include <hbwmalloc.h>
#endif

#include "Geometry.hpp"
#include "SpMP/Utils.hpp"
#include "SpMP/MemoryPool.hpp"

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector & v, local_int_t localLength) {
  v.localLength = localLength;
//#define USE_LARGE_PAGE
#ifdef USE_LARGE_PAGE
  v.values = (double *)malloc_huge_pages(sizeof(double)*localLength);
#elif defined(HBW_MALLOC)
  hbw_posix_memalign((void **)&v.values, 64, sizeof(double)*localLength);
#else
  v.values = (double *)_mm_malloc(sizeof(double)*localLength, 64);
#endif
  v.optimizationData = NULL;
}

inline void SetVector(Vector & v, local_int_t localLength, double *values)
{
  v.localLength = localLength;
  v.values = values;
  v.optimizationData = NULL;
}

inline void InitializeVectorWithMemoryPool(Vector & v, local_int_t localLength) {
  SetVector(
    v, localLength,
    (double *)SpMP::MemoryPool::getSingleton()->allocate(sizeof(double)*localLength));
}

#ifndef HPCG_NO_MPI
struct VectorOptimizationData {
  int id;
  bool persistentCommSetup;
#ifndef HPCG_OFFLOAD
  void *haloRequests;
    // real type is vector<MPI_Request>
    // use void pointer to avoid dependency on MPI
#endif
};

inline void SetHaloCommId(Vector &v, int id)
{
  VectorOptimizationData *optData = new VectorOptimizationData;
  assert(optData);
  optData->id = id;
  optData->persistentCommSetup = false;
#ifndef HPCG_OFFLOAD
  optData->haloRequests = NULL;
#endif
  assert(!v.optimizationData);
  v.optimizationData = optData;
}
#endif // HPCG_NO_MPI

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int iPerThread = (localLength + nthreads - 1)/nthreads;
    int iBegin = std::min(iPerThread*tid, localLength);
    int iEnd = std::min(iBegin + iPerThread, localLength);

    memset((char *)(vv + iBegin), 0, (iEnd - iBegin)*sizeof(double));
  }
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
  double * vv = v.values;
  vv[index] *= value;
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (double)(RAND_MAX) + 1.0;
  return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double * vv = v.values;
  double * wv = w.values;
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];
  return;
}


/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
void DeleteVector(Vector & v);

inline void ReorderVector(Vector& v, Vector& tmp, const int *perm) {
  int l = std::min(v.localLength, tmp.localLength);
  SpMP::reorderVector(v.values, tmp.values, perm, l);
}

inline void ReorderVectorWithInversePerm(Vector& v, Vector& tmp, const int *inversePerm) {
  int l = std::min(v.localLength, tmp.localLength);
  SpMP::reorderVectorWithInversePerm(v.values, tmp.values, inversePerm, l);
}

inline void CorrectnessCheck(
  const Vector& expected, const Vector& actual,
  double tol = 1e-5, double ignoreSmallerThanThis = 0) {
  SpMP::correctnessCheck(
    expected.values, actual.values, expected.localLength, tol, ignoreSmallerThanThis);
}

inline void PrintVector(const Vector& v) {
  for (int i = 0; i < v.localLength; ++i) {
    printf("%g ", v.values[i]);
  }
  printf("\n");
}

#endif // VECTOR_HPP
