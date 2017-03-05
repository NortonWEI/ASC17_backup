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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"
#include "OptimizeProblem.hpp"

using namespace std;

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c;
  if (A.optimizationData && ((OptimizationData *)A.optimizationData)->f2cOperator) {
    f2c = ((OptimizationData *)A.optimizationData)->f2cOperator;
  }
  else {
    f2c = A.mgData->f2cOperator;
  }

  local_int_t nc = A.mgData->rc->localLength;

/*#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int nSlices = nc/16;
    int slicesPerThread = (nSlices + nthreads - 1)/nthreads;
    int sliceBegin = min(slicesPerThread*tid, nSlices);
    int sliceEnd = min(sliceBegin + slicesPerThread, nSlices);

    for (local_int_t i = sliceBegin*16; i < sliceEnd*16; i += 16) {
#ifdef ESB_PREFETCH
      PREFETCH_L1(f2c + i + 64);
      PREFETCH_L2(f2c + i + 256);
#endif

      __m512i idx = _mm512_loadunpacklo_epi32(
        _mm512_undefined_epi32(), f2c + i);

      __m512d a = _mm512_i32logather_pd(idx, rfv, _MM_SCALE_8);
      __m512d b = _mm512_i32logather_pd(idx, Axfv, _MM_SCALE_8);

      _mm512_storenrngo_pd(rcv + i, _mm512_sub_pd(a, b));

      idx = _mm512_loadunpacklo_epi32(
        _mm512_undefined_epi32(), f2c + i + 8);

      a = _mm512_i32logather_pd(idx, rfv, _MM_SCALE_8);
      b = _mm512_i32logather_pd(idx, Axfv, _MM_SCALE_8);

      _mm512_storenrngo_pd(rcv + i + 8, _mm512_sub_pd(a, b));
    }
  }
  for (local_int_t i = nc/16*16; i < nc; ++i) {
    rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  }*/

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
#pragma simd
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

  return 0;
}
