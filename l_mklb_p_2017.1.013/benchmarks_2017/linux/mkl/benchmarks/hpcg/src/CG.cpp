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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>
#include <iostream>

#include <cmath>

#if !defined(HPCG_NO_MPI) && !defined(HPCG_OFFLOAD)
#include "mpi_hpcg.hpp"
#endif

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "OptimizeProblem.hpp"
#include "AllReduce.hpp"

#include "SpMP/MemoryPool.hpp"

using namespace std;
using SpMP::MemoryPool;

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const SparseMatrix & A, CGData & data, Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning,
    double *mgTimes, double *preSmoothTimes, double *postSmoothTimes, double *spmvTimes,
    double *restrictionTimes, double *prolongationTimes,
    double *haloTimes,
    const HPCG_Params& params) {

  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

  OptimizationData *opt_data = (OptimizationData *)A.optimizationData;
  MemoryPool *memoryPool = MemoryPool::getSingleton();

  {
    Vector tmp;
    size_t poolOffset = memoryPool->getTail();
    InitializeVectorWithMemoryPool(tmp, b.localLength);

    if (opt_data->partialColorOrder) {
      ReorderVector(b, tmp, opt_data->partialColorOrder);
      ReorderVector(x, tmp, opt_data->partialColorOrder);
      ReorderVector(data.r, tmp, opt_data->partialColorOrder);
      ReorderVector(data.z, tmp, opt_data->partialColorOrder);
      ReorderVector(data.p, tmp, opt_data->partialColorOrder);
      ReorderVector(data.Ap, tmp, opt_data->partialColorOrder);
    }

    ReorderVector(b, tmp, opt_data->perm);
    ReorderVector(x, tmp, opt_data->perm);
    ReorderVector(data.r, tmp, opt_data->perm);
    ReorderVector(data.z, tmp, opt_data->perm);
    ReorderVector(data.p, tmp, opt_data->perm);
    ReorderVector(data.Ap, tmp, opt_data->perm);

    memoryPool->setTail(poolOffset);
  }

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  local_int_t nrow = A.localNumberOfRows;
  Vector tempB;
  Vector tempX;
  tempB.values = b.values;
  tempX = x;
  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  ComputeSPMV(A, p, Ap, 0, &t3, haloTimes, params); // Ap = A*p

  if (params.logLevel >= 4) {
    HPCG_fout << "[" << A.geom->rank << "] SPMV:0 finished." << endl;
  }

  TICK(); ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized, opt_data); TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);

  if (params.logLevel >= 4) {
    HPCG_fout << "[" << A.geom->rank << "] DotProduct:0 finished." << endl;
  }

  normr = sqrt(normr);
  if (params.logLevel >= 2 || params.logLevel >= 1 && A.geom->rank==0) {
    HPCG_fout << "[" << A.geom->rank << "] Initial Residual = "<< normr << std::endl;
  }

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  for (int k=1; tolerance == 0 ? k<=max_iter : (k <= max_iter || normr/normr0 > tolerance); k++ ) {
    oldrtz = rtz;

    TICK();
    double localRtz;
    if (doPreconditioning) {
      ComputeMG(
        A, r, z, 0,
        mgTimes, preSmoothTimes, postSmoothTimes, spmvTimes,
        restrictionTimes, prolongationTimes,
        haloTimes,
        params, &localRtz); // Apply preconditioner

      if (params.logLevel >= 4) {
        HPCG_fout << "[" << A.geom->rank << "] MG:" << k << " finished." << endl;
      }
    }
    else
      CopyVector(r, z); // copy r to z (no preconditioning)
    TOCK(t5); // Preconditioner apply time

    if (k == 1) {
      TICK(); CopyVector(z, p); // Copy Mr to p
      TOCK(t2);

      if (doPreconditioning) {
        AllReduce(rtz, localRtz, t4);
      }
      else {
        TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
      }
    } else {
      if (doPreconditioning) {
        AllReduce(rtz, localRtz, t4);
      }
      else {
        TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); // rtz = r'*z
        TOCK(t1);
      }
      beta = rtz/oldrtz;
      TICK(); ComputeWAXPBY (nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized, opt_data);  TOCK(t2); // p = beta*p + z
    }

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct1:" << k << " finished." << endl;
    }

    // TODO: start halo exchange earlier by computing waxpby for boundary first
    double localPAp = ComputeSPMV(A, p, Ap, 0, &t3, haloTimes, params, true /*compute dot product*/); // Ap = A*p

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] SpMV:" << k << " finished." << endl;
    }

    AllReduce(pAp, localPAp, t4);

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct2:" << k << " finished." << endl;
    }

    alpha = rtz/pAp;
    TICK();

    double tempNormr = 0;
#pragma omp parallel for reduction(+:tempNormr)
    for (int i = 0; i < nrow; ++i) {
#if defined(HPCG_NO_MPI) || defined(HPCG_OFFLOAD)
      tempX.values[i] += alpha*p.values[i];
        // don't overlap all reduce with x += alpha*p when we offload
#endif
      r.values[i] -= alpha*Ap.values[i];
      tempNormr += r.values[i]*r.values[i];
    }
    normr = tempNormr;

    TOCK(t2);// r = r - alpha*Ap

#if defined(HPCG_NO_MPI) || defined(HPCG_OFFLOAD)
    AllReduce(normr, tempNormr, t4);
#else
    // overlap all reduce with x += alpha*p
    t4 -= mytimer();

    MPI_Request *allReduceRequest = do_MPI_Iallreduce(normr, tempNormr);
#pragma omp parallel for
    for (int i = 0; i < nrow; ++i) {
      tempX.values[i] += alpha*p.values[i];
    }

    do_MPI_Wait(allReduceRequest);
#endif // HPCG_NO_MPI
    t4 += mytimer();

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct3:" << k << " finished." << endl;
    }

    normr = sqrt(normr);
    niters = k;

    if (A.geom->rank == 0 && k > 0 && k%100 == 0) {
      std::cout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
    }
    if (params.logLevel >= 1 && A.geom->rank == 0 && (k%print_freq == 0 || k == max_iter)) {
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
    }
  }

  {
    Vector tmp;
    size_t poolOffset = memoryPool->getTail();
    InitializeVectorWithMemoryPool(tmp, b.localLength);

    ReorderVector(b, tmp, opt_data->inversePerm);
    ReorderVector(x, tmp, opt_data->inversePerm);
    ReorderVector(data.r, tmp, opt_data->inversePerm);
    ReorderVector(data.z, tmp, opt_data->inversePerm);
    ReorderVector(data.p, tmp, opt_data->inversePerm);
    ReorderVector(data.Ap, tmp, opt_data->inversePerm);

    if (opt_data->inversePartialColorOrder) {
      ReorderVector(b, tmp, opt_data->inversePartialColorOrder);
      ReorderVector(x, tmp, opt_data->inversePartialColorOrder);
      ReorderVector(data.r, tmp, opt_data->inversePartialColorOrder);
      ReorderVector(data.z, tmp, opt_data->inversePartialColorOrder);
      ReorderVector(data.p, tmp, opt_data->inversePartialColorOrder);
      ReorderVector(data.Ap, tmp, opt_data->inversePartialColorOrder);
    }

    memoryPool->setTail(poolOffset);
  }

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}
