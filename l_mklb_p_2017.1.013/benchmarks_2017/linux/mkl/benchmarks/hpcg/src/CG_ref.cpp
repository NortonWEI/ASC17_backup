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
 @file CG_ref.cpp

 HPCG routine
 */

#include <fstream>
#include <iostream>

#include <cmath>

#include "hpcg.hpp"

#include "CG_ref.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "ComputeWAXPBY_ref.hpp"

using namespace std;

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Reference routine to compute an approximate solution to Ax = b

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

  @see CG()
*/
int CG_ref(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning,
    const HPCG_Params& params) {

  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;


  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

  local_int_t nrow = A.localNumberOfRows;

  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  TICK(); ComputeSPMV_ref(A, p, Ap);  TOCK(t3); // Ap = A*p

  if (params.logLevel >= 4) {
    HPCG_fout << "[" << A.geom->rank << "] SPMV:0 finished." << endl;
  }

  TICK(); ComputeWAXPBY_ref(nrow, 1.0, b, -1.0, Ap, r); TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct_ref(nrow, r, r, normr, t4);  TOCK(t1);

  if (params.logLevel >= 4) {
    HPCG_fout << "[" << A.geom->rank << "] DotProduct:0 finished." << endl;
  }

  normr = sqrt(normr);
  if (params.logLevel >= 2 || params.logLevel >= 1 && A.geom->rank==0) {
    HPCG_fout << "[" << A.geom->rank << "] Initial Residual = "<< normr << std::endl;
  }

#ifdef HPCG_DEBUG
  if (A.geom->rank==0) HPCG_fout << "Initial Residual [ref] = "<< normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  // The looping condition is different from the reference code to support unstructured matrices.
  // For the HPCG matrix, we set tolerance to 0 so that it runs up to max_iter.
  // For unstructured matrices, we set tolerance to some number so that it runs until we reach
  // the tolerance.
  for (int k=1; tolerance == 0 ? k<=max_iter : (k <= max_iter || normr/normr0 > tolerance); k++ ) {
    if (k > 20000) break;
    TICK();
    if (doPreconditioning) {
      ComputeMG_ref(A, r, z, 0, params); // Apply preconditioner

      if (params.logLevel >= 4) {
        HPCG_fout << "[" << A.geom->rank << "] MG:" << k << " finished." << endl;
      }
    }
    else
      ComputeWAXPBY_ref(nrow, 1.0, r, 0.0, r, z); // copy r to z (no preconditioning)
    TOCK(t5); // Preconditioner apply time

    if (k == 1) {
      TICK(); CopyVector(z, p); TOCK(t2); // Copy Mr to p
      TICK(); ComputeDotProduct_ref(nrow, r, z, rtz, t4); TOCK(t1); // rtz = r'*z
    } else {
      oldrtz = rtz;
      TICK(); ComputeDotProduct_ref(nrow, r, z, rtz, t4); TOCK(t1); // rtz = r'*z
      beta = rtz/oldrtz;
      TICK(); ComputeWAXPBY_ref(nrow, 1.0, z, beta, p, p);  TOCK(t2); // p = beta*p + z
    }

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct1:" << k << " finished." << endl;
    }

    TICK(); ComputeSPMV_ref(A, p, Ap); TOCK(t3); // Ap = A*p

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] SpMV:" << k << " finished." << endl;
    }

    TICK(); ComputeDotProduct_ref(nrow, p, Ap, pAp, t4); TOCK(t1); // alpha = p'*Ap

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct2:" << k << " finished." << endl;
    }

    alpha = rtz/pAp;
    TICK(); ComputeWAXPBY_ref(nrow, 1.0, x, alpha, p, x);// x = x + alpha*p
    ComputeWAXPBY_ref(nrow, 1.0, r, -alpha, Ap, r);  TOCK(t2);// r = r - alpha*Ap

    TICK(); ComputeDotProduct_ref(nrow, r, r, normr, t4); TOCK(t1);

    if (params.logLevel >= 4) {
      HPCG_fout << "[" << A.geom->rank << "] DotProduct3:" << k << " finished." << endl;
    }

    normr = sqrt(normr);

#ifdef HPCG_DEBUG
    if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration [ref] = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;

    if (k > 0 && k%100 == 0) {
      std::cout << "Iteration [ref] = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
    }
    if (params.logLevel >= 1 && A.geom->rank == 0) {
      HPCG_fout << "Iteration [ref] = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
    }
  }

  // Store times
  times[1] += t1; // dot product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}

