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
 @file TestSymmetry.cpp

 HPCG routine
 */

#include <fstream>
#include <iostream>
#include <cfloat>
using std::endl;
#include <vector>
#include <cmath>

#include "hpcg.hpp"

#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "TestSymmetry.hpp"
#include "ExchangeHalo.hpp"
#include "offloadExtHpcg.hpp"

/*!
  Tests symmetry-preserving properties of the sparse matrix vector multiply and multi-grid routines.

  @param[in]    geom   The description of the problem's geometry.
  @param[in]    A      The known system matrix
  @param[in]    b      The known right hand side vector
  @param[in]    xexact The exact solution vector
  @param[inout] testsymmetry_data The data structure with the results of the CG symmetry test including pass/fail information

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
  @see ComputeDotProduct_ref
  @see ComputeSPMV
  @see ComputeSPMV_ref
  @see ComputeMG
  @see ComputeMG_ref
*/
int TestSymmetry(SparseMatrix & A, Vector & b, Vector & xexact, TestSymmetryData & testsymmetry_data, const HPCG_Params& params) {

 local_int_t nrow = A.localNumberOfRows;
 local_int_t ncol = A.localNumberOfColumns;

 Vector x_ncol, y_ncol, z_ncol;
#ifdef HPCG_OFFLOAD
 SetVector(x_ncol, ncol, gbl_offload_xncol);
 SetVector(y_ncol, ncol, gbl_offload_yncol);
 SetVector(z_ncol, ncol, gbl_offload_zncol);
#else
 InitializeVector(x_ncol, ncol);
 InitializeVector(y_ncol, ncol);
 InitializeVector(z_ncol, ncol);
#endif
#ifndef HPCG_NO_MPI
 SetHaloCommId(x_ncol, HPCG_OFFLOAD_VECTOR_XNCOL);
 SetHaloCommId(y_ncol, HPCG_OFFLOAD_VECTOR_YNCOL);
 SetHaloCommId(z_ncol, HPCG_OFFLOAD_VECTOR_ZNCOL);
#endif

 Vector tmp;
 InitializeVector(tmp, nrow);
 double t4 = 0.0; // Needed for dot-product call, otherwise unused
 testsymmetry_data.count_fail = 0;

 // Test symmetry of matrix

  // First load vectors with random values
//#define DBG_SYMMETRY
#ifdef DBG_SYMMETRY
  srand(0);
#endif
  FillRandomVector(x_ncol);
  FillRandomVector(y_ncol);

  double xNorm2, yNorm2;
  double ANorm = 2 * 26.0;

  std::vector< double > haloTimes((size_t)HALO_TIMES_END, 0);

  // Next, compute x'*A*y
  ComputeDotProduct(nrow, y_ncol, y_ncol, yNorm2, t4, A.isDotProductOptimized);
  int ierr = ComputeSPMV(A, y_ncol, z_ncol, 0, NULL, &haloTimes[0], params); // z_nrow = A*y_overlap
  if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  double xtAy = 0.0;
  ierr = ComputeDotProduct(nrow, x_ncol, z_ncol, xtAy, t4, A.isDotProductOptimized); // x'*A*y
  if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  // Next, compute y'*A*x
  ComputeDotProduct(nrow, x_ncol, x_ncol, xNorm2, t4, A.isDotProductOptimized);
  ierr = ComputeSPMV(A, x_ncol, z_ncol, 0, NULL, &haloTimes[0], params); // b_computed = A*x_overlap
  if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  double ytAx = 0.0;
  ierr = ComputeDotProduct(nrow, y_ncol, z_ncol, ytAx, t4, A.isDotProductOptimized); // y'*A*x
  if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  testsymmetry_data.depsym_spmv = std::fabs((long double) (xtAy - ytAx))/((xNorm2*ANorm*yNorm2 + yNorm2*ANorm*xNorm2) * (DBL_EPSILON));
  if (testsymmetry_data.depsym_spmv > 1.0) ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
  if (A.geom->rank==0) HPCG_fout << "Departure from symmetry (scaled) for SpMV abs(x'*A*y - y'*A*x) = " << testsymmetry_data.depsym_spmv << endl;

  // Test symmetry of symmetric Gauss-Seidel

  std::vector< double > mgTimes(params.numberOfMgLevels, 0.0);
  std::vector< double > preSmoothTimes(params.numberOfMgLevels, 0.0);
  std::vector< double > postSmoothTimes(params.numberOfMgLevels, 0.0);
  std::vector< double > spmvTimes(params.numberOfMgLevels, 0.0);
  std::vector< double > restrictionTimes(params.numberOfMgLevels, 0.0);
  std::vector< double > prolongationTimes(params.numberOfMgLevels, 0.0);

  // Compute x'*Minv*y
  double dummy;
  ierr = ComputeMG(
    A, y_ncol, z_ncol, 0,
    &mgTimes[0], &preSmoothTimes[0], &postSmoothTimes[0], &spmvTimes[0],
    &restrictionTimes[0], &prolongationTimes[0], &haloTimes[0],
    params, &dummy); // z_ncol = Minv*y_ncol
#ifdef DBG_SYMMETRY
  if (A.nonzerosInRow) {
    // Do not deallocate nonzerosInRow in main.cpp to do this
    printf("Checking correctness of MG\n");

    Vector z_ncol2;
    InitializeVector(z_ncol2, ncol);
    ComputeMG_ref(A, y_ncol, z_ncol2);
    CorrectnessCheck(z_ncol2, z_ncol);
  }
#endif
  if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  double xtMinvy = 0.0;
#ifdef DBG_SYMMETRY
  if (0 == A.geom->rank) {
    /*for (int i = 0; i < y_ncol.localLength; ++i) {
      printf("%g ", y_ncol.values[i]);
    }
    printf("\n");*/
  }
#endif
  ierr = ComputeDotProduct(nrow, x_ncol, z_ncol, xtMinvy, t4, A.isDotProductOptimized); // x'*Minv*y
  if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  // Next, compute z'*Minv*x
  ierr = ComputeMG(
    A, x_ncol, z_ncol, 0,
    &mgTimes[0], &preSmoothTimes[0], &postSmoothTimes[0], &spmvTimes[0],
    &restrictionTimes[0], &prolongationTimes[0], &haloTimes[0],
    params, &dummy); // z_ncol = Minv*x_ncol
  if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  double ytMinvx = 0.0;
  ierr = ComputeDotProduct(nrow, y_ncol, z_ncol, ytMinvx, t4, A.isDotProductOptimized); // y'*Minv*x
  if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  testsymmetry_data.depsym_mg = std::fabs((long double) (xtMinvy - ytMinvx))/((xNorm2*ANorm*yNorm2 + yNorm2*ANorm*xNorm2) * (DBL_EPSILON));
#ifdef DBG_SYMMETRY
  printf(
    "xtMinvy = %g, ytMinvx = %g, xNorm2 = %g, ANorm = %g, yNorm2 = %g, (xtMinvy - ytMinvx) = %g, depsym_mg = %g\n",
    xtMinvy, ytMinvx, xNorm2, ANorm, yNorm2, 
    xtMinvy - ytMinvx,
    testsymmetry_data.depsym_mg);
#endif
  if (testsymmetry_data.depsym_mg > 1.0) ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
  if (A.geom->rank==0) HPCG_fout << "Departure from symmetry (scaled) for MG abs(x'*Minv*y - y'*Minv*x) = " << testsymmetry_data.depsym_mg << endl;

  CopyVector(xexact, x_ncol); // Copy exact answer into overlap vector

  int numberOfCalls = 2;
  double residual = 0.0;
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV(A, x_ncol, z_ncol, 0, NULL, &haloTimes[0], params); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    if ((ierr = ComputeResidual(A.localNumberOfRows, b, z_ncol, residual)))
      HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
    if (A.geom->rank==0) HPCG_fout << "SpMV call [" << i << "] Residual [" << residual << "]" << endl;
  }
  DeleteVector(x_ncol);
  DeleteVector(y_ncol);
  DeleteVector(z_ncol);
  DeleteVector(tmp);

  return 0;
}
