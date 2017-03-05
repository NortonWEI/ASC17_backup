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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_OFFLOAD
#ifndef HPCG_NO_MPI
#include "mpi_hpcg.hpp"
#endif
#else
#include "offloadExtHpcgLib.hpp"
#endif

#include <cstdio>
#include <string>

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "GenerateProblem.hpp"
#include "GenerateProblem_ref.hpp"

#include "SpMP/MemoryPool.hpp"

using namespace std;
using namespace SpMP;

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(
  SparseMatrix & A, Vector * b, Vector * x, Vector * xexact, const HPCG_Params& params) {

  // The call to this reference version of GenerateProblem can be replaced with custom code.
  // However, the data structures must remain unchanged such that the CheckProblem function is satisfied.
  // Furthermore, any code must work for general unstructured sparse matrices.  Special knowledge about the
  // specific nature of the sparsity pattern may not be explicitly used.
  //return(GenerateProblem_ref(A, b, x, xexact));

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t npx = A.geom->npx;
  global_int_t npy = A.geom->npy;
  global_int_t npz = A.geom->npz;
  global_int_t ipx = A.geom->ipx;
  global_int_t ipy = A.geom->ipy;
  global_int_t ipz = A.geom->ipz;
  global_int_t gnx = nx*npx;
  global_int_t gny = ny*npy;
  global_int_t gnz = nz*npz;

  local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
  local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

  global_int_t totalNumberOfRows = ((global_int_t) localNumberOfRows)*((global_int_t) A.geom->size); // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)


  // Allocate arrays that are of length localNumberOfRows
  MemoryPool *memoryPool = MemoryPool::getSingleton();
  char *nonzerosInRow =
    memoryPool->allocate<char>(localNumberOfRows);
  global_int_t ** mtxIndG =
    memoryPool->allocate<global_int_t *>(localNumberOfRows);
  local_int_t ** mtxIndL =
    memoryPool->allocate<local_int_t *>(localNumberOfRows);
  double ** matrixValues =
    memoryPool->allocate<double *>(localNumberOfRows);
  double ** matrixDiagonal =
    memoryPool->allocate<double *>(localNumberOfRows);

  double * bv = 0;
  double * xv = 0;
  double * xexactv = 0;
  if (b!=0) bv = b->values; // Only compute exact solution if requested
  if (x!=0) xv = x->values; // Only compute exact solution if requested
  if (xexact!=0) xexactv = xexact->values; // Only compute exact solution if requested
#ifndef NDEBUG
  A.localToGlobalMap = new global_int_t[localNumberOfRows];
#endif

  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }

  mtxIndL[0] = memoryPool->allocate<local_int_t>(
    localNumberOfRows*numberOfNonzerosPerRow);
  matrixValues[0] = memoryPool->allocate<double>(
    localNumberOfRows*numberOfNonzerosPerRow);
  mtxIndG[0] = memoryPool->allocate<global_int_t>(
    localNumberOfRows*numberOfNonzerosPerRow);

  // Now allocate the arrays pointed to
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
  for (local_int_t i=1; i< localNumberOfRows; ++i) {
    mtxIndL[i] = mtxIndL[0] + i*numberOfNonzerosPerRow;
    matrixValues[i] = matrixValues[0] + i*numberOfNonzerosPerRow;
    mtxIndG[i] = mtxIndG[0] + i*numberOfNonzerosPerRow;
  }

  A.boundaryRows = new int[nx*ny*nz - (nx-2)*(ny-2)*(nz-2)];
  A.numOfBoundaryRows = 0;
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      A.boundaryRows[A.numOfBoundaryRows++] = y*nx + x;
    }
  }
  for (int z = 1; z < nz - 1; z++) {
    for (int x = 0; x < nx; x++) {
      A.boundaryRows[A.numOfBoundaryRows++] = z*ny*nx + x;
    }
    for (int y = 1; y < ny - 1; y++) {
      A.boundaryRows[A.numOfBoundaryRows++] = (z*ny + y)*nx;
      A.boundaryRows[A.numOfBoundaryRows++] = (z*ny + y)*nx + nx - 1;
    }
    for (int x = 0; x < nx; x++) {
      A.boundaryRows[A.numOfBoundaryRows++] = (z*ny + (ny - 1))*nx + x;
    }
  }
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      A.boundaryRows[A.numOfBoundaryRows++] = ((nz - 1)*ny + y)*nx + x;
    }
  }
  assert(A.numOfBoundaryRows == nx*ny*nz - (nx-2)*(ny-2)*(nz-2));

  local_int_t localNumberOfNonzeros = 0;
  // TODO:  This triply nested loop could be flattened or use nested parallelism
#pragma omp parallel reduction(+:localNumberOfNonzeros)
  {
    local_int_t work = (nz - 2)*(ny - 2);
    local_int_t begin, end;
    getSimpleThreadPartition(&begin, &end, work);


    for (local_int_t i = begin; i < end; i++) {
      local_int_t iz = i/(ny - 2) + 1;
      local_int_t iy = i%(ny - 2) + 1;

      global_int_t giz = ipz*nz+iz;
      global_int_t giy = ipy*ny+iy;

      for (local_int_t ix=1; ix<nx-1; ix++) {
        global_int_t gix = ipx*nx+ix;
        local_int_t currentLocalRow = iz*nx*ny+iy*nx+ix;
        global_int_t currentGlobalRow = giz*gnx*gny+giy*gnx+gix;

#ifndef NDEBUG
        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#endif
#ifdef HPCG_DETAILED_DEBUG
        HPCG_fout << " rank, globalRow, localRow = " << A.geom->rank << " " << currentGlobalRow << " " << currentLocalRow << endl;
#endif
        char numberOfNonzerosInRow = 0;
        double * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
        global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row
        local_int_t * currentIndexPointerL = mtxIndL[currentLocalRow];

        for (int sz = -1; sz <= 1; sz++) {
          for (int sy = -1; sy <= 1; sy++) {
            for (int sx = -1; sx <= 1; sx++) {
              local_int_t shift = (sz*ny + sy)*nx + sx;
              global_int_t curcol = currentGlobalRow+sz*gnx*gny+sy*gnx+sx;
              if (shift==0) {
                matrixDiagonal[currentLocalRow] = currentValuePointer;
                *currentValuePointer++ = 26.0;
              } else {
                *currentValuePointer++ = -1.0;
              }
              *currentIndexPointerG++ = curcol;
              *currentIndexPointerL++ = currentLocalRow + shift;
            } // end sx loop
          } // end sy loop
        } // end sz loop*/
        numberOfNonzerosInRow += 27;
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
        localNumberOfNonzeros += numberOfNonzerosInRow;
        if (b!=0)      bv[currentLocalRow] = 26.0 - ((double) (numberOfNonzerosInRow-1));
        if (x!=0)      xv[currentLocalRow] = 0.0;
        if (xexact!=0) xexactv[currentLocalRow] = 1.0;
      } // end ix loop
    } // end iy-iz loop
  } // omp parallel

#pragma omp parallel for reduction(+:localNumberOfNonzeros)
  for (int i = 0; i < A.numOfBoundaryRows; i++) {
    local_int_t currentLocalRow = A.boundaryRows[i];

    local_int_t iz = currentLocalRow/(ny*nx);
    local_int_t iy = currentLocalRow/nx%ny;
    local_int_t ix = currentLocalRow%nx;

    global_int_t giz = ipz*nz+iz;
    global_int_t giy = ipy*ny+iy;
    global_int_t gix = ipx*nx+ix;

    global_int_t sz_begin = std::max<global_int_t>(-1, -giz);
    global_int_t sz_end = std::min<global_int_t>(1, gnz - giz - 1);

    global_int_t sy_begin = std::max<global_int_t>(-1, -giy);
    global_int_t sy_end = std::min<global_int_t>(1, gny - giy - 1);

    global_int_t sx_begin = std::max<global_int_t>(-1, -gix);
    global_int_t sx_end = std::min<global_int_t>(1, gnx - gix - 1);

    global_int_t currentGlobalRow = giz*gnx*gny+giy*gnx+gix;
#ifndef NDEBUG
    A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#endif

    char numberOfNonzerosInRow = 0;
    double * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
    global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row

    for (global_int_t sz=sz_begin; sz<=sz_end; sz++) {
      for (global_int_t sy=sy_begin; sy<=sy_end; sy++) {
        for (global_int_t sx=sx_begin; sx<=sx_end; sx++) {
          global_int_t curcol = currentGlobalRow+sz*gnx*gny+sy*gnx+sx;
          assert(ComputeRankOfMatrixRow(*A.geom, curcol) < A.geom->size);

          if (curcol==currentGlobalRow) {
            matrixDiagonal[currentLocalRow] = currentValuePointer;
            *currentValuePointer++ = 26.0;
          } else {
            *currentValuePointer++ = -1.0;
          }
          *currentIndexPointerG++ = curcol;
          if (iz + sz >= 0 && iz + sz < nz && iy + sy >= 0 && iy + sy < ny && ix + sx >= 0 && ix + sx < nx) {
            local_int_t shift = (sz*ny + sy)*nx + sx;
            mtxIndL[currentLocalRow][numberOfNonzerosInRow] = currentLocalRow + shift;
            assert(currentLocalRow + shift >= 0 && currentLocalRow + shift < localNumberOfRows);
          }
          else {
            mtxIndL[currentLocalRow][numberOfNonzerosInRow] = -1 - ComputeRankOfMatrixRow(*A.geom, curcol);
          }

          numberOfNonzerosInRow++;
        } // end sx loop
      } // end sy loop
    } // end sz loop*/

    nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
    localNumberOfNonzeros += numberOfNonzerosInRow;
    if (b!=0)      bv[currentLocalRow] = 26.0 - ((double) (numberOfNonzerosInRow-1));
    if (x!=0)      xv[currentLocalRow] = 0.0;
    if (xexact!=0) xexactv[currentLocalRow] = 1.0;
  }

#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout     << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfRows    << " rows."     << endl
      << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfNonzeros<< " nonzeros." <<endl;
#endif

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
#ifndef HPCG_OFFLOAD
  int errorCode = MPI_Allreduce(
    &localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#else
  gbl_offload_int = localNumberOfNonzeros;
  gbl_offload_signal = HPCG_OFFLOAD_ALLRED_INT_SUM;
  while (gbl_offload_signal != HPCG_OFFLOAD_RUN) { };
  totalNumberOfNonzeros = gbl_offload_int;
#endif
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
#ifndef HPCG_OFFLOAD
  int errorCode = MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#else
  gbl_offload_lint = lnnz;
  gbl_offload_signal = HPCG_OFFLOAD_ALLRED_LINT_SUM;
  while (gbl_offload_signal != HPCG_OFFLOAD_RUN) { };
  gnnz = gbl_offload_lint;
#endif
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else // HPCG_NO_MPI
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif // HPCG_NO_MPI
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;
  A.optimizationData = 0;

  return;
}
