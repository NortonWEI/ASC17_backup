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

// The real main routine of a program that can be offloaded

#ifndef HPCG_NO_MPI
#ifndef HPCG_OFFLOAD
#include "mpi_hpcg.hpp" // If this routine is not compiled with HPCG_NO_MPI
#endif
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#ifdef HPCG_OFFLOAD
#include "offloadExtHpcgLib.hpp"
#endif

#include "SpMP/Utils.hpp"
#include "SpMP/MemoryPool.hpp"

using namespace std;
using SpMP::MemoryPool;

#ifdef HPCG_OFFLOAD
#pragma warning (disable:2423)
extern __attribute__((target(mic))) char *txtReport;
#endif

extern bool inTimedCgLoop;

#if !defined(HPCG_NO_MPI) && !defined(HPCG_OFFLOAD)
void handleMPIError(int errorCode)
{
  if (errorCode != MPI_SUCCESS) {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char error_string[BUFSIZ];
    int length_of_error_string;

    MPI_Error_string(errorCode, error_string, &length_of_error_string);
    cout << "[" << my_rank << "] Error!!! " << error_string << endl;
  }
}
#endif

#define diff_time(x,y) ((x.tv_nsec - y.tv_nsec > 0 ) ? ((double) (x.tv_sec - y.tv_sec)) + ((double)x.tv_nsec - y.tv_nsec)/1000000000.0 : ((double)(x.tv_sec - y.tv_sec - 1)+((double) (x.tv_nsec - y.tv_nsec + 1000000000)/1000000000.0)))

double HPCG_Run(HPCG_Params params)
{
#ifdef HPCG_OFFLOAD
  //HPCG_fout.rdbuf(cout.rdbuf());
  HPCG_fout.rdbuf()->pubsetbuf(txtReport, CONTENTS_MAX_LEN);
#endif

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0);
  if (ierr) {
#ifdef HPCG_OFFLOAD
    gbl_offload_signal = HPCG_OFFLOAD_CHECK_FAILED;
#endif
    return ierr;
  }

  MemoryPool *memoryPool = MemoryPool::getSingleton();
  memoryPool->initialize(
    nx < 0
    ? 6L*1024*1024*1024
    : 896L*nx*ny*nz + 1024*1024);

  struct timespec run_time0, run_time1;
  double run_time = 0;

  // //////////////////////
  // Problem setup Phase //
  /////////////////////////
 
  clock_gettime(CLOCK_REALTIME, &run_time0);

  double t1 = mytimer();

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  if (nx > 0) {
    // nx < 0 when reading input matrix from file
    GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom);

    ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
    if (ierr) {
#ifdef HPCG_OFFLOAD
      gbl_offload_signal = HPCG_OFFLOAD_CHECK_FAILED;
#endif
      return ierr;
    }
  }
  else {
    geom->size = 1;
    geom->rank = 0;
    geom->npx = geom->npy = geom->npz = 1;
    geom->ipx = geom->ipy = geom->ipz = 0;
    geom->numThreads = params.numThreads;
    geom->nx = nx;
    geom->ny = ny;
    geom->nz = nz;
  }

  // Use this array for collecting timing information
  vector< double > times(10,0.0);

  double setup_time = mytimer();

  SparseMatrix A;
  InitializeSparseMatrix(A, geom);

  Vector b, x, xexact;
  InitializeVectorWithMemoryPool(b, geom->nx*geom->ny*geom->nz);
  InitializeVectorWithMemoryPool(x, geom->nx*geom->ny*geom->nz);
  InitializeVectorWithMemoryPool(xexact, geom->nx*geom->ny*geom->nz);

#ifndef HPCG_NO_MPI
#ifndef HPCG_OFFLOAD
  MPI_Barrier(MPI_COMM_WORLD);
#else
  gbl_offload_signal = HPCG_OFFLOAD_BARRIER;
  while (gbl_offload_signal != HPCG_OFFLOAD_RUN) { };
#endif
#endif
  
  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] GenerateProblem begin." << endl;
  }
  size_t tailBeforeGenerateProblem = memoryPool->getTail();
  size_t headBeforeGenerateProblem = memoryPool->getHead();
  GenerateProblem(A, &b, &x, &xexact, params);
  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] GenerateProblem finished." << endl;
  }
  SetupHalo(A);
  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] SetupHalo finished." << endl;
  }

#ifdef HPCG_OFFLOAD
  gbl_offload_m0 = A.localNumberOfRows;
  gbl_offload_n0 = A.localNumberOfColumns;

  gbl_offload_level = 0;
  gbl_offload_signal = HPCG_OFFLOAD_SET_PROBLEM_SIZE;
  while (HPCG_OFFLOAD_RUN != gbl_offload_signal) {};
#endif

  int numberOfMgLevels = params.numberOfMgLevels; // Number of levels including first
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
	  GenerateCoarseProblem(*curLevelMatrix, params, level);
	  curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
    if (params.logLevel >= 5 || 4 == params.logLevel && 0 == rank) {
      HPCG_fout << "[" << rank << "] GenerateCoarseProblem at level " << level << " finished." << endl;
    }
  }

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  curLevelMatrix = &A;
  Vector * curb = &b;
  Vector * curx = &x;
  Vector * curxexact = &xexact;
#ifndef HPCG_OFFLOAD
  for (int level = 0; level< numberOfMgLevels; ++level) {
     CheckProblem(*curLevelMatrix, curb, curx, curxexact);
     curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
     curb = 0; // No vectors after the top level
     curx = 0;
     curxexact = 0;
  }
#endif

  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] GenerateCoarseProblem at all levels finished." << endl;
  }

  CGData data;
  InitializeSparseCGData(A, data);


  vector< double > mgTimes(params.numberOfMgLevels, 0.0);
  vector< double > preSmoothTimes(params.numberOfMgLevels, 0.0);
  vector< double > postSmoothTimes(params.numberOfMgLevels, 0.0);
  vector< double > spmvTimes(params.numberOfMgLevels, 0.0);
  vector< double > restrictionTimes(params.numberOfMgLevels, 0.0);
  vector< double > prolongationTimes(params.numberOfMgLevels, 0.0);
  vector< double > haloTimes(HALO_TIMES_END, 0.0);

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
#ifdef HPCG_OFFLOAD
  SetVector(x_overlap, ncol, gbl_offload_xoverlap); // Overlapped copy of x vector
#else
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
#endif
#ifndef HPCG_NO_MPI
  SetHaloCommId(x_overlap, HPCG_OFFLOAD_VECTOR_XOVERLAP);
#endif
  InitializeVector(b_computed, nrow); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = params.runRef ? 10 : 0;
#if defined(__MIC__) || defined(__AVX512F__) || !defined(HPCG_NO_MPI)
  numberOfCalls = 0;
    // we skip this since it takes really long in KNC
    // we also skip in Xeon in MPI because it can be potentially running with symmetric mode
#endif
  double t_begin = mytimer();
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
      HPCG_fout << "[" << rank << "] SPMV_ref finished." << endl;
    }

    ierr = ComputeMG_ref(A, b_computed, x_overlap, 0, params); // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
  if (params.logLevel >= 2 || params.logLevel >= 1 && 0 == rank) {
    HPCG_fout << "[" << rank << "] Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
  }

  DeleteVector(x_overlap);
  DeleteVector(b_computed);

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////
  
  HPCG_Params oldParams;
  if (!params.runRealRef) {
    oldParams = params;

    if (params.runRef) {
      params = HPCG_Params();

      params.comm_size = oldParams.comm_size;
      params.comm_rank = oldParams.comm_rank;
      params.numThreads = oldParams.numThreads;
      params.nx = oldParams.nx;
      params.ny = oldParams.ny;
      params.nz = oldParams.nz;
      params.useEsb = false;
      params.fuseSpmv = false;
      params.overlap = false;
      params.multiColorFromThisLevel = 256;
      params.runRealRef = oldParams.runRealRef;
      params.logLevel = oldParams.logLevel;
      strcpy(params.inputMatrixFileName, oldParams.inputMatrixFileName);
      if (params.nx < 0) {
        params.numberOfMgLevels = 1;
      }

      // In KNC, we run the optimized code for reference run because
      // the reference code is way too slow.

      // deallocate data structure for reference implementation
      // comment out the following block for DBG_SYMMETRY
      SparseMatrix *Ac = &A;

      while (Ac != NULL) { 
        /*delete[] Ac->matrixValues[0];
        delete[] Ac->mtxIndG[0];
        delete[] Ac->mtxIndL[0];*/

        /*delete[] Ac->nonzerosInRow;  */Ac->nonzerosInRow = NULL;
        /*delete[] Ac->mtxIndG;        */Ac->mtxIndG = NULL;
        /*delete[] Ac->mtxIndL;        */Ac->mtxIndL = NULL;
        /*delete[] Ac->matrixValues;   */Ac->matrixValues = NULL;
        /*delete[] Ac->matrixDiagonal; */Ac->matrixDiagonal = NULL;

        Ac = Ac->Ac;
      }

      memoryPool->setTail(tailBeforeGenerateProblem); // deallocate memory of Acs

      // Call user-tunable set up function.
      OptimizeProblemInit(A, params);
      if (params.logLevel >= 1 && 0 == rank) {
        HPCG_fout << "OptimizeProblemInit finished" << endl;
      }

      OptimizeProblem(A, data, b, x, xexact, params);
    }
  } // !params.runRealRef

  if (params.logLevel >= 1) {
    if (0 == rank) {
      HPCG_fout << "OptimizeProblem finished" << endl;
    }
  }

  t1 = mytimer();
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1; // Only need to run the residual reduction analysis once

  vector< double > opt_times(9,0.0);
  vector< double > optMgTimes(params.numberOfMgLevels, 0.0);
  vector< double > optPreSmoothTimes(params.numberOfMgLevels, 0.0);
  vector< double > optPostSmoothTimes(params.numberOfMgLevels, 0.0);
  vector< double > optSpmvTimes(params.numberOfMgLevels, 0.0);
  vector< double > optRestrictionTimes(params.numberOfMgLevels, 0.0);
  vector< double > optProlongationTimes(params.numberOfMgLevels, 0.0);
  vector< double > optHaloTimes(HALO_TIMES_END, 0.0);

  // Compute the residual reduction for the natural ordering and reference kernels
  vector< double > ref_times(9,0.0);
  double tolerance = nx > 0 ? 0 : 1e-7;
    // for hpcg input, set tolerance to zero to make all runs do maxIters iterations
    // for other matrices, converge to 1e-7
  int err_count = 0;
  double refTolerance;
  if (params.runRef) {
    for (int i=0; i< numberOfCalls; ++i) {
      ZeroVector(x);
      if (params.runRealRef) {
        ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true, params);
      }
      else {
        ierr = CG( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &opt_times[0], true, &optMgTimes[0], &optPreSmoothTimes[0], &optPostSmoothTimes[0], &optSpmvTimes[0], &optRestrictionTimes[0], &optProlongationTimes[0], &optHaloTimes[0], params);
      }
      if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
        HPCG_fout << "[" << rank << "] CG_ref finished." << endl;
      }

      if (ierr) ++err_count; // count the number of errors in CG
      totalNiters_ref += niters;
    }
    if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
    refTolerance = normr / normr0;
    refMaxIters = totalNiters_ref/numberOfCalls;
    if (0 == rank && params.logLevel > 0)
      printf("refTolerance = %g\n", refTolerance);

    if (false/*params.logLevel >= 1*/) {
      if (params.logLevel >= 2 || params.logLevel >= 1 && 0 == rank) {
        HPCG_fout << "[" << rank << "] Begin gather" << endl;
      }

      // print out allreduce time of each rank so that we can figue out if there's a
      // lingering node.
#ifndef HPCG_NO_MPI
      double *allReduceTimes = NULL;
      if (0 == rank) {
        allReduceTimes = new double[params.comm_size];
      }

#ifdef HPCG_OFFLOAD
      gbl_offload_dbl = opt_times[4];
      gbl_offload_signal = HPCG_OFFLOAD_GATHER_DBL;
      while(gbl_offload_signal != HPCG_OFFLOAD_RUN) {};
      if (0 == params.comm_rank) {
        memcpy(
          allReduceTimes, gbl_offload_dbl_gather, sizeof(double)*params.comm_size);
      }
#else
      int errorCode = MPI_Gather(
        params.runRealRef ? &ref_times[4] : &opt_times[4],
        1, MPI_DOUBLE,
        allReduceTimes, 1, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
      handleMPIError(errorCode);
#endif

      if (0 == rank) {
        HPCG_fout << "All reduce times of each rank in reference CG run" << endl;
        for (int i = 0; i < params.comm_size; ++i) {
          HPCG_fout << "[" << i << "] " << allReduceTimes[i] << endl;
        }
        HPCG_fout << endl;
        delete[] allReduceTimes;
      }

#endif // HPCG_NO_MPI

      if (params.logLevel >= 2 || params.logLevel >= 1 && 0 == rank) {
        HPCG_fout << "[" << rank << "] End gather" << endl;
      }
    }
  }
  else {
    refTolerance = 0;
  }

  if (!params.runRealRef) {
    params = oldParams;

    if (params.runRef) {
      OptimizeProblemFinalize(A, params);
      memoryPool->setTail(tailBeforeGenerateProblem);
      memoryPool->setHead(headBeforeGenerateProblem);
    }
  }
  else {
    // deallocate data structure for reference implementation
    // comment out the following block for DBG_SYMMETRY
    SparseMatrix *Ac = &A;

    while (Ac != NULL) { 
      /*delete[] Ac->matrixValues[0];
      delete[] Ac->mtxIndG[0];
      delete[] Ac->mtxIndL[0];*/

      /*delete[] Ac->nonzerosInRow; */ Ac->nonzerosInRow = NULL;
      /*delete[] Ac->mtxIndG;       */ Ac->mtxIndG = NULL;
      /*delete[] Ac->mtxIndL;       */ Ac->mtxIndL = NULL;
      /*delete[] Ac->matrixValues;  */ Ac->matrixValues = NULL;
      /*delete[] Ac->matrixDiagonal;*/ Ac->matrixDiagonal = NULL;

      Ac = Ac->Ac;
    }

    memoryPool->setTail(tailBeforeGenerateProblem); // deallocate memory of Acs

    // Call user-tunable set up function.
    OptimizeProblemInit(A, params);
  } // params.runRealRef

  if (params.logLevel >= 2 || params.logLevel >= 1 && 0 == rank) {
    HPCG_fout << "[" << rank << "] Begin OptimizeProblemInit" << endl;
  }

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact, params);
  t7 = mytimer() - t7;
  times[7] = t7;

  if (params.logLevel >= 2 || params.logLevel >= 1 && 0 == rank) {
    HPCG_fout << "[" << rank << "] Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
  }

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

  t1 = mytimer();
  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  if (params.nx > 0) {
    // skip this when input is read from file
    TestCG(A, data, b, x, testcg_data, params);

    if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
      HPCG_fout << "[" << rank << "] TestCG finished." << endl;
    }
  }

  TestSymmetryData testsymmetry_data;
  testsymmetry_data.count_fail = 0; 
  if (params.nx > 0) {
    // skip this when input is read from file
    TestSymmetry(A, b, xexact, testsymmetry_data, params);
  }

  if (params.logLevel >= 2 && params.logLevel >= 1 && rank == 0) {
    HPCG_fout << "[" << rank << "] Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
  }

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = params.runRef ? 0 : refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  //// When we want to force reference tolerance
  //refTolerance = 5.31511e-05;

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true, &optMgTimes[0], &optPreSmoothTimes[0], &optPostSmoothTimes[0], &optSpmvTimes[0], &optRestrictionTimes[0], &optProlongationTimes[0], &optHaloTimes[0], params);
    if (ierr) ++err_count; // count the number of errors in CG
    if (params.runRef && normr / normr0 > refTolerance) ++tolerance_failures; // the number of failures to reduce residual

    // pick the largest number of iterations to guarantee convergence
    if (niters > optNiters) {
      optNiters = niters;
    }
    if (0 == i && 0 == rank && params.logLevel > 0) {
      printf("tolerance = %g, iterations = %d\n", normr/normr0, optNiters);
    }

    double current_time = opt_times[0] - last_cummulative_time;
    if (current_time > opt_worst_time) opt_worst_time = current_time;

    if (false/*params.logLevel >= 1*/) {
      // print out allreduce time of each rank so that we can figue out if there's a
      // lingering node.
#ifndef HPCG_NO_MPI
      double *allReduceTimes = NULL;
      if (0 == rank) {
        allReduceTimes = new double[params.comm_size];
      }

#ifdef HPCG_OFFLOAD
      gbl_offload_dbl = opt_times[4];
      gbl_offload_signal = HPCG_OFFLOAD_GATHER_DBL;
      while(gbl_offload_signal != HPCG_OFFLOAD_RUN) {};
      if (0 == params.comm_rank) {
        memcpy(
          allReduceTimes, gbl_offload_dbl_gather, sizeof(double)*params.comm_size);
      }
#else
      int errorCode = MPI_Gather(
        &opt_times[4],
        1, MPI_DOUBLE,
        allReduceTimes, 1, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
      handleMPIError(errorCode);
#endif

      if (0 == rank) {
        HPCG_fout << "All reduce times of each rank in optimized CG run" << endl;
        for (int i = 0; i < params.comm_size; ++i) {
          HPCG_fout << "[" << i << "] " << allReduceTimes[i] << endl;
        }
        HPCG_fout << endl;
        delete[] allReduceTimes;
      }

#endif // HPCG_NO_MPI
    }
  }

#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD
  gbl_offload_dbl = opt_worst_time;
  gbl_offload_signal = HPCG_OFFLOAD_ALLRED_DBL_MAX;
  while(gbl_offload_signal != HPCG_OFFLOAD_RUN) {};
  opt_worst_time = gbl_offload_dbl; 
#else
// Get the absolute worst time across all MPI ranks (time in CG can be different)
  double local_opt_worst_time = opt_worst_time;
  int errorCode = MPI_Allreduce(
    &local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#endif
#endif


  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
  if (tolerance_failures) {
    global_failure = 1;
    if (rank == 0)
      HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
  }

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

  if (params.logLevel >= 2 || params.logLevel >= 1 && rank==0) {
    HPCG_fout << "[" << rank << "] Projected running time: " << total_runtime << " seconds" << endl;
    HPCG_fout << "[" << rank << "] Number of CG sets: " << numberOfCgSets << endl;
  }

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

#ifdef HPCG_OFFLOAD
  gbl_offload_signal = HPCG_OFFLOAD_START_TIMING;
  while (HPCG_OFFLOAD_RUN != gbl_offload_signal) {};
#endif

  clock_gettime(CLOCK_REALTIME, &run_time1);

  if (params.logLevel >= 1 && 0 == rank) {
    printf("Benchmark initialization phase took %.3f seconds\n", diff_time(run_time1, run_time0));
    run_time += diff_time(run_time1, run_time0);
  }

  clock_gettime(CLOCK_REALTIME, &run_time0);

  inTimedCgLoop = true;

  for (int i=0; i< numberOfCgSets; ++i) {
    ZeroVector(x); // Zero out x

    // Before starting CG, call barrier to synchronize processes.
#ifndef HPCG_NO_MPI
#ifndef HPCG_OFFLOAD
      errorCode = MPI_Barrier(MPI_COMM_WORLD);
      handleMPIError(errorCode);
#else
      gbl_offload_signal = HPCG_OFFLOAD_BARRIER;
      while (gbl_offload_signal != HPCG_OFFLOAD_RUN) { };
#endif
#endif

    ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true, &mgTimes[0], &preSmoothTimes[0], &postSmoothTimes[0], &spmvTimes[0], &restrictionTimes[0], &prolongationTimes[0], &haloTimes[0], params);
    if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
    testnorms_data.values[i] = normr/normr0; // Record scaled residual from this run

    if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
      HPCG_fout << "[" << rank << "] Timed CG:" << i << " finished." << endl;
    }
  }

  inTimedCgLoop = false;

  clock_gettime(CLOCK_REALTIME, &run_time1);

  if (params.logLevel >= 1 && 0 == rank) {
    printf("Benchmark timing phase took %.3f seconds\n", diff_time(run_time1, run_time0));
    run_time += diff_time(run_time1, run_time0);
  }

  clock_gettime(CLOCK_REALTIME, &run_time0);

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.
#ifdef HPCG_DEBUG
  double residual = 0;
  ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
  if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
  if (rank==0) HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
#endif

  // Test Norm Results
  ierr = TestNorms(testnorms_data);

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  double gflops = ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], &mgTimes[0], &preSmoothTimes[0], &postSmoothTimes[0], &spmvTimes[0], &restrictionTimes[0], &prolongationTimes[0], &haloTimes[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, params);

  // Clean up
  OptimizeProblemFinalize(A, params);
  memoryPool->setTail(tailBeforeGenerateProblem);
  memoryPool->setHead(headBeforeGenerateProblem);
  OptimizationData *optData = (OptimizationData *)A.optimizationData;
  SparseMatrix *Ac = &A;
  while (Ac) {
    delete (OptimizationData *)Ac->optimizationData;
    Ac = Ac->Ac;
  }

  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  delete [] testnorms_data.values;

  // Finish up
  HPCG_Finalize();

#ifdef HPCG_OFFLOAD
  gbl_offload_signal = HPCG_OFFLOAD_STOP;
#endif

  clock_gettime(CLOCK_REALTIME, &run_time1);

  if (params.logLevel >= 1 && 0 == rank) {
    printf("Benchmark wrap-up phase took %.3f seconds\n", diff_time(run_time1, run_time0));
    run_time += diff_time(run_time1, run_time0);
    printf("Benchmark execution took %.3f seconds and finished with %.3f GFLOP/s\n", run_time, gflops);
  }

  return times[0] ;
}
