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
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include "mpi_hpcg.hpp"
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
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
#include "offloadExtHpcg.hpp"

#ifndef HPCG_NO_MPI
static void handleMPIError(int errorCode)
{
  if (errorCode != MPI_SUCCESS) {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char error_string[BUFSIZ];
    int length_of_error_string;

    MPI_Error_string(errorCode, error_string, &length_of_error_string);
    std::cout << "[" << my_rank << "] Error!!! " << error_string << std::endl;
  }
}
#endif

#if defined(__AVX__) or defined(__AVX2__)
extern void check_cpu();
__attribute__((constructor(101)))
void check_cpu_features() {
    check_cpu();
}
#endif

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

  int globalUseXeonPhi = 0;
#ifndef HPCG_NO_MPI
  int errorCode = MPI_Init(&argc, &argv);
  handleMPIError(errorCode);

  // Check if Xeon Phi is used somewhere
#ifdef __MIC__
  int localUseXeonPhi = 1; // 1 means Xeon Phi in symmetric mode, 2 means offload
#else
  int localUseXeonPhi = 0;
#endif
  errorCode = MPI_Allreduce(
    &localUseXeonPhi, &globalUseXeonPhi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] HPCG_Init finished." << endl;
  }

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

#ifndef HPCG_NO_MPI
  int namelen;
  // let offloading ranks determine which coprocessor should be used
  char myname[MPI_MAX_PROCESSOR_NAME];
  errorCode = MPI_Get_processor_name(myname, &namelen);
  handleMPIError(errorCode);
  char (*allnames)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(size*MPI_MAX_PROCESSOR_NAME*sizeof(char));  
  int *offloadFlags = (int *)malloc(size*sizeof(int));
  int myOffloadFlag = 0;
  errorCode = MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allgather(&myOffloadFlag, 1, MPI_INT, offloadFlags, 1, MPI_INT, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  free(allnames);
  free(offloadFlags);
#endif

  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    HPCG_fout << "[" << rank << "] Begin HPCG_Run." << endl;
  }

  double totalTime = HPCG_Run(params);
  if (totalTime < 0) {
    return 1;
  }

#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  double max_halo_param = 0.0;
  double max_halo_pciedown = 0.0; 
  double max_halo_fabrics = 0.0;
  double max_halo_pcieup = 0.0;

  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_param, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_pciedown, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_fabrics, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_pcieup, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);

  if (params.logLevel > 0 && rank == 0 && globalUseXeonPhi == 2) {
    std::cout << std::endl;
    std::cout << "Host-side halo-exchange summary:" << std::endl;
    std::cout << "   parameter download:  " << max_halo_param << " (" << (max_halo_param/totalTime*100) << "%% of total)" << std::endl;
    std::cout << "   PCI buffer download: " << max_halo_pciedown << " (" << (max_halo_pciedown/totalTime*100) << "%%)" << std::endl;
    std::cout << "   MPI Irecv/Send/Wait: " << max_halo_fabrics << " (" << (max_halo_fabrics/totalTime*100) << "%%)" << std::endl;
    std::cout << "   PCI buffer upload:   " << max_halo_pcieup << " (" << (max_halo_pcieup/totalTime*100) << "%%)" << std::endl;
    std::cout << std::endl;       
  }
#endif

  MPI_Finalize();
#endif

  return 0;
}
