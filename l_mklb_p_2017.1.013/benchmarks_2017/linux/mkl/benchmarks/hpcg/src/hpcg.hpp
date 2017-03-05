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
 @file hpcg.hpp

 HPCG data structures and functions
 */

#ifndef HPCG_HPP
#define HPCG_HPP

#include <fstream>

#define CONTENTS_MAX_LEN 10000
#define FILENAME_MAX_LEN 100

#ifdef HPCG_OFFLOAD
//extern std::iostream HPCG_fout;
#include <sstream>
#include <string>
extern std::stringstream HPCG_fout;
#else
extern std::ostream& HPCG_fout;
#endif

typedef enum {
  GEOMETRIC, // only works for hpcg input
  GREEDY, // greedy algorithm in the block coloring paper
  MATCHING, // matching
  METIS, // this is the slowest so move to the last
} BLOCK_METHOD;

struct HPCG_Params {
  int comm_size; //!< Number of MPI processes in MPI_COMM_WORLD
  int comm_rank; //!< This process' MPI rank in the range [0 to comm_size - 1]
  int numThreads; //!< This process' number of threads
  int nx; //!< Number of x-direction grid points for each local subdomain
  int ny; //!< Number of y-direction grid points for each local subdomain
  int nz; //!< Number of z-direction grid points for each local subdomain
  int runningTime; //!< Number of seconds to run the timed portion of the benchmark
  char yamlFileName[1024];
  char matrixDumpFileName[1024];
  char inputMatrixFileName[1024];

  bool useEsb;
  bool runRef; // run reference version, true by default
  bool overlap; // overlap communication with computation
  bool measureImbalance; // measure imbalance btw MPI ranks by invoking barriers

  BLOCK_METHOD blockMethod; // default GEOMETRIC

  int multiColorFromThisLevel;
  int blockColorUpToThisLevel;
  int colorBlockSize; // default 1
  int numberOfMgLevels; // default 4
  int numberOfPresmootherSteps; // default 1
  int numberOfPostsmootherSteps; // default 1

  bool fuseSpmv; // default true
  bool runRealRef; // default true

  int logLevel;
    // 0 : no log
    // 1 : rank 0 prints out what HPCG_DEBUG would print in reference code
    // 2 : all ranks print out what HPCG_DEBUG would print
    // 3 : all ranks print out important steps in HPCG_Run
    // 4 : all ranks print out progress within CG loop
    // 5 : all ranks print out progress within MG recursion

  HPCG_Params();
};

extern int HPCG_Init(int * argc_p, char ** *argv_p, HPCG_Params & params);
extern int HPCG_Finalize(void);

/**
 * The main routine of a program that can be offloaded
 *
 * @return total time
 */
#if defined(HPCG_OFFLOAD) && defined(__INTEL_OFFLOAD)
__attribute__((target(mic)))
#endif
double HPCG_Run(HPCG_Params params);

#ifndef HPCG_NO_MPI
void handleMPIError(int errorCode);
#endif

#endif // HPCG_HPP
