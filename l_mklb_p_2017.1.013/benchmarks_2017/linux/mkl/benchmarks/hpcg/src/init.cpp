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

#ifndef HPCG_OFFLOAD
#ifndef HPCG_NO_MPI
#include "mpi_hpcg.hpp"
#endif
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <ctime>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <strings.h>

#include <fstream>
#include <iostream>
#include <ostream>

#include <getopt.h>

#include "hpcg.hpp"

#include "ReadHpcgDat.hpp"

using namespace std;

#ifdef HPCG_OFFLOAD
std::stringstream HPCG_fout; //!< output file stream for logging activities during HPCG run
#else
std::ofstream HPCG_fout_internal;
std::ostream& HPCG_fout = HPCG_fout_internal; //!< output file stream for logging activities during HPCG run
#endif

#ifdef HPCG_OFFLOAD
#pragma warning (disable:2423)
extern __attribute__((target(mic))) char *txtReport;
#endif

static int
startswith(const char * s, const char * prefix) {
  size_t n = strlen( prefix );
  if (strncmp( s, prefix, n ))
    return 0;
  return 1;
}

HPCG_Params::HPCG_Params() :
  comm_size(1), comm_rank(0), numThreads(1),
  nx(10), ny(10), nz(10),
#if defined(__MIC__) || defined(__AVX512F__)
  useEsb(true),
#else
  useEsb(false),
#endif
  runRef(true),
#ifdef HPCG_NO_MPI
  overlap(false),
#else
  overlap(true),
#endif
  measureImbalance(false),
  blockMethod(MATCHING),
#if defined(__MIC__) || defined(__AVX512F__)
  multiColorFromThisLevel(2),
#else
  multiColorFromThisLevel(256),
#endif
  blockColorUpToThisLevel(1),
  colorBlockSize(4),
  numberOfPresmootherSteps(1), numberOfPostsmootherSteps(1), numberOfMgLevels(4),
#if defined(__MIC__) || defined(__AVX512F__)
  fuseSpmv(true),
#else
  fuseSpmv(true),
#endif
  runRealRef(true),
  logLevel(0)
{
  yamlFileName[0] = '\0';
  matrixDumpFileName[0] = '\0';
  inputMatrixFileName[0] = '\0';
}

/*!
  Initializes an HPCG run by obtaining problem parameters (from a file on
  command line) and then broadcasts them to all nodes. It also initializes
  loggin I/O streams that are used throughout the HPCG run. Only MPI rank 0
  performs I/O operations.

  The function assumes that MPI has already been initialized for MPI runs.

  @param[in] argc_p the pointer to the "argc" parameter passed to the main() function
  @param[in] argv_p the pointer to the "argv" parameter passed to the main() function
  @param[out] params the reference to the data structures that is filled the basic parameters of the run

  @return returns 0 upon success and non-zero otherwise

  @see HPCG_Finalize
*/
int
HPCG_Init(int * argc_p, char ** *argv_p, HPCG_Params & params) {
  int argc = *argc_p;
  char ** argv = *argv_p;
  char fname[1024];
  int i, j, iparams[4] = { 0 };// = { 128, 128, 128, 1 };
  time_t rawtime;
  tm * ptm;

  static struct option long_options[] = {
    { "nx", required_argument, 0, 'x' },
    { "ny", required_argument, 0, 'y' },
    { "nz", required_argument, 0, 'z' },
    { "n", required_argument, 0, 'n' },
    { "t", required_argument, 0, 't' }, // time to run
    { "yaml", required_argument, 0, 'o' },
    { "dump-matrix", optional_argument, 0, 'd' },
    { "esb", optional_argument, 0, 'e' },
    { "multi-coloring", optional_argument, 0, 'c' }, // apply multi-coloring to which level?
    { "block-coloring", optional_argument, 0, 'k' }, // apply block-coloring to which level?
    { "color-block", required_argument, 0, 'b' }, // block size
    { "block-method", required_argument, 0, 'a' },
    { "no-ref", no_argument, 0, 'r' },
    { "mg-levels", required_argument, 0, 'p' },
    { "overlap", optional_argument, 0, 'v' },
    { "fuse-spmv", optional_argument, 0, 'f' },
    { "measure-imbalance", no_argument, 0, 'i', },
    { "log-level", required_argument, 0, 'l' },
    { "log-out", required_argument, 0, 'g' },
    { "help", no_argument, 0, 'h' },
    { "run-real-ref", optional_argument, 0, 's' },
    { NULL, 0, 0, 0 }
  };

  string logFileName;

  while (true) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "x:y:z:n:o:d::m:l:c:e:t:b:h:a", long_options, &option_index);
    if (-1 == c)  break;
    if ('?' == c) exit(-1);

    switch (c) {
      case 'x': iparams[0] = atoi(optarg); break;
      case 'y': iparams[1] = atoi(optarg); break;
      case 'z': iparams[2] = atoi(optarg); break;
      case 't': iparams[3] = atoi(optarg); break;
      case 'n': 
      {
        int n = atoi(optarg);
        char buf[1024];
        sprintf(buf, "%d", n);
        if (!strcmp(buf, optarg)) {
          iparams[0] = iparams[1] = iparams[2] = atoi(optarg);
        }
        else {
          strcpy(params.inputMatrixFileName, optarg);
          iparams[0] = iparams[1] = iparams[2] = -1;
          params.numberOfMgLevels = 1;
          if (GEOMETRIC == params.blockMethod) {
            params.blockMethod = GREEDY;
          }
        }
        break;
      }
      case 'a':
      {
        if (!strcmp("GEOMETRIC", optarg)) {
          params.blockMethod = GEOMETRIC;
        }
        else if (!strcmp("GREEDY", optarg)) {
          params.blockMethod = GREEDY;
        }
        else if (!strcmp("METIS", optarg)) {
          params.blockMethod = METIS;
        }
        else if (!strcmp("MATCHING", optarg)) {
          params.blockMethod = MATCHING;
        }
        else {
          assert(false);
        }
        break;
      }
      case 'o': strcpy(params.yamlFileName, optarg); break;
      case 'd': strcpy(params.matrixDumpFileName, optarg); break;
      case 'c': params.multiColorFromThisLevel = atoi(optarg); break;
      case 'k': params.blockColorUpToThisLevel = atoi(optarg); break;
      case 'r': params.runRef = false; break;
      case 'b': params.colorBlockSize = atoi(optarg); break;
      case 'l': params.logLevel = atoi(optarg); break;
      case 'g': logFileName = optarg; break;
      case 'p':
        if (iparams[0] != -1)
          // when reading input from file, mg levels should be 1
          params.numberOfMgLevels = atoi(optarg);
        break;
      case 'e':
      case 'f':
      case 'v':
      case 's':
      {
        bool *flag = NULL;
        switch (c) {
        case 'e': flag = &params.useEsb; break;
        case 'f': flag = &params.fuseSpmv; break;
        case 'v': flag = &params.overlap; break;
        case 's': flag = &params.runRealRef; break;
        }
        if (!optarg || !strlen(optarg) ||
            !strcasecmp(optarg, "y") || !strcasecmp(optarg, "yes") ||
            !strcasecmp(optarg, "t") || !strcasecmp(optarg, "true") ||
            !strcmp(optarg, "1")) {
          *flag = true;
        }
        else if (!strcasecmp(optarg, "n") || !strcasecmp(optarg, "no") ||
            !strcasecmp(optarg, "f") || !strcasecmp(optarg, "false") ||
            !strcmp(optarg, "0")) {
          *flag = false;
        }
        break;
      }
      case 'i':
        params.measureImbalance = true;
        break;
      case 'h':
        fprintf(
          stderr,
          "usage: %s [--nx=nx] [--ny=ny] [--nz=nz] [--n=n] [--t=t]\n"
          "--n: set nx, ny, and nz to the same number\n"
          "--t: time to run\n",
          argv[0]);
        exit(-1);

      default: break;
    }
  }

#ifndef HPCG_NO_MPI
  if (params.fuseSpmv && !params.overlap) {
    fprintf(stderr, "When SpMV is fused with SymGS, we also must overlap communication.\n");
    exit(-1);
  }
#endif

  if (optind < argc - 1) {
    iparams[0] = atoi(argv[optind]);
    iparams[1] = atoi(argv[optind + 1]);
    iparams[2] = atoi(argv[optind + 2]);
  }

  if (! iparams[0] && ! iparams[1] && ! iparams[2]) { /* no geometry arguments on the command line */
    if (ReadHpcgDat(iparams, iparams+3) < 0) {
      iparams[0] = 112;
      iparams[1] = 112;
      iparams[2] = 112;
      iparams[3] = 1;
    };
  }

  // force iparams >= 16
  for (i = 0; i < 3; ++i) {
    if (-1 == iparams[i]) continue; // this is for reading from file
    if (iparams[i] < 16)
      for (j = 1; j <= 2; ++j)
        if (iparams[(i+j)%3] > iparams[i])
          iparams[i] = iparams[(i+j)%3];
    if (iparams[i] < 16)
      iparams[i] = 16;
  }

#ifndef HPCG_NO_MPI
#ifndef HPCG_OFFLOAD
  //MPI_Bcast( iparams, 4, MPI_INT, 0, MPI_COMM_WORLD );
#endif
#endif

  params.nx = iparams[0];
  params.ny = iparams[1];
  params.nz = iparams[2];

  if (0 == params.comm_rank) {
    if (params.nx%8 || params.ny%8 || params.nz%8) {
      fprintf(stderr, "Input size must be a multiple of 8\n");
      exit(-1);
    }
    if (params.overlap && (params.nx < 24 || params.ny < 24 || params.nz < 24)) {
      fprintf(stderr, "Local dimension size of each direction must be greater than or equal to 24\n");
      exit(-1);
    }
  }

  params.runningTime = iparams[3];

#ifdef HPCG_NO_MPI
  params.comm_rank = 0;
  params.comm_size = 1;
#else
#ifndef HPCG_OFFLOAD
  int errorCode = MPI_Comm_rank( MPI_COMM_WORLD, &params.comm_rank );
  handleMPIError(errorCode);
  errorCode = MPI_Comm_size( MPI_COMM_WORLD, &params.comm_size );
  handleMPIError(errorCode);
#endif
#endif

  if (0 == params.comm_rank && params.logLevel > 0) {
    if (params.nx > 0) {
      printf("nx=%d ", params.nx);
      if (params.ny != params.nx) {
        printf("ny=%d ", params.ny);
      }
      if (params.nz != params.nx) {
        printf("nz=%d ", params.nz);
      }
    }
    if (params.useEsb) {
      printf("esb ");
    }
    if (params.multiColorFromThisLevel < 256) {
      printf("multi-color-from-level=%d ", params.multiColorFromThisLevel);
      if (params.blockColorUpToThisLevel > 0) {
        printf("block-color-up-to-level=%d ", params.blockColorUpToThisLevel);
      }
      if (params.colorBlockSize > 1) {
        printf("color-block=%d ", params.colorBlockSize);
//        printf("block-method=%s ", toString(params.blockMethod).c_str());
      }
    }
    if (params.numberOfMgLevels != 4) {
      printf("mg-levels=%d ", params.numberOfMgLevels);
    }
    if (params.numberOfPresmootherSteps != 1) {
      printf("pre-smoother-steps=%d ", params.numberOfPresmootherSteps);
    }
    if (params.numberOfPostsmootherSteps != 1) {
      printf("post-smoother-steps=%d ", params.numberOfPostsmootherSteps);
    }
    if (params.overlap) {
      printf("overlap ");
    }
    if (params.fuseSpmv) {
      printf("fuse-spmv ");
    }
    if (!params.runRealRef) {
      printf("run-real-ref=0 ");
    }
#ifdef HPCG_DEBUG
    printf("debug ");
#endif
    if (params.logLevel > 0) {
      printf("log-level=%d", params.logLevel);
    }
    printf("\n");
  }

#ifdef HPCG_NO_OPENMP
  params.numThreads = 1;
#else
  #pragma omp parallel
  params.numThreads = omp_get_num_threads();
#endif

  time ( &rawtime );
  ptm = localtime(&rawtime);
  if (params.yamlFileName[0]) {
    sprintf(fname, "%s.txt", params.yamlFileName);
  } else {
    sprintf(
      fname, "hpcg_log_n%d_%dp_%dt_%04d.%02d.%02d.%02d.%02d.%02d.txt",
      params.nx, params.comm_size, omp_get_num_threads(),
      1900 + ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
  }

#ifdef HPCG_OFFLOAD
  HPCG_fout.rdbuf()->pubsetbuf(txtReport, CONTENTS_MAX_LEN);
#else
  if (0 == params.comm_rank) {
    if (logFileName.empty()) {
      HPCG_fout_internal.open(fname);
    }
    else if (logFileName == "stdout") {
      HPCG_fout.rdbuf(cout.rdbuf());
    }
    else {
      HPCG_fout_internal.open(logFileName.c_str());
    }
  }
  else {
#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)

    sprintf(
      fname, "hpcg_log_n%d_%dp_%dt_%d_%04d.%02d.%02d.%02d.%02d.%02d.txt",
      params.nx, params.comm_size, omp_get_num_threads(),
      params.comm_rank,
      1900 + ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);

    HPCG_fout_internal.open(fname);
#else
    if (logFileName == "stdout") {
      HPCG_fout.rdbuf(cout.rdbuf());
    }
    else {
      HPCG_fout_internal.open("/dev/null");
    }
#endif
  }
#endif

  return 0;
}
