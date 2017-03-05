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

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <offload.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <sstream>
#include <numeric>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <getopt.h>
#include <unistd.h>

#include <immintrin.h>

#include "offloadExtHpcg.hpp"
#include "hpcg.hpp"

using namespace std;

__attribute__((target(mic))) char *yamlReport, *yamlReportFile;
__attribute__((target(mic))) char *txtReport;
char txtReportFile[FILENAME_MAX_LEN];

__attribute__((target(mic))) volatile int gbl_offload_signal;
__attribute__((target(mic))) volatile double gbl_offload_dbl;
__attribute__((target(mic))) volatile int gbl_offload_int;
__attribute__((target(mic))) volatile long long gbl_offload_lint;

__attribute__((target(mic))) volatile int gbl_offload_halo_max_neighbors;
__attribute__((target(mic))) volatile int gbl_offload_halo_max_mpi_buffer_size;
__attribute__((target(mic))) volatile int gbl_offload_halo_num_neighbors;
__attribute__((target(mic))) int* gbl_offload_halo_neighbors;
__attribute__((target(mic))) int* gbl_offload_halo_recv_sizes;
__attribute__((target(mic))) int* gbl_offload_halo_send_sizes;
__attribute__((target(mic))) double* gbl_offload_halo_send_buf;
__attribute__((target(mic))) volatile int gbl_offload_halo_id;

__attribute__((target(mic))) volatile int gbl_offload_level = -1;

__attribute__((target(mic))) double *gbl_offload_x1 = NULL, *gbl_offload_x2 = NULL, *gbl_offload_x3 = NULL;
__attribute__((target(mic))) double *gbl_offload_p = NULL, *gbl_offload_z = NULL;
__attribute__((target(mic))) double *gbl_offload_xncol = NULL, *gbl_offload_yncol = NULL, *gbl_offload_zncol = NULL;
__attribute__((target(mic))) double *gbl_offload_xoverlap = NULL;

__attribute__((target(mic))) volatile int gbl_offload_m0 = -1, gbl_offload_m1 = -1, gbl_offload_m2 = -1, gbl_offload_m3 = -1;
__attribute__((target(mic))) volatile int gbl_offload_n0 = -1, gbl_offload_n1 = -1, gbl_offload_n2 = -1, gbl_offload_n3 = -1;

__attribute__((target(mic))) double *gbl_offload_dbl_gather = NULL;

#ifndef HPCG_NO_MPI
static void handleMPIError(int errorCode)
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

static double *haloVectors[HPCG_OFFLOAD_VECTOR_XOVERLAP + 1];
static int haloOffset[HPCG_OFFLOAD_VECTOR_XOVERLAP + 1];

int set_problem_size(int micid) {
  char st;

  #pragma offload_transfer target(mic : micid) \
    out(gbl_offload_level)
  {}

  if (0 == gbl_offload_level) {
    #pragma offload_transfer target(mic : micid) \
      out(gbl_offload_m0, gbl_offload_n0)
    {}

    haloOffset[HPCG_OFFLOAD_VECTOR_P] = gbl_offload_m0;
    haloOffset[HPCG_OFFLOAD_VECTOR_Z] = gbl_offload_m0;
    haloOffset[HPCG_OFFLOAD_VECTOR_XNCOL] = gbl_offload_m0;
    haloOffset[HPCG_OFFLOAD_VECTOR_YNCOL] = gbl_offload_m0;
    haloOffset[HPCG_OFFLOAD_VECTOR_ZNCOL] = gbl_offload_m0;
    haloOffset[HPCG_OFFLOAD_VECTOR_XOVERLAP] = gbl_offload_m0;
  }
  else if (1 == gbl_offload_level) {
    #pragma offload_transfer target(mic : micid) \
      out(gbl_offload_m1, gbl_offload_n1)
    {}

    haloOffset[HPCG_OFFLOAD_VECTOR_X1] = gbl_offload_m1;
  }
  else if (2 == gbl_offload_level) {
    #pragma offload_transfer target(mic : micid) \
      out(gbl_offload_m2, gbl_offload_n2)
    {}

    haloOffset[HPCG_OFFLOAD_VECTOR_X2] = gbl_offload_m2;
  }
  else if (3 == gbl_offload_level) {
    #pragma offload_transfer target(mic : micid) \
      out(gbl_offload_m3, gbl_offload_n3)
    {}

    haloOffset[HPCG_OFFLOAD_VECTOR_X3] = gbl_offload_m3;
  }

  return 0;
}

#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD_HALO_PROFILING
double halo_param = 0;
double halo_pciedown = 0;
double halo_fabrics = 0;
double halo_pcieup = 0;
#endif

struct HaloInfo
{
  int totalToBeSent;
  vector<MPI_Request> requests;
};

struct HaloInfo2
{
  int totalToBeSent;
  int numNeighbors;
  int recvSizes[32];
  int sendSizes[32];
  int neighbors[32];
};

static vector<HaloInfo> haloInfos;
static vector<HaloInfo2> haloInfos2;

int halo_setup_wrapper(int micid) {
  char st;

#ifdef HPCG_OFFLOAD_HALO_PROFILING
  double t = MPI_Wtime();
#endif

  // download halo id and sizes
  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_halo_id, gbl_offload_halo_num_neighbors)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  // download offset data
  #pragma offload_transfer target(mic : micid) signal(&st) \
    out(gbl_offload_halo_neighbors : length(gbl_offload_halo_num_neighbors) alloc_if(0) free_if(0) align(64)) \
    out(gbl_offload_halo_recv_sizes : length(gbl_offload_halo_num_neighbors) alloc_if(0) free_if(0) align(64)) \
    out(gbl_offload_halo_send_sizes : length(gbl_offload_halo_num_neighbors) alloc_if(0) free_if(0) align(64))
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  if (gbl_offload_halo_num_neighbors < 0 || gbl_offload_halo_num_neighbors >= 1024) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf(
      "[%d] %s:%d num_neighbors=%d\n",
      rank, __FILE__, __LINE__, gbl_offload_halo_num_neighbors);
    fflush(stdout);
  }

  if (gbl_offload_halo_id >= haloInfos.size()) {
    haloInfos.resize(gbl_offload_halo_id + 1);
  }

  assert(haloInfos[gbl_offload_halo_id].requests.empty());
  haloInfos[gbl_offload_halo_id].requests.resize(gbl_offload_halo_num_neighbors*2);

  int MPI_MY_TAG = 99;

  double *recvBuffer = haloVectors[gbl_offload_halo_id];
  double *sendBuffer = gbl_offload_halo_send_buf;

  for (int i = 0; i < gbl_offload_halo_num_neighbors; ++i) {
    int n_recv = gbl_offload_halo_recv_sizes[i];
    int errorCode = MPI_Recv_init(
      recvBuffer, n_recv, MPI_DOUBLE, gbl_offload_halo_neighbors[i],
      MPI_MY_TAG + gbl_offload_halo_id, MPI_COMM_WORLD,
      &haloInfos[gbl_offload_halo_id].requests[i]);
    handleMPIError(errorCode);
    recvBuffer += n_recv;

    int n_send = gbl_offload_halo_send_sizes[i];
    errorCode = MPI_Send_init(
      sendBuffer, n_send, MPI_DOUBLE, gbl_offload_halo_neighbors[i],
      MPI_MY_TAG + gbl_offload_halo_id, MPI_COMM_WORLD,
      &haloInfos[gbl_offload_halo_id].requests[i + gbl_offload_halo_num_neighbors]);
    handleMPIError(errorCode);
    sendBuffer += n_send;
  }

#ifdef HPCG_OFFLOAD_HALO_PROFILING
  double cur_t = MPI_Wtime() - t;
  halo_param += cur_t;
  t = MPI_Wtime();
#endif

  haloInfos[gbl_offload_halo_id].totalToBeSent = sendBuffer - gbl_offload_halo_send_buf;
  assert(recvBuffer - haloVectors[gbl_offload_halo_id] == sendBuffer - gbl_offload_halo_send_buf);

#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  cur_t = MPI_Wtime() - t;
  halo_fabrics += cur_t;
#endif

  return 0;
}

int halo_exchange_wrapper(int micid) {
  char st;

#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  double t = MPI_Wtime();
#endif

  // download id
  #pragma offload_transfer target(mic : micid) signal(&st) \
    out(gbl_offload_halo_id)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  double cur_t = MPI_Wtime() - t;
  halo_param += cur_t;
  t = MPI_Wtime();
#endif
 
  int totalToBeSent = haloInfos[gbl_offload_halo_id].totalToBeSent;
  int num_neighbors = haloInfos[gbl_offload_halo_id].requests.size()/2;
  if (num_neighbors < 0 || num_neighbors >= 1024) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[%d] %s:%d num_neighbors=%d\n", rank, __FILE__, __LINE__, num_neighbors);
    fflush(stdout);
  }

  // download send buffer
  #pragma offload_transfer target(mic : micid) signal(&st) \
    out(gbl_offload_halo_send_buf : length(totalToBeSent) alloc_if(0) free_if(0) align(64))
  {}

  // Post receives first
  if (num_neighbors > 0) {
    int errorCode = MPI_Startall(num_neighbors, &haloInfos[gbl_offload_halo_id].requests[0]);
    handleMPIError(errorCode);
  }

  #pragma offload target(mic : micid) wait(&st)
  {}

#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  cur_t = MPI_Wtime() - t;
  halo_pciedown += cur_t;
  t = MPI_Wtime();
#endif
 
  // send to each neighbor
  if (num_neighbors > 0) {
    int errorCode = MPI_Startall(
      num_neighbors, &haloInfos[gbl_offload_halo_id].requests[num_neighbors]);
    handleMPIError(errorCode);
  }

  // wait for recvs
  int errorCode = MPI_Waitall(
    num_neighbors,
    &haloInfos[gbl_offload_halo_id].requests[0],
    MPI_STATUSES_IGNORE); 
  handleMPIError(errorCode);
  
#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  cur_t = MPI_Wtime() - t;
  halo_fabrics += cur_t;
  t = MPI_Wtime();
#endif

  // update receive buffer
  double *recvBuffer = haloVectors[gbl_offload_halo_id];
  int recvOffset = haloOffset[gbl_offload_halo_id];

  #pragma offload_transfer target(mic : micid) signal(&st) \
    in(recvBuffer[0:totalToBeSent] : into(recvBuffer[recvOffset:totalToBeSent]) alloc_if(0) free_if(0) align(64))
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  cur_t = MPI_Wtime() - t;
  halo_pcieup += cur_t;
  t = MPI_Wtime();
#endif

  // wait for sends
  errorCode = MPI_Waitall(
    num_neighbors,
    &haloInfos[gbl_offload_halo_id].requests[num_neighbors],
    MPI_STATUSES_IGNORE); 
  handleMPIError(errorCode);
  
#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  cur_t = MPI_Wtime() - t;
  halo_fabrics += cur_t;
#endif

  return 0;
}

void halo_finalize_wrapper(int micid) {
  char st;

  // download id
  #pragma offload_transfer target(mic : micid) signal(&st) \
    out(gbl_offload_halo_id)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  for (int i = 0; i < haloInfos[gbl_offload_halo_id].requests.size(); ++i) {
    int flag;
    int errorCode = MPI_Test(&haloInfos[gbl_offload_halo_id].requests[i], &flag, MPI_STATUS_IGNORE);
    handleMPIError(errorCode);
    if (!flag) {
      errorCode = MPI_Cancel(&haloInfos[gbl_offload_halo_id].requests[i]);
      handleMPIError(errorCode);
    }
    errorCode = MPI_Wait(&haloInfos[gbl_offload_halo_id].requests[i], MPI_STATUS_IGNORE);
    handleMPIError(errorCode);
    errorCode = MPI_Request_free(&haloInfos[gbl_offload_halo_id].requests[i]);
    handleMPIError(errorCode);
  }
  haloInfos[gbl_offload_halo_id].requests.clear();
}

int double_allreduce_wrapper(int type, int micid) {
  char st;
  double tmp_comm;

  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_dbl)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  tmp_comm = gbl_offload_dbl;
  int errorCode;

  switch(type) {
    case HPCG_OFFLOAD_ALLRED_DBL_SUM:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_DBL_MAX:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_DBL_MIN:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
      break;
    default:
      cout << "Offload Driver: Fatal Error: double allredice wrapper!!" << endl;
      return -1;
  }

  handleMPIError(errorCode);
  gbl_offload_dbl = tmp_comm;

  #pragma offload_transfer target(mic : micid) signal(&st) in(gbl_offload_dbl)
  {}
  
  #pragma offload target(mic : micid) wait(&st)
  {}

  return 0;
}

int int_allreduce_wrapper(int type, int micid) {
  char st;
  int tmp_comm;

  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_int)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  tmp_comm = gbl_offload_int;
  int errorCode;

  switch(type) {
    case HPCG_OFFLOAD_ALLRED_INT_SUM:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_INT_MAX:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_INT_MIN:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
      break;
    default:
      cout << "Offload Driver: Fatal Error: integer allreduce wrapper!!" << endl;
      return -1;
  }

  handleMPIError(errorCode);
  gbl_offload_int = tmp_comm;
  
  #pragma offload_transfer target(mic : micid) signal(&st) in(gbl_offload_int)
  {}
          
  #pragma offload target(mic : micid) wait(&st)
  {}

  return 0;
}
  
int lint_allreduce_wrapper(int type, int micid) {
  char st;
  long long tmp_comm;

  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_lint)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  tmp_comm = gbl_offload_lint;
  int errorCode;

  switch(type) {
    case HPCG_OFFLOAD_ALLRED_LINT_SUM:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_LINT_MAX:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
      break;
    case HPCG_OFFLOAD_ALLRED_LINT_MIN:
#ifndef HPCG_NO_MPI
      errorCode = MPI_Allreduce(MPI_IN_PLACE, &tmp_comm, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
      break;
    default:
      cout << "Offload Driver: Fatal Error: long integer allreduce wrapper!!" << endl;
      return -1;
  }

  handleMPIError(errorCode);
  gbl_offload_lint = tmp_comm;

  #pragma offload_transfer target(mic : micid) signal(&st) in(gbl_offload_lint)
  {}
  
  #pragma offload target(mic : micid) wait(&st)
  {}

  return 0;
}

void double_gather_wrapper(int micid) {
  char st;

  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_dbl)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  int size, rank;

  int errorCode = MPI_Comm_size(MPI_COMM_WORLD, &size);
  handleMPIError(errorCode);
  errorCode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  handleMPIError(errorCode);

  errorCode = MPI_Gather(
    (void *)(&gbl_offload_dbl), 1, MPI_DOUBLE,
    gbl_offload_dbl_gather, 1, MPI_DOUBLE,
    0, MPI_COMM_WORLD);
  handleMPIError(errorCode);

  if (0 == rank) {
    #pragma offload_transfer target(mic : micid) signal(&st) \
      in(gbl_offload_dbl_gather : length(size) alloc_if(0) free_if(0) align(64))
    {}
    
    #pragma offload target(mic : micid) wait(&st)
    {}
  }
}

void int_send_to_root_wrapper(int micid) {
  char st;

  #pragma offload_transfer target(mic : micid) signal(&st) out(gbl_offload_int)
  {}

  #pragma offload target(mic : micid) wait(&st)
  {}

  int rank;

  int errorCode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  handleMPIError(errorCode);

  errorCode = MPI_Send(
    (void *)(&gbl_offload_int), 1, MPI_INT,
    0, rank, MPI_COMM_WORLD);
  handleMPIError(errorCode);
}

#endif // HPCG_NO_MPI

const string toString(BLOCK_METHOD method)
{
  if (GEOMETRIC == method) {
    return "GEOMETRIC";
  }
  else if (GREEDY == method) {
    return "GREEDY";
  }
  else if (METIS == method) {
    return "METIS";
  }
  else {
    assert(MATCHING == method);
    return "MATCHING";
  }
}

static int startswith(const char * s, const char * prefix) {
  size_t n = strlen( prefix );
  if (strncmp( s, prefix, n ))
    return 0;
  return 1;
}

HPCG_Params::HPCG_Params() :
  comm_size(1), comm_rank(0), numThreads(1),
  nx(10), ny(10), nz(10),
  useEsb(true),
  runRef(true),
#ifdef HPCG_NO_MPI
  overlap(false),
#else
  overlap(true),
#endif
  measureImbalance(false),
  blockMethod(MATCHING),
  multiColorFromThisLevel(2),
  blockColorUpToThisLevel(1),
  colorBlockSize(4),
  numberOfPresmootherSteps(1), numberOfPostsmootherSteps(1), numberOfMgLevels(4),
  fuseSpmv(true),
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
  char fname[80];
  int i, j, iparams[4] = { 0, 0, 0, 1 };
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
    { "log-level", required_argument, 0, 'l', },
    { "log-out", required_argument, 0, 'g' },
    { "help", no_argument, 0, 'h' },
    { "run-real-ref", optional_argument, 0, 's' },
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
        if (0 == params.comm_rank) {
          fprintf(
            stderr,
            "usage: %s [--nx=nx] [--ny=ny] [--nz=nz] [--n=n] [--t=t]\n"
            "--n: set nx, ny, and nz to the same number\n"
            "--t: time to run\n",
            argv[0]);
        }
        exit(-1);

      default: break;
    }
  }

  if (optind < argc - 1) {
    iparams[0] = atoi(argv[optind]);
    iparams[1] = atoi(argv[optind + 1]);
    iparams[2] = atoi(argv[optind + 2]);
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
  //MPI_Bcast( iparams, 4, MPI_INT, 0, MPI_COMM_WORLD );
#endif

  params.nx = iparams[0];
  params.ny = iparams[1];
  params.nz = iparams[2];

  if (0 == params.comm_rank) {
    if (params.nx%8 || params.ny%8 || params.nz%8) {
      fprintf(stderr, "input size must be a multiple of 8\n");
      exit(-1);
    }
    if (params.overlap && (params.nx < 24 || params.ny < 24 || params.nz < 24)) {
      fprintf(stderr, "Local dimension size of each direction must be at least 24 when communication overlapping is used. Consider increasing local dimensions or disable communication overlapping\n");
      exit(-1);
    }
  }

  params.runningTime = iparams[3];

#ifdef HPCG_NO_MPI
  params.comm_rank = 0;
  params.comm_size = 1;
#else
  MPI_Comm_rank( MPI_COMM_WORLD, &params.comm_rank );
  MPI_Comm_size( MPI_COMM_WORLD, &params.comm_size );
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
        printf("block-method=%s ", toString(params.blockMethod).c_str());
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
    if (params.fuseSpmv) {
      printf("fuse-spmv ");
    }
    if (params.overlap) {
      printf("overlap ");
    }
    if (!params.runRef) {
      printf("run-ref=0 ");
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
    sprintf( fname, "hpcg_log_%d_%d_%d_%dt_%04d.%02d.%02d.%02d.%02d.%02d.txt",
      params.nx, params.ny, params.nz, omp_get_num_threads(),
      1900 + ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );
  }

  if (0 == params.comm_rank) {
      strncpy(txtReportFile, fname, FILENAME_MAX_LEN);
/*#ifdef OFFLOAD_EVERYTHING
    HPCG_fout.rdbuf(cout.rdbuf());
#else
    HPCG_fout.open(fname);
#endif*/
  }
  else {
/*#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
    char local[15];
    sprintf( local, "%d_", params.comm_rank );
    sprintf( fname, "hpcg_log_%d_%d_%d_%dt_%s%04.d%02d.%02d.%02d.%02d.%02d.txt",
        params.nx, params.ny, params.nz, omp_get_num_threads(),
        local,
        1900 + ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );
    HPCG_fout.open(fname);
#else
    HPCG_fout.open("/dev/null");
#endif*/
  }

  return 0;
}

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  int errorCode = MPI_Init(&argc, &argv);
  handleMPIError(errorCode);

  // Check if Xeon Phi is used somewhere
  int localUseXeonPhi = 2; // 1 means Xeon Phi in symmetric mode, 2 means offload
  int globalUseXeonPhi = 0;
  errorCode = MPI_Allreduce(
    &localUseXeonPhi, &globalUseXeonPhi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  char s1,s2;

  int size = params.comm_size;
  int rank = params.comm_rank;

  if (params.logLevel >= 3 || 2 == params.logLevel && 0 == rank) {
    cout << "[" << rank << "] HPCG_Init finished." << endl;
  }

  int nx = params.nx;
  int ny = params.ny;
  int nz = params.nz;

  int micid;

  gbl_offload_signal = HPCG_OFFLOAD_RUN;
  gbl_offload_dbl = 0.0;
  gbl_offload_int = 0;
  gbl_offload_lint = 0;
  gbl_offload_dbl_gather = (double *)_mm_malloc(sizeof(double)*size, 64);

  // halo proxy init
  gbl_offload_halo_max_neighbors = 26;
  gbl_offload_halo_max_mpi_buffer_size = max<int>(max<int>(nx, ny),nz);
  gbl_offload_halo_max_mpi_buffer_size *= gbl_offload_halo_max_mpi_buffer_size;
  gbl_offload_halo_max_mpi_buffer_size *= gbl_offload_halo_max_neighbors;
  gbl_offload_halo_num_neighbors = 0;
  gbl_offload_halo_id = 0;
  gbl_offload_halo_neighbors = (int*) _mm_malloc(sizeof(int)*gbl_offload_halo_max_neighbors, 64);
  gbl_offload_halo_recv_sizes = (int*) _mm_malloc(sizeof(int)*gbl_offload_halo_max_neighbors, 64);
  gbl_offload_halo_send_sizes = (int*) _mm_malloc(sizeof(int)*gbl_offload_halo_max_neighbors, 64);
  gbl_offload_halo_send_buf = (double*) _mm_malloc(sizeof(double)*gbl_offload_halo_max_mpi_buffer_size, 64);

  haloVectors[HPCG_OFFLOAD_VECTOR_P] = gbl_offload_p =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_Z] = gbl_offload_z =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_XNCOL] = gbl_offload_xncol =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_YNCOL] = gbl_offload_yncol =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_ZNCOL] = gbl_offload_zncol =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_XOVERLAP] = gbl_offload_xoverlap =
    (double *)_mm_malloc(sizeof(double)*(nx + 2)*(ny + 2)*(nz + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_X1] = gbl_offload_x1 =
    (double *)_mm_malloc(sizeof(double)*(nx/2 + 2)*(ny/2 + 2)*(nz/2 + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_X2] = gbl_offload_x2 =
    (double *)_mm_malloc(sizeof(double)*(nx/4 + 2)*(ny/4 + 2)*(nz/4 + 2), 64);
  haloVectors[HPCG_OFFLOAD_VECTOR_X3] = gbl_offload_x3 =
    (double *)_mm_malloc(sizeof(double)*(nx/8 + 2)*(ny/8 + 2)*(nz/8 + 2), 64);
  
#ifndef HPCG_NO_MPI
  int namelen;
  // determine which coprocessor should be used
  char myname[MPI_MAX_PROCESSOR_NAME];
  errorCode = MPI_Get_processor_name(myname, &namelen);
  handleMPIError(errorCode);
  char (*allnames)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc((size_t)size*MPI_MAX_PROCESSOR_NAME*sizeof(char));  
  int *offloadFlags = (int *)malloc(size*sizeof(int));
  int myOffloadFlag = 1;
  errorCode = MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allgather(&myOffloadFlag, 1, MPI_INT, offloadFlags, 1, MPI_INT, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  micid = 0;
  int ranksPerNode = 0, offloadRanksPerNode = 0;
  for (int i = 0; i < size; i++) {
    if(strncmp(myname, allnames[i], namelen) == 0) {
      if ( i == rank ) {
        break;
      } else if ( offloadFlags[i] ) {
        micid++;
      }
    } 
  }
  free(allnames);
  free(offloadFlags);

  // Apply multiple ranks per card
  int num_mics = _Offload_number_of_devices();

  // abort if no MIC is found
  if (num_mics == 0) {
    cout << "no MIC card available on rank " << rank << " on host " << myname << endl;
    sleep(10);
    MPI_Abort(MPI_COMM_WORLD,-1);
    return -1;
  }

  int micproc = micid/num_mics;
  micid = micid%num_mics;

  // reset kmp_affinitiy on MIC
  char* p_micthreads;
  int num_micthreads = 240;
  p_micthreads = getenv("MIC_OMP_NUM_THREADS");
  if (p_micthreads != NULL) {
    num_micthreads = atoi(p_micthreads);
  } else {
    if (rank == 0 && params.logLevel > 0) cout << "WARNING ASSUMING 240 MIC THREADS PER PROCESS!" << endl;
  }
  int startcore = num_micthreads*micproc;
  int endcore = startcore+(num_micthreads-1);

  stringstream pinning;
  pinning << "KMP_AFFINITY=proclist=[";
  for (int i = 0; i < num_micthreads; i++) {
    if (i < (num_micthreads-1))
      pinning << startcore+i+1 << ",";
    else
      pinning << startcore+i+1;
  }
  pinning << "],granularity=thread,explicit";

  //cout << pinning.str() << endl; 
  kmp_set_defaults_target(TARGET_MIC, micid, pinning.str().c_str()); 
#else
  micid = 0;
#endif

  if (params.logLevel >= 3) {
    cout << rank << " Offloadling to Xeon Phi, stand by for finishing Xeon Phi execution..." << endl << endl;
  }

  // create buffers on Xeon Phi
  #pragma offload_transfer target(mic : micid) \
    in(gbl_offload_signal,gbl_offload_dbl,gbl_offload_int,gbl_offload_lint) \
    in(gbl_offload_halo_max_neighbors,gbl_offload_halo_max_mpi_buffer_size,gbl_offload_halo_num_neighbors,gbl_offload_halo_id) \
    nocopy(gbl_offload_halo_neighbors : length(gbl_offload_halo_max_neighbors) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_halo_recv_sizes : length(gbl_offload_halo_max_neighbors) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_halo_send_sizes : length(gbl_offload_halo_max_neighbors) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_halo_send_buf : length(gbl_offload_halo_max_mpi_buffer_size) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_p : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_z : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_xncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_yncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_zncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_xoverlap : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_x1 : length((nx/2 + 2)*(ny/2 + 2)*(nz/2 + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_x2 : length((nx/4 + 2)*(ny/4 + 2)*(nz/4 + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_x3 : length((nx/8 + 2)*(ny/8 + 2)*(nz/8 + 2)) alloc_if(1) free_if(0) align(64)) \
    nocopy(gbl_offload_dbl_gather : length(size) alloc_if(1) free_if(0) align(64))
  {}

  if (params.logLevel >= 3) {
    cout << rank << " Buffers created in Xeon Phi, stand by for finishing Xeon Phi execution..." << endl << endl;
  }

  // marshal HPCG_Params
  // params.numThreads will be set in the card

  int runningTime = params.runningTime;

  // ignore yamlFileName, matrixDumpFileName, inputMatrixFileName
  
  bool useEsb = params.useEsb;
  bool runRef = params.runRef;
  bool overlap = params.overlap;
  bool measureImbalance = params.measureImbalance;

  BLOCK_METHOD blockMethod = params.blockMethod;

  int multiColorFromThisLevel = params.multiColorFromThisLevel;
  int blockColorUpToThisLevel = params.blockColorUpToThisLevel;
  int colorBlockSize = params.colorBlockSize;
  int numberOfMgLevels = params.numberOfMgLevels;
  int numberOfPresmootherSteps = params.numberOfPresmootherSteps;
  int numberOfPostsmootherSteps = params.numberOfPostsmootherSteps;

  bool fuseSpmv = params.fuseSpmv;
  bool runRealRef = params.runRealRef;

  int logLevel = params.logLevel;

  if (rank == 0) {
    yamlReport     = (char *)_mm_malloc(sizeof(char)*CONTENTS_MAX_LEN, 64);
    yamlReportFile = (char *)_mm_malloc(sizeof(char)*FILENAME_MAX_LEN, 64);
    txtReport      = (char *)_mm_malloc(sizeof(char)*CONTENTS_MAX_LEN, 64);
    #pragma offload target(mic : micid) \
     nocopy (yamlReport     : length(CONTENTS_MAX_LEN) alloc_if(1) free_if(0)) \
     nocopy (yamlReportFile : length(FILENAME_MAX_LEN) alloc_if(1) free_if(0)) \
     nocopy (txtReport      : length(CONTENTS_MAX_LEN) alloc_if(1) free_if(0))
    {}
  }

  #pragma offload target(mic : micid) \
    in(size, rank, nx, ny, nz, runningTime, useEsb, runRef, overlap, measureImbalance, blockMethod, multiColorFromThisLevel, colorBlockSize, numberOfMgLevels, numberOfPresmootherSteps, numberOfPostsmootherSteps, fuseSpmv, runRealRef, logLevel) \
    signal(&s1)
  {
    HPCG_Params paramsInCard;

    paramsInCard.comm_size = size;
    paramsInCard.comm_rank = rank;

    paramsInCard.nx = nx;
    paramsInCard.ny = ny;
    paramsInCard.nz = nz;

    paramsInCard.runningTime = runningTime;

    paramsInCard.useEsb = useEsb;
    paramsInCard.runRef = runRef;
    paramsInCard.overlap = overlap;
    paramsInCard.measureImbalance = measureImbalance;

    paramsInCard.blockMethod = blockMethod;

    paramsInCard.multiColorFromThisLevel = multiColorFromThisLevel;
    paramsInCard.blockColorUpToThisLevel = blockColorUpToThisLevel;
    paramsInCard.colorBlockSize = colorBlockSize;
    paramsInCard.numberOfMgLevels = numberOfMgLevels;
    paramsInCard.numberOfPresmootherSteps = numberOfPresmootherSteps;
    paramsInCard.numberOfPostsmootherSteps = numberOfPostsmootherSteps;

    paramsInCard.fuseSpmv = fuseSpmv;
    paramsInCard.runRealRef = runRealRef;

    paramsInCard.logLevel = logLevel;

    HPCG_Run(paramsInCard);
  }

  while(gbl_offload_signal != HPCG_OFFLOAD_STOP) {
    #pragma offload_transfer target(mic : micid) out(gbl_offload_signal) signal(&s2)
    {}
    
    #pragma offload target(mic : micid) wait(&s2)
    {}

    if (gbl_offload_signal != HPCG_OFFLOAD_RUN) {
      switch(gbl_offload_signal) {
#ifndef HPCG_NO_MPI
        case HPCG_OFFLOAD_BARRIER:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Barrier request" << endl;
          }
          errorCode = MPI_Barrier(MPI_COMM_WORLD);
          handleMPIError(errorCode);
          break;
        case HPCG_OFFLOAD_ALLRED_DBL_SUM:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Allreduce(double, sum) request" << endl;
          }
          double_allreduce_wrapper(gbl_offload_signal, micid);
          break;
        case HPCG_OFFLOAD_ALLRED_DBL_MIN:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Allreduce(double, min) request" << endl;
          }
          double_allreduce_wrapper(gbl_offload_signal, micid);
          break;
        case HPCG_OFFLOAD_ALLRED_DBL_MAX:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Allreduce(double, max) request" << endl;
          }
          double_allreduce_wrapper(gbl_offload_signal, micid);
          break;
        case HPCG_OFFLOAD_ALLRED_INT_SUM:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Allreduce(integer, sum) request" << endl;
          }
          int_allreduce_wrapper(gbl_offload_signal, micid);
          break;
        case HPCG_OFFLOAD_ALLRED_LINT_SUM:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received MPI_Allreduce(long integer, sum) request" << endl;
          }
          lint_allreduce_wrapper(gbl_offload_signal, micid);
          break;
        case HPCG_OFFLOAD_HALO_SETUP:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received halo exchange setup request" << endl;
          }
          halo_setup_wrapper(micid);
          break;
        case HPCG_OFFLOAD_HALO_EXCH:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received halo exchange request" << endl;
          }
          halo_exchange_wrapper(micid);
          break;
        case HPCG_OFFLOAD_HALO_FINALIZE:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received halo exchange finalize request" << endl;
          }
          halo_finalize_wrapper(micid);
          break;
        case HPCG_OFFLOAD_GATHER_DBL:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received gather double request" << endl;
          }
          double_gather_wrapper(micid);
          break;
        case HPCG_OFFLOAD_SEND_TO_ROOT_INT:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received send to root int request" << endl;
          }
          int_send_to_root_wrapper(micid);
          break;
#endif // HPCG_NO_MPI
        case HPCG_OFFLOAD_SET_PROBLEM_SIZE:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received set problem size request" << endl;
          }
          set_problem_size(micid);
          break;
        case HPCG_OFFLOAD_START_TIMING:
          if (params.logLevel >= 3) {
            cout << rank << " Offload Driver received start timing request" << endl;
          }
#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD_HALO_PROFILING
          halo_param = 0;
          halo_pciedown = 0;
          halo_fabrics = 0;
          halo_pcieup = 0;
#endif
#endif
          break;
        case HPCG_OFFLOAD_CHECK_FAILED:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received check failed status!" << endl;
          }
#ifndef HPCG_NO_MPI
          MPI_Finalize();
#endif
          return 1;
        case HPCG_OFFLOAD_STOP:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver received end of HPCG request! Huhu!" << endl;
          }
          break;
        default:
          if (params.logLevel >= 5) {
            cout << rank << " Offload Driver: Fatal Error!!" << endl;
          }
          return -1;
      }

      if (gbl_offload_signal != HPCG_OFFLOAD_STOP) {
        gbl_offload_signal = HPCG_OFFLOAD_RUN;

        #pragma offload_transfer target(mic : micid) in(gbl_offload_signal) signal(&s2)
        {}

        #pragma offload target(mic : micid) wait(&s2)
        {}
      }
    }
  }

  #pragma offload target(mic : micid) wait(&s1)
  {}

  // free buffers on Xeon Phi
  #pragma offload_transfer target(mic : micid) \
    nocopy(gbl_offload_halo_neighbors : length(gbl_offload_halo_max_neighbors) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_halo_recv_sizes : length(gbl_offload_halo_max_neighbors) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_halo_send_sizes : length(gbl_offload_halo_max_neighbors) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_halo_send_buf : length(gbl_offload_halo_max_mpi_buffer_size) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_p : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_z : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_xncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_yncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_zncol : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_xoverlap : length((nx + 2)*(ny + 2)*(nz + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_x1 : length((nx/2 + 2)*(ny/2 + 2)*(nz/2 + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_x2 : length((nx/4 + 2)*(ny/4 + 2)*(nz/4 + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_x3 : length((nx/8 + 2)*(ny/8 + 2)*(nz/8 + 2)) alloc_if(0) free_if(1) align(64)) \
    nocopy(gbl_offload_dbl_gather : length(size) alloc_if(0) free_if(1) align(64))
  {}

  if (params.logLevel >= 3) {
    cout << rank << " ... offload to Xeon Phi returned." << endl << endl;
  }

#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD_HALO_PROFILING   
  double max_halo_param = halo_param;
  double max_halo_pciedown = halo_pciedown;
  double max_halo_fabrics = halo_fabrics;
  double max_halo_pcieup = halo_pcieup;

  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_param, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_pciedown, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_fabrics, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);
  errorCode = MPI_Allreduce(MPI_IN_PLACE, &max_halo_pcieup, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  handleMPIError(errorCode);

  if (rank == 0) {
    #pragma offload target(mic) \
            out(yamlReportFile : length(FILENAME_MAX_LEN) free_if(0)) \
            out(yamlReport     : length(CONTENTS_MAX_LEN) free_if(0)) \
            out(txtReport      : length(CONTENTS_MAX_LEN) free_if(0))
    {}

    std::ofstream HPCG_fout;
    HPCG_fout.open( txtReportFile );
    HPCG_fout << txtReport;
    HPCG_fout.close();
    if (params.yamlFileName[0]) {
      HPCG_fout.open( string(params.yamlFileName) + ".yaml" );
    } else {
      HPCG_fout.open( yamlReportFile );
    }
    HPCG_fout << yamlReport;
    HPCG_fout.close();

    if (params.logLevel > 0) {
      cout << endl;
      cout << "Host-side halo-exchange summary:" << endl;
      cout << "   parameter download:  " << max_halo_param << endl;
      cout << "   PCI buffer download: " << max_halo_pciedown << endl;
      cout << "   MPI Irecv/Send/Wait: " << max_halo_fabrics << endl;
      cout << "   PCI buffer upload:   " << max_halo_pcieup << endl;
      cout << endl;       
    }
  }
#endif
  MPI_Finalize();
#endif

  _mm_free(gbl_offload_dbl_gather);

  _mm_free(gbl_offload_halo_neighbors);
  _mm_free(gbl_offload_halo_recv_sizes);
  _mm_free(gbl_offload_halo_send_sizes);
  _mm_free(gbl_offload_halo_send_buf);

  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_P]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_Z]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_XNCOL]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_YNCOL]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_ZNCOL]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_XOVERLAP]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_X1]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_X2]);
  _mm_free(haloVectors[HPCG_OFFLOAD_VECTOR_X3]);

  return 0 ;
}
