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
 @file ExchangeHalo.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
// Compile this routine only if running in parallel
#ifndef HPCG_OFFLOAD
#include "mpi_hpcg.hpp"
#endif
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include "mytimer.hpp"
#include <cstdlib>
#include "OptimizeProblem.hpp"

#ifdef HPCG_OFFLOAD
#include "offloadExtHpcgLib.hpp"
#endif

using namespace std;

extern bool inTimedCgLoop;

#ifndef HPCG_OFFLOAD
void ExchangeHalo_ref(const SparseMatrix & A, Vector & x) {

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  double * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }

  delete [] request;

  return;
}
#endif

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix & A, Vector & x, double *haloTimes /* = NULL*/, const HPCG_Params& params /* = HPCG_Params()*/, bool overlap /*=false*/) {
  if (!inTimedCgLoop) haloTimes = NULL;

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  
  if (num_neighbors < 0 || num_neighbors >= 1024) {
    printf("[%d] %s:%d num_neighbors=%d\n", A.geom->rank, __FILE__, __LINE__, num_neighbors);
    fflush(stdout);
  }
  assert(num_neighbors >= 0 && num_neighbors < 1024);
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend;
  if (A.optimizationData && ((OptimizationData *)A.optimizationData)->elementsToSend) {
    elementsToSend = ((OptimizationData *)A.optimizationData)->elementsToSend;
  }
  else {
    elementsToSend = A.elementsToSend;
  }

  double * const xv = x.values;

  double t;

  VectorOptimizationData *optData = (VectorOptimizationData *)x.optimizationData;
  if (!optData) {
    printf("[%d] %s:%d\n", A.geom->rank, __FILE__, __LINE__);
    fflush(stdout);
  }
  assert(optData);

#ifdef HPCG_OFFLOAD
  if (!optData->persistentCommSetup) {
    t = mytimer();

    gbl_offload_halo_id = optData->id;

    gbl_offload_halo_num_neighbors = num_neighbors;

    // fill constant information buffers
    for (int i = 0; i < num_neighbors; i++) {
      gbl_offload_halo_neighbors[i] = neighbors[i];
      gbl_offload_halo_recv_sizes[i] = receiveLength[i];
      gbl_offload_halo_send_sizes[i] = sendLength[i];
    }

    gbl_offload_signal = HPCG_OFFLOAD_HALO_SETUP;

    while (gbl_offload_signal != HPCG_OFFLOAD_RUN) {};

    if (haloTimes) {
      haloTimes[SETUP_PERSISTENT_COMM] += mytimer() - t;
    }

    optData->persistentCommSetup = true;
  }

  t = mytimer();

#pragma omp parallel for
  for (local_int_t i=0; i<totalToBeSent; i++) gbl_offload_halo_send_buf[i] = xv[elementsToSend[i]];

  if (haloTimes) {
    haloTimes[FILL_UP_SEND_BUF] += mytimer() - t;
  }

  gbl_offload_halo_id = optData->id;
  gbl_offload_signal = HPCG_OFFLOAD_HALO_EXCH;

  if (!overlap) WaitHalo(A, x, haloTimes);

#else // HPCG_OFFLOAD

//#define OLD_HALO
#ifdef OLD_HALO
  return ExchangeHalo_ref(A, x);
#endif

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  //
  // Fill up send buffer
  //

  t = mytimer();
#pragma omp parallel for
  for (local_int_t i=0; i<totalToBeSent; i++) {
    A.sendBuffer[i] = xv[elementsToSend[i]];
  }
  if (haloTimes) {
    haloTimes[FILL_UP_SEND_BUF] += mytimer() - t;
  }

#if defined(HPCG_RMA_FENCE) | defined(HPCG_RMA_FLUSH)
#ifdef HPCG_RMA_FENCE
  MPI_Win_fence(0, *((MPI_Win *)A.send_window));
#endif
  MPI_Aint target_disp = 0;
  for (int i = 0; i < num_neighbors; ++i) {
    local_int_t n_recv = receiveLength[i];
    local_int_t n_send = sendLength[i];
    assert(n_recv == n_send);
    MPI_Get(x_external, (int)n_recv, MPI_DOUBLE, neighbors[i],
            target_disp, (int)n_send, MPI_DOUBLE, *((MPI_Win *)A.send_window));
    x_external += n_recv; /* pointer arithmetic increments in units of sizeof(type=double) */
    target_disp += n_send * sizeof(double); /* integer artihmetic requires explicit type-size increment */
  }
#ifdef HPCG_RMA_FENCE
  MPI_Win_fence(0, *((MPI_Win *)A.send_window));
#elif defined(HPCG_RMA_FLUSH)
  MPI_Win_flush_all(*((MPI_Win *)A.send_window));
#else // HPCG_RMA_{FENCE,FLUSH}
#error
#endif

#else // HPCG_RMA_FENCE | HPCG_RMA_FLUSH
  // Lazy persistent communication setup
  if (!optData->persistentCommSetup) {
    t = mytimer();

    vector<MPI_Request> *haloRequests = new vector<MPI_Request>(2*num_neighbors);
    if (!haloRequests) {
      printf("[%d] %s:%d\n", A.geom->rank, __FILE__, __LINE__);
      fflush(stdout);
    }
    assert(haloRequests);
    optData->haloRequests = haloRequests;

    for (int i = 0; i < num_neighbors; ++i) {
      local_int_t n_recv = receiveLength[i];

      int errorCode = MPI_Recv_init(
        x_external, n_recv, MPI_DOUBLE, neighbors[i],
        MPI_MY_TAG + optData->id, MPI_COMM_WORLD, &(*haloRequests)[i]);
      handleMPIError(errorCode);

      x_external += n_recv;

      local_int_t n_send = sendLength[i];

      errorCode = MPI_Send_init(
        sendBuffer, n_send, MPI_DOUBLE, neighbors[i],
        MPI_MY_TAG + optData->id, MPI_COMM_WORLD, &(*haloRequests)[i + num_neighbors]);
      handleMPIError(errorCode);

      sendBuffer += n_send;
    }
    if (sendBuffer - A.sendBuffer != A.totalToBeSent) {
      printf(
        "[%d] %s:%d (sendBuffer - A.sendBuffer)=%ld A.totalToBeSent=%d\n",
        A.geom->rank, __FILE__, __LINE__,
        sendBuffer - A.sendBuffer, A.totalToBeSent);
      fflush(stdout);
    }
    assert(sendBuffer - A.sendBuffer == A.totalToBeSent);
    if (x_external - xv != x.localLength) {
      printf(
        "[%d] %s:%d (x_external - xv)=%ld x.localLength=%d\n",
        A.geom->rank, __FILE__, __LINE__,
        x_external - xv, x.localLength);
      fflush(stdout);
    }
    assert(x_external - xv == x.localLength);

    if (haloTimes) {
      haloTimes[SETUP_PERSISTENT_COMM] += mytimer() - t;
    }

    optData->persistentCommSetup = true;
  } // !optData->persistentCommSetup

  if (params.measureImbalance) {
    t = mytimer();

    int errorCode = MPI_Barrier(MPI_COMM_WORLD);
    handleMPIError(errorCode);

    if (haloTimes) {
      haloTimes[IMBALANCE] += mytimer() - t;
    }
  }

  //
  // Send to each neighbor
  //

  if (num_neighbors > 0) {
    t = mytimer();

    vector<MPI_Request> *haloRequests =
      (vector<MPI_Request> *)optData->haloRequests;

    int errorCode = MPI_Startall(num_neighbors*2, &(*haloRequests)[0]);
    handleMPIError(errorCode);

    if (haloTimes) {
      haloTimes[START_SEND] += mytimer() - t;
    }
  }
#endif // !HPCG_RMA_FENCE && !HPCG_RMA_FLUSH

  //
  // Complete the reads issued above
  //

  if (!overlap) WaitHalo(A, x, haloTimes);

#endif // HPCG_OFFLOAD

  return;
}

void WaitHalo(const SparseMatrix& A, Vector& x, double *haloTimes)
{
  if (!inTimedCgLoop) haloTimes = NULL;

  double t = mytimer();

#ifdef HPCG_OFFLOAD
  while (HPCG_OFFLOAD_RUN != gbl_offload_signal) {};

  if (haloTimes) {
    haloTimes[WAIT] += mytimer() - t;
  }
#else // HPCG_OFFLOAD
  if (!x.optimizationData) {
    // code for debugging strange deadlock in Stampede with >= 2K nodes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);
    fflush(stdout);
  }
  assert(x.optimizationData);

#ifdef HPCG_RMA_FENCE
  MPI_Win_fence(0, *((MPI_Win *)A.send_window));
#elif defined(HPCG_RMA_FLUSH)
  MPI_Win_flush_all(*((MPI_Win *)A.send_window));
#else
  VectorOptimizationData *optData = (VectorOptimizationData *)x.optimizationData;
  if (!optData->persistentCommSetup) {
    // code for debugging strange deadlock in Stampede with >= 2K nodes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);
    fflush(stdout);
  }
  assert(optData->persistentCommSetup);

  vector<MPI_Request> *haloRequests = (vector<MPI_Request> *)optData->haloRequests;
  if (!haloRequests) {
    // code for debugging strange deadlock in Stampede with >= 2K nodes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[%d] %s:%d\n", rank, __FILE__, __LINE__);
    fflush(stdout);
  }
  assert(haloRequests);
  if (haloRequests->size() < 0 || haloRequests->size() >= 1024) {
    // code for debugging strange deadlock in Stampede with >= 2K nodes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf(
      "[%d] %s:%d haloRequests->size()=%ld\n",
      rank, __FILE__, __LINE__, haloRequests->size());
    fflush(stdout);
  }
  assert(haloRequests->size() >= 0 && haloRequests->size() < 1024);

  // Wait for receive
  int errorCode = MPI_Waitall(
    haloRequests->size(),
    &(*haloRequests)[0],
    MPI_STATUSES_IGNORE);
  handleMPIError(errorCode);
#endif // !HPCG_RMA_FENCE && !HPCG_RMA_FLUSH

  if (haloTimes) {
    haloTimes[WAIT] += mytimer() - t;
  }
#endif // HPCG_OFFLOAD
}

#endif
// ifndef HPCG_NO_MPI
