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

#if defined(HPCG_RMA_FENCE) | defined(HPCG_RMA_FLUSH)
#ifdef HPCG_OFFLOAD
#error "MPI one-sided not supported with HPCG_OFFLOAD"
#endif
#include "mpi_hpcg.h"
#endif

#include "SparseMatrix.hpp"

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

  A.optimizationData = 0;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#ifdef HPCG_NEIGHBORHOOD_COLLECTIVES
  A.halo_neighborhood_comm = new MPI_Comm;
  *((MPI_Comm *)A.halo_neighborhood_comm) = MPI_COMM_NULL;
#elif defined(HPCG_RMA_FENCE) | defined(HPCG_RMA_FLUSH)
  /* HPCG_RMA_METHOD = PSCW, FENCE, LOCK, FLUSH all use this */
  A.send_window = new MPI_Win;
  *((MPI_Win *)A.send_window) = MPI_WIN_NULL;
#endif
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  A.localToGlobalMap = NULL;
  A.boundaryRows = NULL;
  return;
}
