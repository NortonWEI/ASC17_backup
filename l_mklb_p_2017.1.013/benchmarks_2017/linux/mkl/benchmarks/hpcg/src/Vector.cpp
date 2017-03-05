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

#include <vector>

#ifndef HPCG_NO_MPI
#ifdef HPCG_OFFLOAD
#include <offloadExtHpcgLib.hpp>
#else
#include "mpi_hpcg.hpp"
#endif // HPCG_OFFLOAD
#endif // HPCG_NO_MPI

#include "Vector.hpp"
#include "hpcg.hpp"

#include "SpMP/MemoryPool.hpp"

void DeleteVector(Vector& v)
{
  if (!SpMP::MemoryPool::getSingleton()->contains(v.values)) {
#ifdef USE_LARGE_PAGE
    free_huge_pages(v.values);
#elif defined(HBW_MALLOC)
    hbw_free(v.values);
#else
    _mm_free(v.values);
#endif
  }
  v.localLength = 0;
#ifndef HPCG_NO_MPI
  if (v.optimizationData) {
    VectorOptimizationData *optData = (VectorOptimizationData *)v.optimizationData;

#ifdef HPCG_OFFLOAD
    if (optData->persistentCommSetup) {
      gbl_offload_halo_id = optData->id;

      gbl_offload_signal = HPCG_OFFLOAD_HALO_FINALIZE;
      while (HPCG_OFFLOAD_RUN != gbl_offload_signal) {};
    }
#else
    std::vector<MPI_Request> *haloRequests =
      (std::vector<MPI_Request> *)optData->haloRequests;

    if (haloRequests) {
      for (int i = 0; i < (*haloRequests).size(); ++i) {
        int flag;
        int errorCode = MPI_Test(&(*haloRequests)[i], &flag, MPI_STATUS_IGNORE);
        handleMPIError(errorCode);
        if (!flag) {
          MPI_Cancel(&(*haloRequests)[i]);
          handleMPIError(errorCode);
        }
        errorCode = MPI_Wait(&(*haloRequests)[i], MPI_STATUS_IGNORE);
        handleMPIError(errorCode);
        errorCode = MPI_Request_free(&(*haloRequests)[i]);
        handleMPIError(errorCode);
      }

      delete haloRequests;
    }
#endif // HPCG_OFFLOAD

    delete v.optimizationData;
    v.optimizationData = NULL;
  }
#endif // HPCG_NO_MPI
}
