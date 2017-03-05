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

#if !defined(HPCG_NO_MPI) && !defined(HPCG_OFFLOAD)
#include "mpi_hpcg.hpp"
#endif

#include "AllReduce.hpp"
#include "mytimer.hpp"

int AllReduce(
  double& globalResult, double& localResult, double& timeAllReduce)
{
  double t0 = mytimer();
#ifdef HPCG_NO_MPI
  globalResult = localResult;
#else
#ifdef HPCG_OFFLOAD
  gbl_offload_dbl = localResult;
  gbl_offload_signal = HPCG_OFFLOAD_ALLRED_DBL_SUM;
  while(gbl_offload_signal != HPCG_OFFLOAD_RUN) {};
  globalResult = gbl_offload_dbl; 
#else
  // Use MPI's reduce function to collect all partial sums
  globalResult = 0.0;
  int errorCode = MPI_Allreduce(
    &localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  handleMPIError(errorCode);
#endif
#endif // HPCG_NO_MPI
  timeAllReduce += mytimer() - t0;

  return 0;
}

#if !defined(HPCG_NO_MPI) && !defined(HPCG_OFFLOAD)
HPCG_MPI_Request *do_MPI_Iallreduce(double& globalResult, double& localResult)
{
  MPI_Request *request = new MPI_Request();

  int errorCode = MPI_Iallreduce(
    &localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD,
    request);
  handleMPIError(errorCode);

  return (HPCG_MPI_Request *)request;
}

void do_MPI_Wait(HPCG_MPI_Request *request)
{
  int errorCode = MPI_Wait((MPI_Request *)request, MPI_STATUS_IGNORE);
  handleMPIError(errorCode);

  delete (MPI_Request *)request;
}
#endif
