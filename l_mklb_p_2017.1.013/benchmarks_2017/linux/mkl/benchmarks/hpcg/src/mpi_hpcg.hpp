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

#ifndef MPI_HPCG_HPP
#define MPI_HPCG_HPP

#include "mpi_hpcg_api.hpp"

/* Types */

#define MPI_Request HPCG_MPI_Request
#define MPI_Status HPCG_MPI_Status

#define MPI_Datatype HPCG_MPI_Datatype
#define MPI_Op HPCG_MPI_Op
#define MPI_Comm HPCG_MPI_Comm

#define MPI_CHAR HPCG_MPI_CHAR

/* Contants */

#define MPI_ANY_SOURCE HPCG_MPI_ANY_SOURCE

#define MPI_MAX_PROCESSOR_NAME HPCG_MPI_MAX_PROCESSOR_NAME
#define MPI_IN_PLACE HPCG_MPI_IN_PLACE

#define MPI_COMM_NULL HPCG_MPI_COMM_NULL
#define MPI_COMM_WORLD HPCG_MPI_COMM_WORLD
#define MPI_BYTE HPCG_MPI_BYTE
#define MPI_DOUBLE HPCG_MPI_DOUBLE
#define MPI_DOUBLE_INT HPCG_MPI_DOUBLE_INT
#define MPI_DOUBLE_PRECISION HPCG_MPI_DOUBLE_PRECISION
#define MPI_FLOAT HPCG_MPI_FLOAT
#define MPI_LONG_LONG_INT HPCG_MPI_LONG_LONG_INT
#define MPI_INT HPCG_MPI_SCALAPACK_INT
#define MPI_MAX HPCG_MPI_MAX
#define MPI_MAXLOC HPCG_MPI_MAXLOC
#define MPI_MIN HPCG_MPI_MIN
#define MPI_SUM HPCG_MPI_SUM

#define MPI_REQUEST_NULL HPCG_MPI_REQUEST_NULL
#define MPI_STATUSES_IGNORE HPCG_MPI_STATUSES_IGNORE
#define MPI_STATUS_IGNORE HPCG_MPI_STATUS_IGNORE

#define MPI_SIMILAR HPCG_MPI_SIMILAR
#define MPI_SUCCESS HPCG_MPI_SUCCESS
#define MPI_UNDEFINED HPCG_MPI_UNDEFINED

/* (Some) MPI functions */

#define MPI_Init HPCG_MPI_Initmpi
#define MPI_Finalize HPCG_MPI_Finalize
#define MPI_Get_processor_name HPCG_MPI_Get_processor_name
#define MPI_Wtime HPCG_MPI_Wtime
#define MPI_Comm_rank HPCG_MPI_Comm_rank
#define MPI_Comm_size HPCG_MPI_Comm_size
#define MPI_Request_free HPCG_MPI_Request_free
#define MPI_Send HPCG_MPI_Send
#define MPI_Irecv HPCG_MPI_Irecv
#define MPI_Bcast HPCG_MPI_Bcast
#define MPI_Wait HPCG_MPI_Wait
#define MPI_Barrier HPCG_MPI_Barrier
#define MPI_Waitall HPCG_MPI_Waitall
#define MPI_Allreduce HPCG_MPI_Allreduce
#define MPI_Recv_init HPCG_MPI_Recv_init
#define MPI_Send_init HPCG_MPI_Send_init
#define MPI_Startall HPCG_MPI_Startall
#define MPI_Allgather HPCG_MPI_Allgather
#define MPI_Iallreduce HPCG_MPI_Iallreduce

#define MPI_Error_string HPCG_MPI_Error_string
#define MPI_Gather HPCG_MPI_Gather
#define MPI_Test HPCG_MPI_Test
#define MPI_Testany HPCG_MPI_Testany
#define MPI_Cancel HPCG_MPI_Cancel
#define MPI_Abort HPCG_MPI_Abort

#endif // MPI_HPCG_HPP

