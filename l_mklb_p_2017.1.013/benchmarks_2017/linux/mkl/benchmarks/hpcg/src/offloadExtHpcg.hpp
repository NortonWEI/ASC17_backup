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

#ifndef OFFLOADEXTHPCG_HPP
#define OFFLOADEXTHPCG_HPP

#define HPCG_OFFLOAD_RUN		0
#define HPCG_OFFLOAD_STOP		1000

#define HPCG_OFFLOAD_BARRIER		1

#define HPCG_OFFLOAD_ALLRED_DBL_SUM	2
#define HPCG_OFFLOAD_ALLRED_DBL_MIN     3
#define HPCG_OFFLOAD_ALLRED_DBL_MAX     4

#define HPCG_OFFLOAD_ALLRED_INT_SUM     5
#define HPCG_OFFLOAD_ALLRED_INT_MIN     6
#define HPCG_OFFLOAD_ALLRED_INT_MAX     7

#define HPCG_OFFLOAD_ALLRED_LINT_SUM	8
#define HPCG_OFFLOAD_ALLRED_LINT_MIN	9
#define HPCG_OFFLOAD_ALLRED_LINT_MAX	10

#define HPCG_OFFLOAD_HALO_EXCH		11
#define HPCG_OFFLOAD_HALO_SETUP		12
#define HPCG_OFFLOAD_HALO_FINALIZE		13

#define HPCG_OFFLOAD_SET_PROBLEM_SIZE 14

#define HPCG_OFFLOAD_GATHER_DBL 15
#define HPCG_OFFLOAD_SEND_TO_ROOT_INT 16

#define HPCG_OFFLOAD_START_TIMING 17

#define HPCG_OFFLOAD_CHECK_FAILED 18

#define HPCG_OFFLOAD_VECTOR_X1 0
#define HPCG_OFFLOAD_VECTOR_X2 1
#define HPCG_OFFLOAD_VECTOR_X3 2
#define HPCG_OFFLOAD_VECTOR_P 3
#define HPCG_OFFLOAD_VECTOR_Z 4
#define HPCG_OFFLOAD_VECTOR_XNCOL 5
#define HPCG_OFFLOAD_VECTOR_YNCOL 6
#define HPCG_OFFLOAD_VECTOR_ZNCOL 7
#define HPCG_OFFLOAD_VECTOR_XOVERLAP 8

#define HPCG_OFFLOAD_HALO_PROFILING

#endif /* OFFLOADEXTHPCG_HPP */
