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

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "hpcg.hpp"

typedef enum
{
  SETUP_PERSISTENT_COMM = 0,
  IMBALANCE,
  START_RECV,
  FILL_UP_SEND_BUF,
  START_SEND,
  WAIT,
  HALO_TIMES_END,
} HALO_TIMES;

void ExchangeHalo(const SparseMatrix & A, Vector & x, double *haloTimes = NULL, const HPCG_Params& params = HPCG_Params(), bool overlap = false);

void WaitHalo(const SparseMatrix& A, Vector& x, double *haloTimes);

#endif // EXCHANGEHALO_HPP
