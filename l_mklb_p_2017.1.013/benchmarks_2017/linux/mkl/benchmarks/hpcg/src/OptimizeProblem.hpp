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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include <vector>

#include "hpcg.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

namespace SpMP
{
  class LevelSchedule;
}

struct OptimizationData
{
  void *A, *ABoundary;
  void *AReordered; // for GS smoothing w/ coloring

#if defined(__MIC__) || defined(__AVX512F__)
  void *AEllReordered, *AEllBoundary; // for SymGS with coloring

  void *AEllhReordered; // triangular matrices for SymGS with SpMV + trsolve

  void *spmvBalancer;
  void *forwardBalancers, *backwardBalancers;
#endif

  double *tempVector;
  double *diagValues;

  // for reordering
  int *perm, *inversePerm;
  std::vector<int> levvec;

  SpMP::LevelSchedule *levelSchedule;

  int level;

  int *f2cOperator, *elementsToSend;

  OptimizationData(int l);

  int *partialColorOrder, *inversePartialColorOrder;
};

void OptimizeProblemInit(SparseMatrix & A, const HPCG_Params& params);

int OptimizeProblem(
  SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact,
  const HPCG_Params& params);

void OptimizeProblemFinalize(SparseMatrix& A, const HPCG_Params& params);

double OptimizeProblemMemoryUse(const SparseMatrix & A);

#endif  // OPTIMIZEPROBLEM_HPP
