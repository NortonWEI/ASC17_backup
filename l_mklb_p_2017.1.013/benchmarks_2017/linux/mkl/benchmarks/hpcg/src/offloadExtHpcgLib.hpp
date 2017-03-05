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

#include "offloadExtHpcg.hpp"

#ifndef OFFLOADEXTHPCGLIB
#define OFFLOADEXTHPCGLIB

extern volatile int gbl_offload_signal;
extern volatile double gbl_offload_dbl;
extern volatile int gbl_offload_int;
extern volatile long long gbl_offload_lint;

extern volatile int gbl_offload_halo_max_neighbors;
extern volatile int gbl_offload_halo_max_mpi_buffer_size;
extern volatile int gbl_offload_halo_num_neighbors;
extern int* gbl_offload_halo_neighbors;
extern int* gbl_offload_halo_recv_sizes;
extern int* gbl_offload_halo_send_sizes;
extern double* gbl_offload_halo_send_buf;
extern volatile int gbl_offload_halo_id;
extern volatile int gbl_offload_level;

extern double *gbl_offload_x1, *gbl_offload_x2, *gbl_offload_x3;
extern double *gbl_offload_p, *gbl_offload_z;
extern double *gbl_offload_xncol, *gbl_offload_yncol, *gbl_offload_zncol;
extern double *gbl_offload_xoverlap;

extern volatile int gbl_offload_m0, gbl_offload_m1, gbl_offload_m2, gbl_offload_m3;
extern volatile int gbl_offload_n0, gbl_offload_n1, gbl_offload_n2, gbl_offload_n3;

extern double *gbl_offload_dbl_gather;

#endif
