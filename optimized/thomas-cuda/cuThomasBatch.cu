/**
 *
 *  @file cuThomasBatch.cu
 *
 *  @brief cuThomasBatch kernel implementaion.
 *
 *  cuThomasBatch is a software package provided by
 *  Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *  @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *  @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

/**
 *
 *  @ingroup cuThomasBatch
 *  
 *  Solve a set of Tridiagonal linear systems:
 *
 *      A_ix_i = RHS_i, for all i = 0, ..., N
 *
 *      N = BATCHCOUNT
 *
 *  where A is a MxM tridiagonal matrix:
 *
 *      A_i = [ D_i[0]     U_i[1]    .        .    .          .       
 *              L_i[0]     D_i[1]    U_i[2]   .    .          .          
 *              .          L_i[1]    D_i[2]   .    .          .     
 *              .          .         L_i[2]   .    .          U_i[M-1] 
 *              .          .         .        .    L_i[M-2]   D_i[M-1] ]
 *
 *  Note that the elements of the inputs must be interleaved by following the
 *  next pattern for N (BATCHCOUNT) tridiagonal systems and M elements each:
 *
 *      D_0[0], D_1[0], ..., D_N[0], ..., D_0[M-1], D_1[M-1], ..., D_N[M-1]
 *
**/

/**
 *  
 *  @param[in]
 *  L           double *.
 *              L is a pointer to the lower-diagonal vector
 *          
 *  @param[in]
 *  D           double *.
 *              D is a pointer to the diagonal vector
 *
 *  @param[in,out]
 *  U           double *.
 *              U is a pointer to the uper-diagonal vector
 *
 *  @param[in,out]
 *  RHS         double *.    
 *              RHS is a pointer to the Right Hand Side vector
 *   
 *   
 *  @param[in]
 *  M           int.
 *              M specifies the number of elemets of the systems 
 *
 *  @param[in]
 *  BATCHCOUNT  int.
 *              BATCHCOUNT specifies to number of systems to be procesed
 **/
#include "cuThomasBatch.h"

__global__ void cuThomasBatch(const double *__restrict__ L,
                              const double *__restrict__ D,
                                    double *__restrict__ U,
                                    double *__restrict__ RHS,
                              const int M,
                              const int BATCHCOUNT)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < BATCHCOUNT) {

    int first = tid;
    int last  = BATCHCOUNT*(M-1)+tid;

    // Optimized (same algorithm, fewer ops/loads): the baseline recomputed the
    // denominator D[i]-L[i]*U[i-stride] twice and re-loaded U[i-stride] /
    // RHS[i-stride] from global memory each iteration. Carry the previous U and
    // RHS in registers and compute the denominator once. Kernel/result identical;
    // this just removes redundant FP ops and global loads in the hot sweep.
    double u_prev = U[first] / D[first];     // forward-eliminated U at first
    double r_prev = RHS[first] / D[first];   // forward-eliminated RHS at first
    U[first]   = u_prev;
    RHS[first] = r_prev;

    for (int i = first + BATCHCOUNT; i < last; i+=BATCHCOUNT) {
      double denom = D[i] - L[i] * u_prev;   // computed once
      u_prev = U[i] / denom;
      r_prev = ( RHS[i] - L[i] * r_prev ) / denom;
      U[i]   = u_prev;
      RHS[i] = r_prev;
    }

    {
      double denom = D[last] - L[last] * u_prev;
      r_prev = ( RHS[last] - L[last] * r_prev ) / denom;
      RHS[last] = r_prev;
    }

    // back-substitution: RHS[i] -= U[i]*RHS[i+stride]; carry RHS[i+stride]
    for (int i = last-BATCHCOUNT; i >= first; i-=BATCHCOUNT) {
      r_prev = RHS[i] - U[i] * r_prev;
      RHS[i] = r_prev;
    }
  }
}
