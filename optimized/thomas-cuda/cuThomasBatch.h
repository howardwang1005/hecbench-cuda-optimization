/**
 *
 *  @file cuThomasBatch.h
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

#include <cstddef>   // size_t

__global__ void cuThomasBatch(
            const double *L, const double *D, double *U, double *RHS,
            const int M,
            const int BATCHCOUNT);

// Optimized PCR solver: one block per system, M threads per block,
// requires 6*M*sizeof(double) bytes of dynamic shared memory.
// NOTE: L, D, U are all const (PCR does not modify them); this MUST match the
// definition in cuThomasBatch.cu exactly, or the mangled names differ and the
// link fails (U as non-const here vs const there was the earlier bug).
__global__ void cuThomasBatchPCR(
            const double *L, const double *D, const double *U, double *RHS,
            const int M,
            const int BATCHCOUNT);
