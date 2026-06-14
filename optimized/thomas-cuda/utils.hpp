/**
 *
 *   @file utils.hpp
 *
 *   @brief utils funtions
 *
 *   cuThomasBatch is a software package provided by
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *   @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *   @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

#include <cmath>

template <typename T>
void calcError(T* src,T* dst, int size){
  double error=0;
  for (int i = 0; i < size; ++i)
  {
    if (error < std::fabs(std::fabs(src[i]) - std::fabs(dst[i])))
    {
      error = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
    }
  }
  printf("Maximum error: %e\n",error);
}

template <typename T>
void calcError_print(T* src,T* dst, int size){

  double error=0;

  for (int i = 0; i < size; ++i)
  {
    printf("src: %e - dst: %e\n",src[i],dst[i]);

    if (error < std::fabs(std::fabs(src[i]) - std::fabs(dst[i])))
    {
      error = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
    }
  }
  printf("Maximum error: %e\n",error);
}
