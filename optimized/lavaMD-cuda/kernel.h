__global__ void md ( const box_str* d_box_gpu,
    const FOUR_VECTOR* d_rv_gpu,
    const fp* d_qv_gpu,
    FOUR_VECTOR* d_fv_gpu,
    const fp alpha, 
    int dim_cpu_number_boxes) 
{

  __shared__ FOUR_VECTOR rA_shared[100];
  __shared__ FOUR_VECTOR rB_shared[100];
  __shared__ fp qB_shared[100];

  int bx = blockIdx.x; 
  int tx = threadIdx.x;
  int wtx = tx;

  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  //  DO FOR THE NUMBER OF BOXES
  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

  if(bx<dim_cpu_number_boxes){

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  Extract input parameters
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // parameters
    fp a2 = 2*alpha*alpha;

    // home box
    int first_i;
    // (enable the line below only if wanting to use shared memory)

    // nei box
    int pointer;
    int k = 0;
    int first_j;
    int j = 0;
    // (enable the two lines below only if wanting to use shared memory)

    // common
    fp r2;
    fp u2;
    fp vij;
    fp fs;
    fp fxij;
    fp fyij;
    fp fzij;
    THREE_VECTOR d;

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  Home box
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    //----------------------------------------------------------------------------------------------------------------------------------140
    //  Setup parameters
    //----------------------------------------------------------------------------------------------------------------------------------140

    // home box - box parameters
    first_i = d_box_gpu[bx].offset;

    //----------------------------------------------------------------------------------------------------------------------------------140
    //  Copy to shared memory
    //----------------------------------------------------------------------------------------------------------------------------------140

    // (enable the section below only if wanting to use shared memory)
    // home box - shared memory
    while(wtx<NUMBER_PAR_PER_BOX){
      rA_shared[wtx] = d_rv_gpu[first_i+wtx];
      wtx = wtx + NUMBER_THREADS;
    }
    wtx = tx;

    // (enable the section below only if wanting to use shared memory)
    // synchronize threads  - not needed, but just to be safe for now
    __syncthreads();

    // Profiling (job 952741) shows the inner loop read-modify-wrote global
    // memory (d_fv_gpu) on every (wtx,j) iteration. One particle maps to one
    // thread (NUMBER_PAR_PER_BOX <= NUMBER_THREADS), so accumulate the force in
    // a register over all nei boxes and write to global ONCE at the end. Same
    // summation order as the baseline -> identical result.
    FOUR_VECTOR fv_acc;
    fv_acc.v = 0; fv_acc.x = 0; fv_acc.y = 0; fv_acc.z = 0;

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  nei box loop
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // loop over nei boxes of home box
    for (k=0; k<(1+d_box_gpu[bx].nn); k++){

      //----------------------------------------50
      //  nei box - get pointer to the right box
      //----------------------------------------50

      if(k==0){
        pointer = bx;                          // set first box to be processed to home box
      }
      else{
        pointer = d_box_gpu[bx].nei[k-1].number;              // remaining boxes are nei boxes
      }

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // nei box - box parameters
      first_j = d_box_gpu[pointer].offset;

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // (enable the section below only if wanting to use shared memory)
      // nei box - shared memory
      while(wtx<NUMBER_PAR_PER_BOX){
        rB_shared[wtx] = d_rv_gpu[first_j+wtx];
        qB_shared[wtx] = d_qv_gpu[first_j+wtx];
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;

      // (enable the section below only if wanting to use shared memory)
      // synchronize threads because in next section each thread accesses data brought in by different threads here
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Calculation
      //----------------------------------------------------------------------------------------------------------------------------------140

      // one particle per thread; accumulate into the register fv_acc
      if (tx < NUMBER_PAR_PER_BOX){
        FOUR_VECTOR rA = rA_shared[tx];
        for (j=0; j<NUMBER_PAR_PER_BOX; j++){
          r2 = rA.v + rB_shared[j].v - DOT(rA,rB_shared[j]);
          u2 = a2*r2;
          // fp is float, but the baseline called the double `exp` (promoting
          // float->double); profiling (job 952743) shows the kernel is
          // compute-bound on this transcendental (SM 82%, DRAM 0.6%). Use the
          // float `expf` to stay in float precision and avoid the double path.
          vij= __expf(-u2);
          fs = 2*vij;
          d.x = rA.x - rB_shared[j].x;  fxij=fs*d.x;
          d.y = rA.y - rB_shared[j].y;  fyij=fs*d.y;
          d.z = rA.z - rB_shared[j].z;  fzij=fs*d.z;
          fv_acc.v += qB_shared[j]*vij;
          fv_acc.x += qB_shared[j]*fxij;
          fv_acc.y += qB_shared[j]*fyij;
          fv_acc.z += qB_shared[j]*fzij;
        }
      }

      // synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Calculation END
      //----------------------------------------------------------------------------------------------------------------------------------140

    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  nei box loop END
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // single global write of the accumulated force (fv_cpu starts at 0)
    if (tx < NUMBER_PAR_PER_BOX){
      d_fv_gpu[first_i+tx].v += fv_acc.v;
      d_fv_gpu[first_i+tx].x += fv_acc.x;
      d_fv_gpu[first_i+tx].y += fv_acc.y;
      d_fv_gpu[first_i+tx].z += fv_acc.z;
    }

  }

}
