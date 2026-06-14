#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;
// Emulate binary_warp_scan: exclusive prefix count of (x>0) within a warp of size W.
// Reference: exclusive scan of predicate.
int main(){
  srand(123);
  for (int W : {32}) {            // single-warp case (N=32)
    for (int trial=0; trial<5; trial++){
      vector<int> x(W), ref(W);
      for(int i=0;i<W;i++) x[i]=rand()%W - W/2;
      // reference exclusive scan of (x[i]>0)
      ref[0]=0;
      for(int i=1;i<W;i++) ref[i]=ref[i-1]+(x[i-1]>0);
      // candidate: binary_warp_scan == popc(ballot(p) & lanemask_lt(lane))
      unsigned ballot=0;
      for(int i=0;i<W;i++) if(x[i]>0) ballot |= (1u<<i);
      vector<int> got(W);
      bool ok=true;
      for(int lane=0; lane<W; lane++){
        unsigned mask=(1u<<lane)-1;       // lanemask_lt
        got[lane]=__builtin_popcount(ballot & mask);
        if(got[lane]!=ref[lane]) ok=false;
      }
      printf("W=%d trial=%d %s\n", W, trial, ok?"OK":"FAIL");
    }
  }
  return 0;
}
