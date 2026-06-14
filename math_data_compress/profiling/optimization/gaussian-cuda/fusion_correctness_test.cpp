#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
void ref_forward(vector<float>&a, vector<float>&b, vector<float>&m, int size){
  for(int t=0;t<size-1;t++){
    for(int i=0;i<size-1-t;i++) m[size*(i+t+1)+t]=a[size*(i+t+1)+t]/a[size*t+t];
    for(int x=0;x<size-1-t;x++) for(int y=0;y<size-t;y++){
      a[size*(x+t+1)+y+t]-=m[size*(x+t+1)+t]*a[size*t+y+t];
      if(y==0) b[x+1+t]-=m[size*(x+1+t)+(y+t)]*b[t];
    }
  }
}
// Fused, race-free: NEVER write the pivot column a[row][t]; start subtraction at y=1.
// b update done by the y==0 thread using m read from original a[row][t] (untouched).
void fused_nopivotwrite(vector<float>&a, vector<float>&b, vector<float>&m, int size){
  for(int t=0;t<size-1;t++){
    // emulate full parallel: all threads read original a[row][t] (never written) -> no race
    for(int x=0;x<size-1-t;x++){
      int row=x+t+1;
      float mv = a[size*row+t]/a[size*t+t];   // original, untouched
      // y starts at 1: skip pivot column write (a[row][t] left as-is; unused downstream)
      for(int y=1;y<size-t;y++) a[size*row+(y+t)] -= mv*a[size*t+(y+t)];
      b[row]-= mv*b[t];                        // the old y==0 b-update
    }
  }
}
double maxdiff(vector<float>&x,vector<float>&y){double e=0;for(size_t i=0;i<x.size();i++)e=fmax(e,fabs(x[i]-y[i]));return e;}
// backsub reads only upper triangle, so compare finalVec, not raw a.
void backsub(vector<float>&a,vector<float>&b,vector<float>&fv,int size){
  fv.assign(size,0);
  for(int i=0;i<size;i++){fv[size-i-1]=b[size-i-1];
    for(int j=0;j<i;j++) fv[size-i-1]-=a[size*(size-i-1)+(size-j-1)]*fv[size-j-1];
    fv[size-i-1]/=a[size*(size-i-1)+(size-i-1)];}
}
int main(){
  int size=128;
  vector<float> a0(size*size),b0(size);
  for(int i=0;i<size;i++){b0[i]=1.0f;for(int j=0;j<size;j++)a0[i*size+j]=(float)((i*7+j*13+5)%17)+1.0f+(i==j?2*size:0);}
  vector<float> a,b,m,fvR,fvF;
  a=a0;b=b0;m.assign(size*size,0);ref_forward(a,b,m,size);backsub(a,b,fvR,size);
  a=a0;b=b0;m.assign(size*size,0);fused_nopivotwrite(a,b,m,size);backsub(a,b,fvF,size);
  printf("finalVec max diff (fused vs ref) = %.3e  %s\n", maxdiff(fvR,fvF), maxdiff(fvR,fvF)<1e-3?"OK":"FAIL");
  return 0;
}
