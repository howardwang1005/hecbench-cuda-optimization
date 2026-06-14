#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;
// Model the EXACT kernel: blockDim=(X=8,Y=4,Z=8), grid z parallel rows,
// y parallel j within row, x parallel ref-row elems + shfl reduction over X lanes.
int main(){
  srand(2);
  int numRow=200,numCol=150;
  vector<vector<float>> M(numRow, vector<float>(numCol));
  for(int r=0;r<numRow;r++)for(int c=0;c<numCol;c++)M[r][c]=rand()%10;
  vector<float> val; vector<int> ptr={0}, ind; int nnz=0;
  for(int i=0;i<numRow;i++){for(int j=0;j<numCol;j++)if(M[i][j]!=0){val.push_back(M[i][j]);ind.push_back(j);nnz++;}ptr.push_back(nnz);}
  int n=numRow,e=nnz;
  auto vv=[&](int c){return (float)(c+1)/e;};
  auto rl=[&](int r){return ptr[r+1]-ptr[r];};

  vector<double> wi_ref(e,0.0), wi_new(e,0.0);
  // reference (serial two-pointer, x==0 only)
  for(int row=0;row<n;row++)for(int j=ptr[row];j<ptr[row+1];j++){
    int col=ind[j];int Ni=rl(row),Nj=rl(col);int ref=(Ni<Nj)?row:col,cur=(Ni<Nj)?col:row;
    double ls=0;int ip=ptr[ref],jp=ptr[cur];
    while(ip<ptr[ref+1]&&jp<ptr[cur+1]){int rc=ind[ip],cc=ind[jp];
      if(rc==cc){ls+=vv(rc);ip++;jp++;}else if(rc<cc)ip++;else jp++;}
    if(ls!=0)wi_ref[j]+=ls;
  }
  // new: exact lane decomposition with X=8 (binary search + reduction)
  const int X=8;
  for(int row=0;row<n;row++)for(int j=ptr[row];j<ptr[row+1];j++){
    int col=ind[j];int Ni=rl(row),Nj=rl(col);int ref=(Ni<Nj)?row:col,cur=(Ni<Nj)?col:row;
    double lane[X]={0};
    for(int lx=0;lx<X;lx++){
      for(int i=ptr[ref]+lx;i<ptr[ref+1];i+=X){
        int rc=ind[i];int lo=ptr[cur],hi=ptr[cur+1]-1,m=-1;
        while(lo<=hi){int mid=(lo+hi)>>1;int mc=ind[mid];
          if(mc>rc)hi=mid-1;else if(mc<rc)lo=mid+1;else{m=mid;break;}}
        if(m!=-1)lane[lx]+=vv(rc);
      }
    }
    // shfl-down tree reduction
    for(int off=X>>1;off>0;off>>=1)for(int lx=0;lx<X-off;lx++)lane[lx]+=lane[lx+off];
    if(lane[0]!=0)wi_new[j]+=lane[0];
  }
  double maxd=0;for(int j=0;j<e;j++)maxd=fmax(maxd,fabs(wi_ref[j]-wi_new[j]));
  printf("n=%d e=%d  max|new-ref| = %.3e  %s\n",n,e,maxd,maxd<1e-4?"OK":"FAIL");
  return 0;
}
