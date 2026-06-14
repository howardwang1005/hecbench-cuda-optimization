#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;

// Simulate the GPU PCR exactly as the kernel does, single system, M elements.
void pcr_solve(int M, vector<double> l, vector<double> d, vector<double> u,
               vector<double> r, vector<double>& x) {
  vector<double> l2(M),d2(M),u2(M),r2(M);
  for (int delta=1; delta<M; delta<<=1) {
    for (int i=0;i<M;i++){
      double dl=l[i],dd=d[i],du=u[i],dr=r[i];
      double k1=0,k2=0;
      if (i-delta>=0) k1=dl/d[i-delta];
      if (i+delta< M) k2=du/d[i+delta];
      double nd=dd,nr=dr,nl=0,nu=0;
      if (i-delta>=0){ nd-=u[i-delta]*k1; nr-=r[i-delta]*k1; nl=-l[i-delta]*k1; }
      if (i+delta< M){ nd-=l[i+delta]*k2; nr-=r[i+delta]*k2; nu=-u[i+delta]*k2; }
      l2[i]=nl;d2[i]=nd;u2[i]=nu;r2[i]=nr;
    }
    l=l2;d=d2;u=u2;r=r2;
  }
  x.resize(M);
  for(int i=0;i<M;i++) x[i]=r[i]/d[i];
}

// Reference: serial Thomas.
void thomas(int M, vector<double> l, vector<double> d, vector<double> u,
            vector<double> r, vector<double>& x){
  u[0]/=d[0]; r[0]/=d[0];
  for(int i=1;i<M;i++){
    double m=d[i]-l[i]*u[i-1];
    u[i]/=m;
    r[i]=(r[i]-l[i]*r[i-1])/m;
  }
  x.resize(M); x[M-1]=r[M-1];
  for(int i=M-2;i>=0;i--) x[i]=r[i]-u[i]*x[i+1];
}

int main(){
  srand(123);
  for (int M : {1,2,3,7,8,15,16,1000,1024}) {
    vector<double> l(M),d(M),u(M),r(M);
    for(int i=0;i<M;i++){
      l[i]= (i==0)?0 : (-2.0+4.0*rand()/RAND_MAX);
      u[i]= (i==M-1)?0 : (-2.0+4.0*rand()/RAND_MAX);
      d[i]= 5.0+5.0*rand()/RAND_MAX;
      r[i]= -2.0+4.0*rand()/RAND_MAX;
    }
    vector<double> xp, xt;
    pcr_solve(M,l,d,u,r,xp);
    thomas(M,l,d,u,r,xt);
    double err=0; for(int i=0;i<M;i++) err=max(err,fabs(xp[i]-xt[i]));
    printf("M=%-5d max|PCR-Thomas| = %.3e  %s\n", M, err, err<1e-9?"OK":"FAIL");
  }
  return 0;
}
