// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_lj_cut_threadpool.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"
#include "neighbor.h"

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCutThreadpool::PairLJCutThreadpool(LAMMPS *lmp) :
  PairLJCut(lmp), ThrThreadpool(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  cut_respa = nullptr;
  pair_style_debug = 555;
}

/* ---------------------------------------------------------------------- */

void PairLJCutThreadpool::compute(int eflag, int vflag, int tid) {
  // ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;
 
  int ifrom, ito;

  ThrData *thr = fix->get_thr(tid);

  thr->timer(Timer::START);

  const int idelta = 1 + inum / nthreads;
  ifrom = tid * idelta;
  ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
  // ifrom = neighbor->pair_ifrom[tid];
  // ito = neighbor->pair_ito[tid];

  // if(comm->me == 0)utils::logmesg(PairLJCut::lmp,"Threadpool::compute tid {} ifrom {} ito {} nthreads  {} flag {} {} nlocal {}\n", 
  //         tid, ifrom, ito, nthreads, eflag, vflag, atom->nlocal);

  if(DEBUG_MSG) utils::logmesg(PairLJCut::lmp,"Threadpool::compute tid {} ifrom {} ito {} nthreads  {} flag {} {} nlocal {} faddr {}\n", 
          tid, ifrom, ito, nthreads, eflag, vflag, atom->nlocal, (void*)thr->get_f()[0]);
  
  ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

  if (evflag) {
    if (eflag) {
      if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
      else eval<1,1,0>(ifrom, ito, thr);
    } else {
      if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
      else eval<1,0,0>(ifrom, ito, thr);
    }
  } else {
    if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
    else eval<0,0,0>(ifrom, ito, thr);
  }

  thr->timer(Timer::PAIR);
  if(DEBUG_MSG) utils::logmesg(PairLJCut::lmp,"Threadpool::compute begin reduce tid {} \n", tid);

  reduce_thr(this, eflag, vflag, thr);    


  // if(comm->me == 0 && tid == 11) utils::logmesg(PairLJCut::lmp, "pairthreadpol virtual {} {} {} {} {} {}\n", 
  //               virial[0],virial[1],virial[2],
  //               virial[3],virial[4],virial[5]); 
}


void PairLJCutThreadpool::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  // utils::logmesg(PairLJCut::lmp,"PairLJCutThreadpool::compute \n");

  int ifrom, ito, tid;

  loop_setup_thr(ifrom, ito, tid, inum, nthreads);
  ThrData *thr = fix->get_thr(tid);
  thr->timer(Timer::START);
  ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

  if (evflag) {
    if (eflag) {
      if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
      else eval<1,1,0>(ifrom, ito, thr);
    } else {
      if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
      else eval<1,0,0>(ifrom, ito, thr);
    }
  } else {
    if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
    else eval<0,0,0>(ifrom, ito, thr);
  }
  thr->timer(Timer::PAIR);
  reduce_thr(this, eflag, vflag, thr);    
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJCutThreadpool::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;

  double tmp_fx, tmp_fy, tmp_fz;

  const int nlocal = atom->nlocal;
  const int nall   = nlocal + atom->nghost;
  int j,jj,jnum,jtype;

  evdwl = 0.0;

  std::string mesg;

  // loop over neighbors of my atoms

  // utils::logmesg(PairLJCut::lmp, "eval tid {} ifrom {} ito {} nall  {}\n", 
  //         thr->get_tid(), iifrom, iito, nall);

  for (int ii = iifrom; ii < iito; ++ii) {  
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];
    const double * _noalias const cutsqi = cutsq[itype];
    const double * _noalias const offseti = offset[itype];
    const double * _noalias const lj1i = lj1[itype];
    const double * _noalias const lj2i = lj2[itype];
    const double * _noalias const lj3i = lj3[itype];
    const double * _noalias const lj4i = lj4[itype];

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;


    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // if(j < 0 || j >= nall) {
      //   utils::logmesg(PairLJCut::lmp,"eval overflow tid {} jj {}\n", thr->get_tid(), j);
      // }
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqi[jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
        fpair = factor_lj*forcelj*r2inv;
        

        tmp_fx = delx*fpair;
        tmp_fy = dely*fpair;
        tmp_fz = delz*fpair;

        fxtmp += tmp_fx;
        fytmp += tmp_fy;
        fztmp += tmp_fz;

        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= tmp_fx;
          f[j].y -= tmp_fy;
          f[j].z -= tmp_fz;
        }

        if (EFLAG) {
          evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
          evdwl *= factor_lj;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairLJCutThreadpool::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairLJCut::memory_usage();

  return bytes;
}
