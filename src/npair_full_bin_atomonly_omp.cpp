// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "npair_full_bin_atomonly_omp.h"

#include "atom.h"
#include "error.h"
#include "my_page.h"
#include "neigh_list.h"
#include "npair_omp.h"

#include "omp_compat.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

NPairFullBinAtomonlyOmp::NPairFullBinAtomonlyOmp(LAMMPS *lmp) : NPair(lmp) {}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void NPairFullBinAtomonlyOmp::build(NeighList *list)
{
  const int nlocal = (includegroup) ? atom->nfirst : atom->nlocal;

  NPAIR_OMP_INIT;
#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(list)
#endif
  NPAIR_OMP_SETUP(nlocal);

  int i,j,k,n,itype,jtype,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  // loop over owned atoms, storing neighbors

  for (i = ifrom; i < ito; i++) {

    n = 0;
    neighptr = ipage.vget();

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms in surrounding bins in stencil including self
    // skip i = j

    ibin = atom2bin[i];

    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
        if (i == j) continue;

        jtype = type[j];
        if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq[itype][jtype]) neighptr[n++] = j;
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
  NPAIR_OMP_CLOSE;
  list->inum = nlocal;
  list->gnum = 0;
}

void NPairFullBinAtomonlyOmp::build_parral(NeighList *list, int tid)
{
  const int pair_len = comm->pair_len;

  const int nthreads = comm->nthreads;

  int ifrom, ito, idelta;

  int idelta_i = pair_len / nthreads;
  int idelta_j = pair_len % nthreads;
  int _bias    = idelta_j == 0 ? 0 : 1;
  
  if(tid >= idelta_j) {
    ifrom = (idelta_i + _bias) * idelta_j + idelta_i * (tid - idelta_j);
    ito = ifrom + idelta_i; 
  } else {
    ifrom = (idelta_i + _bias) * (tid);
    ito = ifrom + idelta_i + _bias; 
  }
  ito = (ito > pair_len) ? pair_len : ito;
  
  if(DEBUG_MSG) utils::logmesg(lmp, "[INFO] NPairFullBinAtomonlyOmp tid {} ifrom {} ito {}  pair_len {} nthreads {} molu {}\n", tid, ifrom, ito, pair_len, nthreads, (void*)atom->molecule); 

  int i,j,k,n,itype,jtype,ibin;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // each thread has its own page allocator
  MyPage<int> &ipage = list->ipage[tid];
  ipage.reset();

  // loop over owned atoms, storing neighbors
  for (int _i = ifrom; _i < ito; _i++) {
    i = comm->pair_index[_i];

    // if(i >= list->maxatom) error->one(FLERR, "i {} exceed list maxatom {}", i, list->maxatom);

    n = 0;
    neighptr = ipage.vget();

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms in surrounding bins in stencil including self
    // skip i = j

    ibin = nuatom2bin[i];

    for (k = 0; k < nstencil; k++) {
      // if(ibin+nustencil[k] < 0 || ibin+nustencil[k] >= numbins) {
      //     error->one(FLERR,"ibin overflow  ibin {}  nustencil {}  result {} numbins {}",
      //           ibin, nustencil[k], ibin+nustencil[k], numbins);
      // }

      for (j = nubinhead[ibin+nustencil[k]]; j >= 0; j = nubins[j]) {
        if (i == j) continue;

        jtype = type[j];
        if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq[itype][jtype]) neighptr[n++] = j;
      }
    }

    ilist[i] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    ipage.vgot(n);
    if (ipage.status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }

  // list->inum = nlocal;
  // list->gnum = 0;
}

