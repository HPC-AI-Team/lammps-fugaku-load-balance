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

#include "nbin_standard.h"
#include "neighbor.h"
#include "atom.h"
#include "group.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "memory.h"
#include "utils.h"

using namespace LAMMPS_NS;

#define SMALL 1.0e-6
#define CUT2BIN_RATIO 100

/* ---------------------------------------------------------------------- */

NBinStandard::NBinStandard(LAMMPS *lmp) : NBin(lmp) {}

/* ----------------------------------------------------------------------
   setup for bin_atoms()
------------------------------------------------------------------------- */

void NBinStandard::bin_atoms_setup(int nall)
{
  // binhead = per-bin vector, mbins in length
  // add 1 bin for INTEL package

  if (mbins > maxbin) {
    maxbin = mbins;
    memory->destroy(binhead);
    memory->create(binhead,maxbin,"neigh:binhead");
  }

  // bins and atom2bin = per-atom vectors
  // for both local and ghost atoms

  if (nall > maxatom) {
    maxatom = nall;
    memory->destroy(bins);
    memory->create(bins,maxatom,"neigh:bins");
    memory->destroy(atom2bin);
    memory->create(atom2bin,maxatom,"neigh:atom2bin");
  }
}

void NBinStandard::bin_atoms_setup_numa(int nunall)
{
  // binhead = per-bin vector, mbins in length
  // add 1 bin for INTEL package

  if (numbins > numaxbin) {
    numaxbin = numbins;
    memory->destroy(nubinhead);
    memory->create(nubinhead,numaxbin,"neigh:nubinhead");
  }

  // bins and atom2bin = per-atom vectors
  // for both local and ghost atoms

  if (nunall > numaxatom) {
    numaxatom = nunall;
    memory->destroy(nubins);
    memory->create(nubins,numaxatom,"neigh:nubins");
    memory->destroy(nuatom2bin);
    memory->create(nuatom2bin,numaxatom,"neigh:nuatom2bin");
  }
}

/* ----------------------------------------------------------------------
   setup neighbor binning geometry
   bin numbering in each dimension is global:
     0 = 0.0 to binsize, 1 = binsize to 2*binsize, etc
     nbin-1,nbin,etc = bbox-binsize to bbox, bbox to bbox+binsize, etc
     -1,-2,etc = -binsize to 0.0, -2*binsize to -binsize, etc
   code will work for any binsize
     since next(xyz) and stencil extend as far as necessary
     binsize = 1/2 of cutoff is roughly optimal
   for orthogonal boxes:
     a dim must be filled exactly by integer # of bins
     in periodic, procs on both sides of PBC must see same bin boundary
     in non-periodic, coord2bin() still assumes this by use of nbin xyz
   for triclinic boxes:
     tilted simulation box cannot contain integer # of bins
     stencil & neigh list built differently to account for this
   mbinlo = lowest global bin any of my ghost atoms could fall into
   mbinhi = highest global bin any of my ghost atoms could fall into
   mbin = number of bins I need in a dimension
------------------------------------------------------------------------- */

void NBinStandard::setup_bins(int style)
{
  // bbox = size of bbox of entire domain
  // bsubbox lo/hi = bounding box of my subdomain extended by comm->cutghost
  // for triclinic:
  //   bbox bounds all 8 corners of tilted box
  //   subdomain is in lamda coords
  //   include dimension-dependent extension via comm->cutghost
  //   domain->bbox() converts lamda extent to box coords and computes bbox

  double bbox[3],bsubboxlo[3],bsubboxhi[3];
  double *cutghost = comm->cutghost;

  if (triclinic == 0) {
    bsubboxlo[0] = domain->sublo[0] - cutghost[0];
    bsubboxlo[1] = domain->sublo[1] - cutghost[1];
    bsubboxlo[2] = domain->sublo[2] - cutghost[2];
    bsubboxhi[0] = domain->subhi[0] + cutghost[0];
    bsubboxhi[1] = domain->subhi[1] + cutghost[1];
    bsubboxhi[2] = domain->subhi[2] + cutghost[2];
  } else {
    double lo[3],hi[3];
    lo[0] = domain->sublo_lamda[0] - cutghost[0];
    lo[1] = domain->sublo_lamda[1] - cutghost[1];
    lo[2] = domain->sublo_lamda[2] - cutghost[2];
    hi[0] = domain->subhi_lamda[0] + cutghost[0];
    hi[1] = domain->subhi_lamda[1] + cutghost[1];
    hi[2] = domain->subhi_lamda[2] + cutghost[2];
    domain->bbox(lo,hi,bsubboxlo,bsubboxhi);
  }

  bbox[0] = bboxhi[0] - bboxlo[0];
  bbox[1] = bboxhi[1] - bboxlo[1];
  bbox[2] = bboxhi[2] - bboxlo[2];

  // optimal bin size is roughly 1/2 the cutoff
  // for BIN style, binsize = 1/2 of max neighbor cutoff
  // for MULTI_OLD style, binsize = 1/2 of min neighbor cutoff
  // special case of all cutoffs = 0.0, binsize = box size

  double binsize_optimal;
  if (binsizeflag) binsize_optimal = binsize_user;
  else if (style == Neighbor::BIN) binsize_optimal = 0.5*cutneighmax;
  else binsize_optimal = 0.5*cutneighmin;
  if (binsize_optimal == 0.0) binsize_optimal = bbox[0];
  double binsizeinv = 1.0/binsize_optimal;

  // test for too many global bins in any dimension due to huge global domain

  if (bbox[0]*binsizeinv > MAXSMALLINT || bbox[1]*binsizeinv > MAXSMALLINT ||
      bbox[2]*binsizeinv > MAXSMALLINT)
    error->all(FLERR,"Domain too large for neighbor bins");

  // create actual bins
  // always have one bin even if cutoff > bbox
  // for 2d, nbinz = 1

  nbinx = static_cast<int> (bbox[0]*binsizeinv);
  nbiny = static_cast<int> (bbox[1]*binsizeinv);
  if (dimension == 3) nbinz = static_cast<int> (bbox[2]*binsizeinv);
  else nbinz = 1;

  if (nbinx == 0) nbinx = 1;
  if (nbiny == 0) nbiny = 1;
  if (nbinz == 0) nbinz = 1;

  // compute actual bin size for nbins to fit into box exactly
  // error if actual bin size << cutoff, since will create a zillion bins
  // this happens when nbin = 1 and box size << cutoff
  // typically due to non-periodic, flat system in a particular dim
  // in that extreme case, should use NSQ not BIN neighbor style

  binsizex = bbox[0]/nbinx;
  binsizey = bbox[1]/nbiny;
  binsizez = bbox[2]/nbinz;

  bininvx = 1.0 / binsizex;
  bininvy = 1.0 / binsizey;
  bininvz = 1.0 / binsizez;

  if (binsize_optimal*bininvx > CUT2BIN_RATIO ||
      binsize_optimal*bininvy > CUT2BIN_RATIO ||
      binsize_optimal*bininvz > CUT2BIN_RATIO)
    error->all(FLERR,"Cannot use neighbor bins - box size << cutoff");

  // mbinlo/hi = lowest and highest global bins my ghost atoms could be in
  // coord = lowest and highest values of coords for my ghost atoms
  // static_cast(-1.5) = -1, so subract additional -1
  // add in SMALL for round-off safety

  int mbinxhi,mbinyhi,mbinzhi;
  double coord;

  coord = bsubboxlo[0] - SMALL*bbox[0];
  mbinxlo = static_cast<int> ((coord-bboxlo[0])*bininvx);
  if (coord < bboxlo[0]) mbinxlo = mbinxlo - 1;
  coord = bsubboxhi[0] + SMALL*bbox[0];
  mbinxhi = static_cast<int> ((coord-bboxlo[0])*bininvx);

  coord = bsubboxlo[1] - SMALL*bbox[1];
  mbinylo = static_cast<int> ((coord-bboxlo[1])*bininvy);
  if (coord < bboxlo[1]) mbinylo = mbinylo - 1;
  coord = bsubboxhi[1] + SMALL*bbox[1];
  mbinyhi = static_cast<int> ((coord-bboxlo[1])*bininvy);

  if (dimension == 3) {
    coord = bsubboxlo[2] - SMALL*bbox[2];
    mbinzlo = static_cast<int> ((coord-bboxlo[2])*bininvz);
    if (coord < bboxlo[2]) mbinzlo = mbinzlo - 1;
    coord = bsubboxhi[2] + SMALL*bbox[2];
    mbinzhi = static_cast<int> ((coord-bboxlo[2])*bininvz);
  }

  // extend bins by 1 to insure stencil extent is included
  // for 2d, only 1 bin in z

  mbinxlo = mbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  mbinx = mbinxhi - mbinxlo + 1;

  mbinylo = mbinylo - 1;
  mbinyhi = mbinyhi + 1;
  mbiny = mbinyhi - mbinylo + 1;

  if (dimension == 3) {
    mbinzlo = mbinzlo - 1;
    mbinzhi = mbinzhi + 1;
  } else mbinzlo = mbinzhi = 0;
  mbinz = mbinzhi - mbinzlo + 1;

  bigint bbin = ((bigint) mbinx) * ((bigint) mbiny) * ((bigint) mbinz) + 1;
  if (bbin > MAXSMALLINT) error->one(FLERR,"Too many neighbor bins");
  mbins = bbin;
}

void NBinStandard::setup_bins_numa(int style) {
  double bbox[3],bsubboxlo[3],bsubboxhi[3];
  double *cutghost = comm->cutghost;

  if (triclinic == 0) {
    bsubboxlo[0] = domain->nusublo[0] - cutghost[0];
    bsubboxlo[1] = domain->nusublo[1] - cutghost[1];
    bsubboxlo[2] = domain->nusublo[2] - cutghost[2];
    bsubboxhi[0] = domain->nusubhi[0] + cutghost[0];
    bsubboxhi[1] = domain->nusubhi[1] + cutghost[1];
    bsubboxhi[2] = domain->nusubhi[2] + cutghost[2];
  } else {
    double lo[3],hi[3];
    lo[0] = domain->sublo_lamda[0] - cutghost[0];
    lo[1] = domain->sublo_lamda[1] - cutghost[1];
    lo[2] = domain->sublo_lamda[2] - cutghost[2];
    hi[0] = domain->subhi_lamda[0] + cutghost[0];
    hi[1] = domain->subhi_lamda[1] + cutghost[1];
    hi[2] = domain->subhi_lamda[2] + cutghost[2];
    domain->bbox(lo,hi,bsubboxlo,bsubboxhi);
  }

  bbox[0] = bboxhi[0] - bboxlo[0];
  bbox[1] = bboxhi[1] - bboxlo[1];
  bbox[2] = bboxhi[2] - bboxlo[2];

  // optimal bin size is roughly 1/2 the cutoff
  // for BIN style, binsize = 1/2 of max neighbor cutoff
  // for MULTI_OLD style, binsize = 1/2 of min neighbor cutoff
  // special case of all cutoffs = 0.0, binsize = box size

  double binsize_optimal;
  if (binsizeflag) binsize_optimal = binsize_user;
  else if (style == Neighbor::BIN) binsize_optimal = 0.5*cutneighmax;
  else binsize_optimal = 0.5*cutneighmin;
  if (binsize_optimal == 0.0) binsize_optimal = bbox[0];
  double binsizeinv = 1.0/binsize_optimal;

  // test for too many global bins in any dimension due to huge global domain

  if (bbox[0]*binsizeinv > MAXSMALLINT || bbox[1]*binsizeinv > MAXSMALLINT ||
      bbox[2]*binsizeinv > MAXSMALLINT)
    error->all(FLERR,"Domain too large for neighbor bins");

  // create actual bins
  // always have one bin even if cutoff > bbox
  // for 2d, nbinz = 1

  nbinx = static_cast<int> (bbox[0]*binsizeinv);
  nbiny = static_cast<int> (bbox[1]*binsizeinv);
  if (dimension == 3) nbinz = static_cast<int> (bbox[2]*binsizeinv);
  else nbinz = 1;

  if (nbinx == 0) nbinx = 1;
  if (nbiny == 0) nbiny = 1;
  if (nbinz == 0) nbinz = 1;

  // compute actual bin size for nbins to fit into box exactly
  // error if actual bin size << cutoff, since will create a zillion bins
  // this happens when nbin = 1 and box size << cutoff
  // typically due to non-periodic, flat system in a particular dim
  // in that extreme case, should use NSQ not BIN neighbor style

  binsizex = bbox[0]/nbinx;
  binsizey = bbox[1]/nbiny;
  binsizez = bbox[2]/nbinz;

  bininvx = 1.0 / binsizex;
  bininvy = 1.0 / binsizey;
  bininvz = 1.0 / binsizez;

  if (binsize_optimal*bininvx > CUT2BIN_RATIO ||
      binsize_optimal*bininvy > CUT2BIN_RATIO ||
      binsize_optimal*bininvz > CUT2BIN_RATIO)
    error->all(FLERR,"Cannot use neighbor bins - box size << cutoff");

  // mbinlo/hi = lowest and highest global bins my ghost atoms could be in
  // coord = lowest and highest values of coords for my ghost atoms
  // static_cast(-1.5) = -1, so subract additional -1
  // add in SMALL for round-off safety

  int mbinxhi,mbinyhi,mbinzhi;
  double coord;

  coord = bsubboxlo[0] - SMALL*bbox[0];
  numbinxlo = static_cast<int> ((coord-bboxlo[0])*bininvx);
  if (coord < bboxlo[0]) numbinxlo = numbinxlo - 1;
  coord = bsubboxhi[0] + SMALL*bbox[0];
  mbinxhi = static_cast<int> ((coord-bboxlo[0])*bininvx);

  coord = bsubboxlo[1] - SMALL*bbox[1];
  numbinylo = static_cast<int> ((coord-bboxlo[1])*bininvy);
  if (coord < bboxlo[1]) numbinylo = numbinylo - 1;
  coord = bsubboxhi[1] + SMALL*bbox[1];
  mbinyhi = static_cast<int> ((coord-bboxlo[1])*bininvy);

  if (dimension == 3) {
    coord = bsubboxlo[2] - SMALL*bbox[2];
    numbinzlo = static_cast<int> ((coord-bboxlo[2])*bininvz);
    if (coord < bboxlo[2]) numbinzlo = numbinzlo - 1;
    coord = bsubboxhi[2] + SMALL*bbox[2];
    mbinzhi = static_cast<int> ((coord-bboxlo[2])*bininvz);
  }

  // extend bins by 1 to insure stencil extent is included
  // for 2d, only 1 bin in z

  numbinxlo = numbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  numbinx = mbinxhi - numbinxlo + 1;

  numbinylo = numbinylo - 1;
  mbinyhi = mbinyhi + 1;
  numbiny = mbinyhi - numbinylo + 1;

  if (dimension == 3) {
    numbinzlo = numbinzlo - 1;
    mbinzhi = mbinzhi + 1;
  } else numbinzlo = mbinzhi = 0;
  numbinz = mbinzhi - numbinzlo + 1;

  bigint bbin = ((bigint) numbinx) * ((bigint) numbiny) * ((bigint) numbinz) + 1;
  if (bbin > MAXSMALLINT) error->one(FLERR,"Too many neighbor bins");
  numbins = bbin;

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] bbox {:<3.3g} {:<3.3g} {:<3.3g} \n", bbox[0], bbox[1],bbox[2]);
  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] bbox {:<3.3g}:{:<3.3g} {:<3.3g}:{:<3.3g} {:<3.3g}:{:<3.3g} \n", bboxlo[0],bboxhi[0], bboxlo[1],bboxhi[1],bboxlo[2], bboxhi[2]);
  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] nbin {} {} {}  numbins {}  numbin {} {} {} numbinlo {} {} {}\n", 
                    nbinx, nbiny, nbinz, numbins, numbinx, numbiny, numbinz, numbinxlo,numbinylo,numbinzlo );

}

/* ----------------------------------------------------------------------
   bin owned and ghost atoms
------------------------------------------------------------------------- */

void NBinStandard::bin_atoms()
{
  int i,ibin;

  last_bin = update->ntimestep;
  for (i = 0; i < mbins; i++) binhead[i] = -1;

  // bin in reverse order so linked list will be in forward order
  // also puts ghost atoms at end of list, which is necessary

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  if (includegroup) {
    int bitmask = group->bitmask[includegroup];
    for (i = nall-1; i >= nlocal; i--) {
      if (mask[i] & bitmask) {
        ibin = coord2bin(x[i]);
        atom2bin[i] = ibin;
        bins[i] = binhead[ibin];
        binhead[ibin] = i;
      }
    }
    for (i = atom->nfirst-1; i >= 0; i--) {
      ibin = coord2bin(x[i]);
      atom2bin[i] = ibin;
      bins[i] = binhead[ibin];
      binhead[ibin] = i;
    }

  } else {
    for (i = nall-1; i >= 0; i--) {
      // double bsubboxlo[3], bsubboxhi[3];

      // double *cutghost = comm->cutghost;
      // bsubboxlo[0] = domain->sublo[0] - cutghost[0];
      // bsubboxlo[1] = domain->sublo[1] - cutghost[1];
      // bsubboxlo[2] = domain->sublo[2] - cutghost[2];
      // bsubboxhi[0] = domain->subhi[0] + cutghost[0];
      // bsubboxhi[1] = domain->subhi[1] + cutghost[1];
      // bsubboxhi[2] = domain->subhi[2] + cutghost[2];
      // if (i > nlocal) {
      //   if(x[i][0] < bsubboxlo[0] || x[i][0] > bsubboxhi[0] || 
      //     x[i][1] < bsubboxlo[1] || x[i][1] > bsubboxhi[1] || 
      //     x[i][2] < bsubboxlo[2] || x[i][2] > bsubboxhi[2]){
      //       // error->one(FLERR,"[info] ghost atom error i {} x {:.2f} {:.2f} {:.2f} \n", 
      //       //       i, x[i][0], x[i][1], x[i][2]);
      //       continue;
      //     }
      // }
      ibin = coord2bin(x[i]);
      if (ibin > mbins || ibin < 0) {
        error->one(FLERR,"[info] ibin oversize mbins {} ibin {} x {:.2f} {:.2f} {:.2f} \n", 
              mbins, ibin, x[i][0], x[i][1], x[i][2]);
      }
      atom2bin[i] = ibin;
      bins[i] = binhead[ibin];
      binhead[ibin] = i;
    }
  }

  // utils::logmesg_arry(lmp, fmt::format("[NUMA] binhead "), binhead, mbins, 1); 
  // utils::logmesg_arry(lmp, fmt::format("[NUMA] bins "), bins, nall, 1); 

}

void NBinStandard::bin_atoms_numa()
{
  int i,ibin;
  double bsubboxlo[3], bsubboxhi[3];

  double *cutghost = comm->cutghost;
  bsubboxlo[0] = domain->nusublo[0] - cutghost[0];
  bsubboxlo[1] = domain->nusublo[1] - cutghost[1];
  bsubboxlo[2] = domain->nusublo[2] - cutghost[2];
  bsubboxhi[0] = domain->nusubhi[0] + cutghost[0];
  bsubboxhi[1] = domain->nusubhi[1] + cutghost[1];
  bsubboxhi[2] = domain->nusubhi[2] + cutghost[2];

  last_bin = update->ntimestep;
  for (i = 0; i < numbins; i++) nubinhead[i] = -1;

  // bin in reverse order so linked list will be in forward order
  // also puts ghost atoms at end of list, which is necessary

  double **nux = atom->x;
  int *numask = atom->mask;
  int nunlocal = atom->nunlocal;
  int nunall = nunlocal + atom->nunghost;

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] begin bin_atoms_numa nunall {} nunlocal {}  nughost {} numbins {} \n", 
                        nunall, nunlocal, atom->nunghost, numbins);
  
  for (i = nunall-1; i >= 0; i--) {
    if (i >= nunlocal) {
      if(nux[i][0] < bsubboxlo[0] || nux[i][0] > bsubboxhi[0] || 
        nux[i][1] < bsubboxlo[1] || nux[i][1] > bsubboxhi[1] || 
        nux[i][2] < bsubboxlo[2] || nux[i][2] > bsubboxhi[2]) {
          continue;
        }
    }
    // if (i < nunlocal) {
    //   if(nux[i][0] < bsubboxlo[0] || nux[i][0] > bsubboxhi[0] || 
    //     nux[i][1] < bsubboxlo[1] || nux[i][1] > bsubboxhi[1] || 
    //     nux[i][2] < bsubboxlo[2] || nux[i][2] > bsubboxhi[2]) {
          
    //     error->one(FLERR,"[NUMA] local atom flys {:<3.3g} {:<3.3g} {:<3.3g} \n", 
    //       nux[i][0], nux[i][1], nux[i][2]);
    //   }
    // }
    
    ibin = coord2bin_numa(nux[i]);
    if (ibin > numbins || ibin < 0) {
      error->one(FLERR,"[NUMA] ibin oversize numbins {} ibin {} x {:<3.3g} {:<3.3g} {:<3.3g} \n", 
            numbins, ibin, nux[i][0], nux[i][1], nux[i][2]);
    }


    nuatom2bin[i] = ibin;
    nubins[i] = nubinhead[ibin];
    nubinhead[ibin] = i;
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] nuatom2bin  "),  nuatom2bin, nunall, 1);
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] nubins      "),  nubins, nunall, 1);
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] nubinhead   "),  nubinhead, numbins, 1);

}

/* ---------------------------------------------------------------------- */

double NBinStandard::memory_usage()
{
  double bytes = 0;
  bytes += (double)maxbin*sizeof(int);
  bytes += (double)2*maxatom*sizeof(int);
  return bytes;
}
