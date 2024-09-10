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

/* ----------------------------------------------------------------------
   Contributing author (triclinic) : Pieter in 't Veld (SNL)
------------------------------------------------------------------------- */

#include "comm_brick.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "compute.h"
#include "domain.h"
#include "dump.h"
#include "error.h"
#include "fix.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "platform.h"
#include "timer.h"
#include "force.h"
#include "thr_data.h"
#include "modify.h"
#include <stdlib.h>

#include <cmath>
#include <cstring>
#include <cstdlib>

#include <chrono>
#include <thread>

using namespace LAMMPS_NS;

#define BUFFACTOR 1.5
#define BUFMIN 1024
#define BIG 1.0e20
#define SWAP(x,y)  (x = (x)+(y),y=(x)-(y), x=(x)-(y)); 
#define EPSINON_B 0.01

#ifdef OPT_NEWTON
  #define COMM_STEP 2
#else
  #define COMM_STEP 1
#endif


/* ---------------------------------------------------------------------- */

CommBrick::CommBrick(LAMMPS *lmp) :
  Comm(lmp),
  sendnum(nullptr), recvnum(nullptr), sendproc(nullptr), recvproc(nullptr),
  size_forward_recv(nullptr), size_reverse_send(nullptr), size_reverse_recv(nullptr),
  slablo(nullptr), slabhi(nullptr), multilo(nullptr), multihi(nullptr),
  multioldlo(nullptr), multioldhi(nullptr), cutghostmulti(nullptr), cutghostmultiold(nullptr),
  pbc_flag(nullptr), pbc(nullptr), firstrecv(nullptr), sendlist(nullptr),
  localsendlist(nullptr), maxsendlist(nullptr), buf_send(nullptr), buf_recv(nullptr)
{
  style = Comm::BRICK;
  layout = Comm::LAYOUT_UNIFORM;
  pbc_flag = nullptr;
  init_buffers();
  buildMPIType();  
  first_init_flag = false;
}

inline uintptr_t align_cache_size(uintptr_t ptr) {
  uintptr_t cache_size = sysconf(_SC_LEVEL1_ICACHE_LINESIZE);
  ptr = ((ptr + cache_size - 1) / cache_size) * cache_size;
  return ptr;
}

inline uintptr_t align_cache_size(uintptr_t ptr, uintptr_t add) {
  ptr += add;
  uintptr_t cache_size = sysconf(_SC_LEVEL1_ICACHE_LINESIZE);
  ptr = ((ptr + cache_size - 1) / cache_size) * cache_size;
  return ptr;
}

inline uintptr_t align_double(uintptr_t offset) {
  offset = ((offset + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  return offset;
}

inline uintptr_t get_align_border_ptr(uintptr_t n) {
  n = n * (sizeof(double) * 3 + sizeof(int) * 3);
  n = ((n + sizeof(double) - 1) / sizeof(double));
  return n;
}

inline int find_rank_id_numa(int data, int *array, int length){
  int ptr = -1;
  for(int i = 0; i < length; i++) {
    if(data == (array[i] / 4)) {
      ptr = i;
      return ptr;
    }
  }
  return ptr;
}

inline int find_rank_id(int data, int *array, int length){
  int ptr = -1;
  for(int i = 0; i < length; i++) {
    if(data == (array[i])) {
      ptr = i;
      return ptr;
    }
  }
  return ptr;
}

inline void arm_store(int ptr, int data) {
    __asm__ __volatile__(
        "STR %1, [%0]"
        : // 没有输出操作数
        : "r" (ptr), "r" (data) // 输入操作数：内存地址和要存储的值
        : "memory" // 告诉编译器内存内容被修改
    );
};

inline void arm_barrier(uint64_t &data, uint64_t goal ) {
    while(1) {
      uint64_t result;
      __asm__ __volatile__(
          "ldr x0, %1;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "mov x1, %2;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "sub %0, x0, x1;"  // 执行减法 r0 - r1，并将结果存储在 result 中
          : "=r" (result)    // 输出操作数
          : "m" (data), "r" (goal) // 输入操作数（注意使用 "m" 约束来引用内存地址）
          : "x0", "x1"      // 破坏描述
      );

      if(result) break;
      else {
          // printf("mem_barrier cq_id %d i %d iter %d recv %d\n", _cq_id, _i ,_iter, recv_buffer[_cq_id][_i][_pos]);  fflush(stdout);
          // sleep(1);
          // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }; 
    // printf("mem_barrier cq_id %d i %d iter %d recv %d\n", _cq_id, _i ,_iter, recv_buffer[_cq_id][_i][_pos]);  fflush(stdout);
  };



inline bool in_neighbor_box(double *x, double *sublo, double *subhi){
  if(x[0] >= sublo[0] && x[0] < subhi[0] &&
    x[1] >= sublo[1] && x[1] < subhi[1] &&
    x[2] >= sublo[2] && x[2] < subhi[2]) {
      return true;
    }
  else {
    return false;
  }
}

inline int atom2bin(double *x, double bin_split_line[3][2] ){
  int ibin_pos[3];

  ibin_pos[0] = (x[0] < bin_split_line[0][0]) ? 0 :
                (x[0] >= bin_split_line[0][0] && x[0] < bin_split_line[0][1]) ? 1 : 2;
  ibin_pos[1] = (x[1] < bin_split_line[1][0]) ? 0 :
                (x[1] >= bin_split_line[1][0] && x[1] < bin_split_line[1][1]) ? 1 : 2;
  ibin_pos[2] = (x[2] < bin_split_line[2][0]) ? 0 :
                (x[2] >= bin_split_line[2][0] && x[2] < bin_split_line[2][1]) ? 1 : 2;
  
  return ibin_pos[2] * 9 + ibin_pos[1] * 3 + ibin_pos[0];
}

inline bool isEqualBigger(double x, double y){
  return ((x - y) > 0) || std::fabs(x - y) < EPSINON_B; 
}

inline bool isEqualSmaller(double x, double y){
  return ((y - x) > 0) || std::fabs(y - x) < EPSINON_B; 
}

/* ---------------------------------------------------------------------- */

CommBrick::~CommBrick()
{
  CommBrick::free_swap();
  if (mode == Comm::MULTI) {
    CommBrick::free_multi();
    memory->destroy(cutghostmulti);
  }

  if (mode == Comm::MULTIOLD) {
    CommBrick::free_multiold();
    memory->destroy(cutghostmultiold);
  }

  if (sendlist) for (int i = 0; i < maxswap; i++) memory->destroy(sendlist[i]);
  if (localsendlist) memory->destroy(localsendlist);
  memory->sfree(sendlist);
  memory->destroy(maxsendlist);

  memory->destroy(buf_send);
  memory->destroy(buf_recv);


}

/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.

CommBrick::CommBrick(LAMMPS * /*lmp*/, Comm *oldcomm) : Comm(*oldcomm)
{
  if (oldcomm->layout == Comm::LAYOUT_TILED)
    error->all(FLERR,"Cannot change to comm_style brick from tiled layout");

  style = Comm::BRICK;
  layout = oldcomm->layout;
  Comm::copy_arrays(oldcomm);
  init_buffers();
}

/* ----------------------------------------------------------------------
   initialize comm buffers and other data structs local to CommBrick
------------------------------------------------------------------------- */

void CommBrick::init_buffers()
{
  multilo = multihi = nullptr;
  cutghostmulti = nullptr;

  multioldlo = multioldhi = nullptr;
  cutghostmultiold = nullptr;

  buf_send = buf_recv = nullptr;
  maxsend = maxrecv = BUFMIN;

  utofu_init_flag = false;

  CommBrick::grow_send(maxsend,2);
  memory->create(buf_recv,maxrecv,"comm:buf_recv");

  nswap = 0;
  maxswap = 6;

  c_vcq = 0;

  opt_maxforward = 0;
  CommBrick::allocate_swap(maxswap);

  sendlist = (int **) memory->smalloc(maxswap*sizeof(int *),"comm:sendlist");
  memory->create(maxsendlist,maxswap,"comm:maxsendlist");
  for (int i = 0; i < maxswap; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
  }
}

void CommBrick::init_buffers_opt() {
  std::string mesg;
  double *sublo,*subhi, cutoff;
  double length[3];

  opt_maxswap = 2 * ndims;


  memory->create(opt_pbc_flag,  opt_maxswap,"comm:opt_pbc_flag");
  memory->create(opt_pbc,       opt_maxswap,6,"comm:opt_pbc");
  memory->create(opt_pbc_flag_recv,  opt_maxswap,"comm:opt_pbc_flag_recv");
  memory->create(opt_pbc_recv,       opt_maxswap,6,"comm:opt_pbc_recv");
  // memory->create(opt_recv_pbc_flag,  opt_maxswap,   "comm:opt_recv_pbc_flag");
  // memory->create(opt_recv_pbc,       opt_maxswap,6, "comm:opt_recv_pbc");

  memory->create(opt_size_reverse_send, opt_maxswap,"comm:opt_size_reverse_send");
  memory->create(opt_size_reverse_recv, opt_maxswap,"comm:opt_size_reverse_recv");
  memory->create(opt_size_forward_recv, opt_maxswap,"comm:opt_size_forward_recv");
  memory->create(opt_reverse_send_pos, opt_maxswap,"comm:opt_reverse_send_pos");
  memory->create(opt_forward_send_pos, opt_maxswap,"comm:opt_forward_send_pos");

  memory->create(opt_firstrecv, opt_maxswap,"comm:opt_firstrecv");
  memory->create(remaind_iswap, opt_maxswap,"comm:remaind_iswap");

  memory->create(opt_sendproc, opt_maxswap,"comm:opt_sendproc");
  memory->create(opt_recvproc, opt_maxswap,"comm:opt_recvproc");
  memory->create(opt_maxsend, opt_maxswap,"comm:opt_maxsend");
  memory->create(opt_maxrecv, opt_maxswap,"comm:opt_maxrecv");
  memory->create(opt_forw_pos, opt_maxswap,"comm:opt_forw_pos");
  memory->create(opt_maxsendlist, opt_maxswap,"comm:opt_maxsendlist");

  memory->create(opt_slablo,opt_maxswap,3,"comm:opt_slablo");
  memory->create(opt_slabhi,opt_maxswap,3,"comm:opt_slabhi");

  opt_sendlist  = (int **) memory->smalloc(opt_maxswap*sizeof(int *),"comm:sendlist");

  for(int i = 0; i < VCQ_NUM; i++) {
    memory->create(opt_stadd_send_offset[i],opt_maxswap,"comm:opt_stadd_send_offset");
    memory->create(opt_stadd_recv_offset[i],opt_maxswap,"comm:opt_stadd_recv_offset");
  }

  for(int i = 0; i < VCQ_NUM; i++) {
    opt_buf_send[i] = (double **) memory->smalloc(opt_maxswap*sizeof(double *),"comm:opt_buf_send");
    opt_buf_recv[i] = (double **) memory->smalloc(opt_maxswap*sizeof(double *),"comm:opt_buf_recv");
  }

  sublo = domain->sublo;
  subhi = domain->subhi;
  cutoff = cutghost[0];

  int density = + 1 + atom->natoms / ((domain->boxhi[0] - domain->boxlo[0]) * (domain->boxhi[1] - domain->boxlo[1]) *(domain->boxhi[2] - domain->boxlo[2])) ;

  for(int i = 0; i < 3; i++){
    length[i] = subhi[i] - sublo[i];
  }

  if(ndims == 13) {
    opt_maxsendlist[0] =  opt_maxsendlist[1] = length[1] * length[0] * cutoff;
    opt_maxsendlist[2] =  opt_maxsendlist[3] = length[0] * length[2] * cutoff;
    opt_maxsendlist[4] =  opt_maxsendlist[5] = length[2] * length[1] * cutoff;

    opt_maxsendlist[6] =  opt_maxsendlist[7] = opt_maxsendlist[8] =  opt_maxsendlist[9] = 
                                                              length[0] * cutoff * cutoff * 1.5;
    opt_maxsendlist[10] =  opt_maxsendlist[11] = opt_maxsendlist[12] =  opt_maxsendlist[13] = 
                                                              length[1] * cutoff * cutoff * 1.5;
    opt_maxsendlist[14] =  opt_maxsendlist[15] = opt_maxsendlist[16] =  opt_maxsendlist[17] = 
                                                              length[2] * cutoff * cutoff * 1.5;

    for(int i = 18; i < opt_maxswap; i++){
      opt_maxsendlist[i] = length[1] * cutoff * cutoff * 1.5;
    }
  } else {
     for(int i = 0; i < opt_maxswap; i++){
      opt_maxsendlist[i] = atom->nlocal * 1.5;
    }
  }

  for(int i = 0; i < opt_maxswap; i++) opt_maxsendlist[i] *= density;

  total_buffer_size = 0;

  for(int i = 0; i < opt_maxswap; i++) {
    opt_maxsendlist[i]  *= 2;
    if(opt_maxsendlist[i] < BUFMIN) {
      opt_maxsendlist[i] = BUFMIN;
    }
    opt_maxsend[i]      = opt_maxsendlist[i] * size_border;
    opt_maxrecv[i]      = opt_maxsend[i] ;

    total_buffer_size += opt_maxsend[i];
  }
  total_buffer_size *= 2 * VCQ_NUM;

  opt_maxsend[0] *= NUMA_NUM;
  total_buffer_size = opt_maxswap * opt_maxsend[0] * 2 * VCQ_NUM;


  if(DEBUG_MSG) utils::logmesg(lmp,"[INFO] total_buffer_size {}\n", total_buffer_size);

  if(!comm->numa_flag) {
    memory->create(opt_recvnum, opt_maxswap,"comm:opt_recvnum");
    memory->create(opt_sendnum, opt_maxswap,"comm:opt_sendnum");

    all_recv_buffer = (double *) memory->smalloc(total_buffer_size*sizeof(double),"comm:all_recv_buffer");

    uintptr_t cur_offset = 0;
    for(int i = 0; i < VCQ_NUM; i++) {
      for(int j = 0; j < opt_maxswap; j++) {
        opt_buf_send[i][j] = all_recv_buffer + cur_offset; opt_stadd_send_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxsend[0];
        opt_buf_recv[i][j] = all_recv_buffer + cur_offset; opt_stadd_recv_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxsend[0];
      }
    }
  }

  for (int i = 0; i < opt_maxswap; i++) {
    memory->create(opt_sendlist[i],opt_maxsendlist[i],"comm:opt_sendlist[i]");
  }

  // for(int i = 0; i < VCQ_NUM; i++)
  //   for(int j = 0; j < opt_maxswap; j++) 
  //     opt_buf_send[i][j] = opt_buf_recv[i][j] = nullptr;

  // for(int i = 0; i < VCQ_NUM; i++){
  //   for(int j = 0; j < opt_maxswap; j++) {    
  //     memory->create(opt_buf_send[i][j],opt_maxsend[i],"comm:opt_buf_send[iswap]");
  //     memory->create(opt_buf_recv[i][j],opt_maxrecv[i],"comm:opt_buf_recv[iswap]");
  //     memset(opt_buf_recv[i][j], 0, sizeof(double) * opt_maxrecv[i]);
  //   }
  // }  
}

void CommBrick::init_buffers_numa()
{
  std::string mesg;
  double cutoff, *nusublo,*nusubhi;
  double length[3];

  numa_maxswap = 2 * nundims;

  memory->create(numa_pbc_flag,  numa_maxswap,"comm:numa_pbc_flag");
  memory->create(numa_pbc,       numa_maxswap,6,"comm:numa_pbc");
  memory->create(numa_pbc_flag_recv,  numa_maxswap,"comm:numa_pbc_flag_recv");
  memory->create(numa_pbc_recv,       numa_maxswap,6,"comm:numa_pbc_recv");

  memory->create(pair_index, atom->nmax,"comm:pair_index");

  memory->create(numa_sendproc, numa_maxswap,"comm:numa_sendproc");
  memory->create(numa_recvproc, numa_maxswap,"comm:numa_recvproc");
  memory->create(numa_maxsend, numa_maxswap,"comm:numa_maxsend");
  memory->create(numa_maxrecv, numa_maxswap,"comm:numa_maxrecv");
  memory->create(numa_sendnum, numa_maxswap,"comm:numa_sendnum");
  memory->create(numa_recvnum, numa_maxswap,"comm:numa_recvnum");
  memory->create(numa_maxsendlist, numa_maxswap,"comm:numa_maxsendlist");

  memory->create(opt_recvnum_numa, 4, opt_maxswap,"comm:opt_recvnum_numa");
  memory->create(opt_firstrecv_numa, 4, opt_maxswap,"comm:opt_firstrecv_numa");
  memory->create(opt2numa_swap, opt_maxswap,"comm:opt2numa_swap");

  memory->create(numa_slablo,numa_maxswap,3,"comm:numa_slablo");
  memory->create(numa_slabhi,numa_maxswap,3,"comm:numa_slabhi");

  memory->create(numa_firstrecv,numa_maxswap,"comm:numa_slabhi");

  nusublo = domain->nusublo;
  nusubhi = domain->nusubhi;
  cutoff = cutghost[0];

  for(int i = 0; i < 3; i++){
    length[i] = nusubhi[i] - nusublo[i];
  }

 if(ndims == 13) {
    numa_maxsendlist[0] =  numa_maxsendlist[1] = length[1] * length[0] * cutoff;
    numa_maxsendlist[2] =  numa_maxsendlist[3] = length[0] * length[2] * cutoff;
    numa_maxsendlist[4] =  numa_maxsendlist[5] = length[2] * length[1] * cutoff;

    numa_maxsendlist[6] =  numa_maxsendlist[7] = numa_maxsendlist[8] =  numa_maxsendlist[9] = 
                                                              length[0] * cutoff * cutoff * 1.5;
    numa_maxsendlist[10] =  numa_maxsendlist[11] = numa_maxsendlist[12] =  numa_maxsendlist[13] = 
                                                              length[1] * cutoff * cutoff * 1.5;
    numa_maxsendlist[14] =  numa_maxsendlist[15] = numa_maxsendlist[16] =  numa_maxsendlist[17] = 
                                                              length[2] * cutoff * cutoff * 1.5;

    for(int i = 18; i < numa_maxswap; i++){
      numa_maxsendlist[i] = length[1] * cutoff * cutoff * 1.5;
    }
  } else {
     for(int i = 0; i < numa_maxswap; i++){
      numa_maxsendlist[i] = atom->nlocal * 1.5;
    }
  }

  for(int i = 0; i < numa_maxswap; i++) {
    numa_maxsendlist[i]  *= 2;
    if(numa_maxsendlist[i] < BUFMIN) {
      numa_maxsendlist[i] = BUFMIN;
    }
    numa_maxsend[i]      = numa_maxsendlist[i] * size_border;
    numa_maxrecv[i]      = numa_maxsend[i] ;
  }

  if(me == 0){
    mesg = "[NUMA] sendlist max : ";
    for(int i = 0; i < numa_maxswap; i++) {
      mesg += fmt::format(" {} ", numa_maxsendlist[i]);
    }
    mesg += "\n";
    utils::logmesg(lmp,mesg);
    mesg = "[NUMA] numa_maxsend max : ";
    for(int i = 0; i < numa_maxswap; i++) {
      mesg += fmt::format(" {} ", numa_maxsend[i]);    
    }
    mesg += "\n";
    utils::logmesg(lmp,mesg);
  }
}

/* ---------------------------------------------------------------------- */
void CommBrick::init()
{
  Comm::init();

  int bufextra_old = bufextra;
  init_exchange();
  if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);

  // memory for multi style communication
  // allocate in setup

  if (mode == Comm::MULTI) {
    // If inconsitent # of collections, destroy any preexisting arrays (may be missized)
    if (ncollections != neighbor->ncollections) {
      ncollections = neighbor->ncollections;
      if (multilo != nullptr) {
        free_multi();
        memory->destroy(cutghostmulti);
      }
    }

    // delete any old user cutoffs if # of collections chanaged
    if (cutusermulti && ncollections != ncollections_cutoff) {
      if(me == 0) error->warning(FLERR, "cutoff/multi settings discarded, must be defined"
                                        " after customizing collections in neigh_modify");
      memory->destroy(cutusermulti);
      cutusermulti = nullptr;
    }

    if (multilo == nullptr) {
      allocate_multi(maxswap);
      memory->create(cutghostmulti,ncollections,3,"comm:cutghostmulti");
    }
  }
  if ((mode == Comm::SINGLE || mode == Comm::MULTIOLD) && multilo) {
    free_multi();
    memory->destroy(cutghostmulti);
  }

  // memory for multi/old-style communication

  if (mode == Comm::MULTIOLD && multioldlo == nullptr) {
    allocate_multiold(maxswap);
    memory->create(cutghostmultiold,atom->ntypes+1,3,"comm:cutghostmultiold");
  }
  if ((mode == Comm::SINGLE || mode == Comm::MULTI) && multioldlo) {
    free_multiold();
    memory->destroy(cutghostmultiold);
  }
}

/* ----------------------------------------------------------------------
   setup spatial-decomposition communication patterns
   function of neighbor cutoff(s) & cutghostuser & current box size
   single mode sets slab boundaries (slablo,slabhi) based on max cutoff
   multi mode sets collection-dependent slab boundaries (multilo,multihi)
   multi/old mode sets type-dependent slab boundaries (multioldlo,multioldhi)
------------------------------------------------------------------------- */
double CommBrick::box_distance(const int *dist, double *lcl_prd) {
  double delx,dely,delz;
  double lcl_xprd, lcl_yprd, lcl_zprd; 

  lcl_xprd = lcl_prd[0];
  lcl_yprd = lcl_prd[1];
  lcl_zprd = lcl_prd[2]; 

  if (dist[0] > 0) delx = (dist[0]-1)*lcl_xprd;
  else if (dist[0] == 0) delx = 0.0;
  else delx = (dist[0]+1)*lcl_xprd;

  if (dist[1] > 0) dely = (dist[1]-1)*lcl_yprd;
  else if (dist[1] == 0) dely = 0.0;
  else dely = (dist[1]+1)*lcl_yprd;

  if (dist[2] > 0) delz = (dist[2]-1)*lcl_zprd;
  else if (dist[2] == 0) delz = 0.0;
  else delz = (dist[2]+1)*lcl_zprd;

  return (delx*delx + dely*dely + delz*delz);
}

void CommBrick::setup() {
  // cutghost[] = max distance at which ghost atoms need to be acquired
  // for orthogonal:
  //   cutghost is in box coords = neigh->cutghost in all 3 dims
  // for triclinic:
  //   neigh->cutghost = distance between tilted planes in box coords
  //   cutghost is in lamda coords = distance between those planes
  // for multi:
  //   cutghostmulti = same as cutghost, only for each atom collection
  // for multi/old:
  //   cutghostmultiold = same as cutghost, only for each atom type

  int i,j,dim;
  int ntypes = atom->ntypes;
  double *prd,*sublo,*subhi;
  double cutneighmaxsq;

  double cut = get_comm_cutoff();

  ndims = 0;

  prd = domain->prd;
  sublo = domain->sublo;
  subhi = domain->subhi;
  cutghost[0] = cutghost[1] = cutghost[2] = cut;
  
  cutneighmaxsq = neighbor->cutneighmaxsq;

  init_buffers();

  int ncoords[3]; 

  if (mode == Comm::MULTI) {
    double **cutcollectionsq = neighbor->cutcollectionsq;

    // build collection array for atom exchange
    neighbor->build_collection(0);

    // If using multi/reduce, communicate particles a distance equal
    // to the max cutoff with equally sized or smaller collections
    // If not, communicate the maximum cutoff of the entire collection
    for (i = 0; i < ncollections; i++) {
      if (cutusermulti) {
        cutghostmulti[i][0] = cutusermulti[i];
        cutghostmulti[i][1] = cutusermulti[i];
        cutghostmulti[i][2] = cutusermulti[i];
      } else {
        cutghostmulti[i][0] = 0.0;
        cutghostmulti[i][1] = 0.0;
        cutghostmulti[i][2] = 0.0;
      }

      for (j = 0; j < ncollections; j++){
        if (multi_reduce && (cutcollectionsq[j][j] > cutcollectionsq[i][i])) continue;
        cutghostmulti[i][0] = MAX(cutghostmulti[i][0],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][1] = MAX(cutghostmulti[i][1],sqrt(cutcollectionsq[i][j]));
        cutghostmulti[i][2] = MAX(cutghostmulti[i][2],sqrt(cutcollectionsq[i][j]));
      }
    }
  }

  if (mode == Comm::MULTIOLD) {
    double *cuttype = neighbor->cuttype;
    for (i = 1; i <= ntypes; i++) {
      double tmp = 0.0;
      if (cutusermultiold) tmp = cutusermultiold[i];
      cutghostmultiold[i][0] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][1] = MAX(tmp,cuttype[i]);
      cutghostmultiold[i][2] = MAX(tmp,cuttype[i]);
    }
  }

  if (triclinic == 0) {
    prd = domain->prd;
    sublo = domain->sublo;
    subhi = domain->subhi;
    cutghost[0] = cutghost[1] = cutghost[2] = cut;
  } else {
    prd = domain->prd_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
    double *h_inv = domain->h_inv;
    double length0,length1,length2;
    length0 = sqrt(h_inv[0]*h_inv[0] + h_inv[5]*h_inv[5] + h_inv[4]*h_inv[4]);
    cutghost[0] = cut * length0;
    length1 = sqrt(h_inv[1]*h_inv[1] + h_inv[3]*h_inv[3]);
    cutghost[1] = cut * length1;
    length2 = h_inv[2];
    cutghost[2] = cut * length2;
    if (mode == Comm::MULTI) {
      for (i = 0; i < ncollections; i++) {
        cutghostmulti[i][0] *= length0;
        cutghostmulti[i][1] *= length1;
        cutghostmulti[i][2] *= length2;
      }
    }

    if (mode == Comm::MULTIOLD) {
      for (i = 1; i <= ntypes; i++) {
        cutghostmultiold[i][0] *= length0;
        cutghostmultiold[i][1] *= length1;
        cutghostmultiold[i][2] *= length2;
      }
    }
  }

  // recvneed[idim][0/1] = # of procs away I recv atoms from, within cutghost
  //   0 = from left, 1 = from right
  //   do not cross non-periodic boundaries, need[2] = 0 for 2d
  // sendneed[idim][0/1] = # of procs away I send atoms to
  //   0 = to left, 1 = to right
  //   set equal to recvneed[idim][1/0] of neighbor proc
  // maxneed[idim] = max procs away any proc recvs atoms in either direction
  // layout = UNIFORM = uniform sized sub-domains:
  //   maxneed is directly computable from sub-domain size
  //     limit to procgrid-1 for non-PBC
  //   recvneed = maxneed except for procs near non-PBC
  //   sendneed = recvneed of neighbor on each side
  // layout = NONUNIFORM = non-uniform sized sub-domains:
  //   compute recvneed via updown() which accounts for non-PBC
  //   sendneed = recvneed of neighbor on each side
  //   maxneed via Allreduce() of recvneed

  int *periodicity = domain->periodicity;
  int left,right;

  if (layout == Comm::LAYOUT_UNIFORM) {
    maxneed[0] = static_cast<int> (cutghost[0] * procgrid[0] / prd[0]) + 1;
    maxneed[1] = static_cast<int> (cutghost[1] * procgrid[1] / prd[1]) + 1;
    maxneed[2] = static_cast<int> (cutghost[2] * procgrid[2] / prd[2]) + 1;
    if (domain->dimension == 2) maxneed[2] = 0;
    if (!periodicity[0]) maxneed[0] = MIN(maxneed[0],procgrid[0]-1);
    if (!periodicity[1]) maxneed[1] = MIN(maxneed[1],procgrid[1]-1);
    if (!periodicity[2]) maxneed[2] = MIN(maxneed[2],procgrid[2]-1);

    if (!periodicity[0]) {
      recvneed[0][0] = MIN(maxneed[0],myloc[0]);
      recvneed[0][1] = MIN(maxneed[0],procgrid[0]-myloc[0]-1);
      left = myloc[0] - 1;
      if (left < 0) left = procgrid[0] - 1;
      sendneed[0][0] = MIN(maxneed[0],procgrid[0]-left-1);
      right = myloc[0] + 1;
      if (right == procgrid[0]) right = 0;
      sendneed[0][1] = MIN(maxneed[0],right);
    } else recvneed[0][0] = recvneed[0][1] =
             sendneed[0][0] = sendneed[0][1] = maxneed[0];

    if (!periodicity[1]) {
      recvneed[1][0] = MIN(maxneed[1],myloc[1]);
      recvneed[1][1] = MIN(maxneed[1],procgrid[1]-myloc[1]-1);
      left = myloc[1] - 1;
      if (left < 0) left = procgrid[1] - 1;
      sendneed[1][0] = MIN(maxneed[1],procgrid[1]-left-1);
      right = myloc[1] + 1;
      if (right == procgrid[1]) right = 0;
      sendneed[1][1] = MIN(maxneed[1],right);
    } else recvneed[1][0] = recvneed[1][1] =
             sendneed[1][0] = sendneed[1][1] = maxneed[1];

    if (!periodicity[2]) {
      recvneed[2][0] = MIN(maxneed[2],myloc[2]);
      recvneed[2][1] = MIN(maxneed[2],procgrid[2]-myloc[2]-1);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = MIN(maxneed[2],procgrid[2]-left-1);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = MIN(maxneed[2],right);
    } else recvneed[2][0] = recvneed[2][1] =
             sendneed[2][0] = sendneed[2][1] = maxneed[2];

  } else {
    recvneed[0][0] = updown(0,0,myloc[0],prd[0],periodicity[0],xsplit);
    recvneed[0][1] = updown(0,1,myloc[0],prd[0],periodicity[0],xsplit);
    left = myloc[0] - 1;
    if (left < 0) left = procgrid[0] - 1;
    sendneed[0][0] = updown(0,1,left,prd[0],periodicity[0],xsplit);
    right = myloc[0] + 1;
    if (right == procgrid[0]) right = 0;
    sendneed[0][1] = updown(0,0,right,prd[0],periodicity[0],xsplit);

    recvneed[1][0] = updown(1,0,myloc[1],prd[1],periodicity[1],ysplit);
    recvneed[1][1] = updown(1,1,myloc[1],prd[1],periodicity[1],ysplit);
    left = myloc[1] - 1;
    if (left < 0) left = procgrid[1] - 1;
    sendneed[1][0] = updown(1,1,left,prd[1],periodicity[1],ysplit);
    right = myloc[1] + 1;
    if (right == procgrid[1]) right = 0;
    sendneed[1][1] = updown(1,0,right,prd[1],periodicity[1],ysplit);

    if (domain->dimension == 3) {
      recvneed[2][0] = updown(2,0,myloc[2],prd[2],periodicity[2],zsplit);
      recvneed[2][1] = updown(2,1,myloc[2],prd[2],periodicity[2],zsplit);
      left = myloc[2] - 1;
      if (left < 0) left = procgrid[2] - 1;
      sendneed[2][0] = updown(2,1,left,prd[2],periodicity[2],zsplit);
      right = myloc[2] + 1;
      if (right == procgrid[2]) right = 0;
      sendneed[2][1] = updown(2,0,right,prd[2],periodicity[2],zsplit);
    } else recvneed[2][0] = recvneed[2][1] =
             sendneed[2][0] = sendneed[2][1] = 0;

    int all[6];
    MPI_Allreduce(&recvneed[0][0],all,6,MPI_INT,MPI_MAX,world);
    maxneed[0] = MAX(all[0],all[1]);
    maxneed[1] = MAX(all[2],all[3]);
    maxneed[2] = MAX(all[4],all[5]);
  }

  // allocate comm memory

  nswap = 2 * (maxneed[0]+maxneed[1]+maxneed[2]);
  if (nswap > maxswap) grow_swap(nswap);

  // setup parameters for each exchange:
  // sendproc = proc to send to at each swap
  // recvproc = proc to recv from at each swap
  // for mode SINGLE:
  //   slablo/slabhi = boundaries for slab of atoms to send at each swap
  //   use -BIG/midpt/BIG to insure all atoms included even if round-off occurs
  //   if round-off, atoms recvd across PBC can be < or > than subbox boundary
  //   note that borders() only loops over subset of atoms during each swap
  //   treat all as PBC here, non-PBC is handled in borders() via r/s need[][]
  // for mode MULTI:
  //   multilo/multihi is same, with slablo/slabhi for each atom type
  // pbc_flag: 0 = nothing across a boundary, 1 = something across a boundary
  // pbc = -1/0/1 for PBC factor in each of 3/6 orthogonal/triclinic dirs
  // for triclinic, slablo/hi and pbc_border will be used in lamda (0-1) coords
  // 1st part of if statement is sending to the west/south/down
  // 2nd part of if statement is sending to the east/north/up

  int ineed, iswap;

  iswap = 0;
 
  for (dim = 0; dim < 3; dim++) {
    for (ineed = 0; ineed < 2*maxneed[dim]; ineed++) {
      pbc_flag[iswap] = 0;
      pbc[iswap][0] = pbc[iswap][1] = pbc[iswap][2] =
        pbc[iswap][3] = pbc[iswap][4] = pbc[iswap][5] = 0;

      if (ineed % 2 == 0) {
        sendproc[iswap] = procneigh[dim][0];
        recvproc[iswap] = procneigh[dim][1];
        if (mode == Comm::SINGLE) {
          if (ineed < 2) slablo[iswap] = -BIG;
          else slablo[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
          slabhi[iswap] = sublo[dim] + cutghost[dim];
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            if (ineed < 2) multilo[iswap][i] = -BIG;
            else multilo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multihi[iswap][i] = sublo[dim] + cutghostmulti[i][dim];
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            if (ineed < 2) multioldlo[iswap][i] = -BIG;
            else multioldlo[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
            multioldhi[iswap][i] = sublo[dim] + cutghostmultiold[i][dim];
          }
        }
        if (myloc[dim] == 0) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = 1;
          if (triclinic) {
            if (dim == 1) pbc[iswap][5] = 1;
            else if (dim == 2) pbc[iswap][4] = pbc[iswap][3] = 1;
          }
        }
      } else {
        sendproc[iswap] = procneigh[dim][1];
        recvproc[iswap] = procneigh[dim][0];
        if (mode == Comm::SINGLE) {
          slablo[iswap] = subhi[dim] - cutghost[dim];
          if (ineed < 2) slabhi[iswap] = BIG;
          else slabhi[iswap] = 0.5 * (sublo[dim] + subhi[dim]);
        } else if (mode == Comm::MULTI) {
          for (i = 0; i < ncollections; i++) {
            multilo[iswap][i] = subhi[dim] - cutghostmulti[i][dim];
            if (ineed < 2) multihi[iswap][i] = BIG;
            else multihi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        } else {
          for (i = 1; i <= ntypes; i++) {
            multioldlo[iswap][i] = subhi[dim] - cutghostmultiold[i][dim];
            if (ineed < 2) multioldhi[iswap][i] = BIG;
            else multioldhi[iswap][i] = 0.5 * (sublo[dim] + subhi[dim]);
          }
        }
        if (myloc[dim] == procgrid[dim]-1) {
          pbc_flag[iswap] = 1;
          pbc[iswap][dim] = -1;
          if (triclinic) {
            if (dim == 1) pbc[iswap][5] = -1;
            else if (dim == 2) pbc[iswap][4] = pbc[iswap][3] = -1;
          }
        }
      }

      iswap++;
    }
  }

  setup_opt();

  if(comm->numa_flag)
    setup_numa();

}

void CommBrick::setup_opt() {
  // cutghost[] = max distance at which ghost atoms need to be acquired
  // for orthogonal:
  //   cutghost is in box coords = neigh->cutghost in all 3 dims
  // for triclinic:
  //   neigh->cutghost = distance between tilted planes in box coords
  //   cutghost is in lamda coords = distance between those planes
  // for multi:
  //   cutghostmulti = same as cutghost, only for each atom collection
  // for multi/old:
  //   cutghostmultiold = same as cutghost, only for each atom type

  int i,j,dim;
  int ntypes = atom->ntypes;
  double *prd,*sublo,*subhi, *nusublo,*nusubhi;
  double cutneighmaxsq;

  double cut = get_comm_cutoff();

  bit_nid = 1 << numa_id;

  ndims = 0;

  prd = domain->prd;
  sublo = domain->sublo;
  subhi = domain->subhi;
  cutghost[0] = cutghost[1] = cutghost[2] = cut;
 
  cutneighmaxsq = neighbor->cutneighmaxsq;

  memory->create(neigh_sublo,nprocs,3,"comm:neigh_sublo");
  memory->create(neigh_subhi,nprocs,3,"comm:neigh_subhi");

  MPI_Allgather(domain->sublo,3,MPI_DOUBLE,neigh_sublo[0],3,MPI_DOUBLE,world);
  MPI_Allgather(domain->subhi,3,MPI_DOUBLE,neigh_subhi[0],3,MPI_DOUBLE,world);

  for(i = 0; i < DIM_NUM; i++){
    if(box_distance(con_direction[i], domain->lcl_prd) < cutneighmaxsq) {
      directions[ndims][0] = con_direction[i][0];
      directions[ndims][1] = con_direction[i][1];
      directions[ndims][2] = con_direction[i][2];
      ndims++;
    }
  }

  memory->create(opt_procneigh,   ndims,  2,  "comm:opt_procneigh");

  init_buffers_opt();

  if(me == 0) utils::logmesg(lmp,"[info] setup ndims   {} opt_maxswap   {}\n", ndims, opt_maxswap);

  int ncoords[3]; 
  for(dim = 0; dim < ndims; dim++) {
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 3; j++) {
        if(i%2 == 0)
          ncoords[j] = myloc[j] - directions[dim][j];
        else
          ncoords[j] = myloc[j] + directions[dim][j];
        
        if(ncoords[j] < 0) ncoords[j] += procgrid[j];
        if(ncoords[j] >= procgrid[j]) ncoords[j] -= procgrid[j];
      }
      opt_procneigh[dim][i] = grid2proc[ncoords[0]][ncoords[1]][ncoords[2]];
    }
  }
  if(DEBUG_MSG){
    auto mesg = fmt::format("[info] opt_procneigh :"); 
    for(dim = 0; dim < ndims; dim++) {
      mesg += fmt::format("{} {}, ", opt_procneigh[dim][0],opt_procneigh[dim][1]); 
    }
    utils::logmesg(lmp,"{}\n", mesg);
  }
    
  if ((cut == 0.0) && (me == 0))
    error->warning(FLERR,"Communication cutoff is 0.0. No ghost atoms "
                   "will be generated. Atoms may get lost.");

  int *periodicity = domain->periodicity;
  int left,right;

  int ineed, iswap;

  iswap = 0;

  for(dim = 0; dim < ndims; dim++) {
    for(ineed = 0; ineed < 2; ineed++) {
      opt_pbc_flag[iswap] = 0;
      opt_pbc[iswap][0] = opt_pbc[iswap][1] = opt_pbc[iswap][2] =
        opt_pbc[iswap][3] = opt_pbc[iswap][4] = opt_pbc[iswap][5] = 0;

      if(ineed %2 == 0) {
        opt_sendproc[iswap] = opt_procneigh[dim][0];
        opt_recvproc[iswap] = opt_procneigh[dim][1];
        swap_direct[iswap][0] = -directions[dim][0]; 
        swap_direct[iswap][1] = -directions[dim][1]; 
        swap_direct[iswap][2] = -directions[dim][2]; 
      } else {
        opt_sendproc[iswap] = opt_procneigh[dim][1];
        opt_recvproc[iswap] = opt_procneigh[dim][0];
        swap_direct[iswap][0] = directions[dim][0]; 
        swap_direct[iswap][1] = directions[dim][1]; 
        swap_direct[iswap][2] = directions[dim][2]; 
      }
      iswap++;
    }
  }
  
  for(i = 0; i < 3; i++) {
    for(iswap = 0; iswap < opt_maxswap; iswap++) {
      if(myloc[i] + swap_direct[iswap][i] < 0) {
        opt_pbc_flag[iswap] = 1;
        opt_pbc[iswap][i]   = 1;
        opt_slablo[iswap][i] = neigh_sublo[opt_sendproc[iswap]][i] - cut - domain->prd[i];
        opt_slabhi[iswap][i] = neigh_subhi[opt_sendproc[iswap]][i] + cut - domain->prd[i];
      }
      else if(myloc[i] + swap_direct[iswap][i] >= procgrid[i]) {
        opt_pbc_flag[iswap] = 1;
        opt_pbc[iswap][i]   = -1;
        opt_slablo[iswap][i] = neigh_sublo[opt_sendproc[iswap]][i] - cut + domain->prd[i];
        opt_slabhi[iswap][i] = neigh_subhi[opt_sendproc[iswap]][i] + cut + domain->prd[i];
      } else {
        opt_slablo[iswap][i] = neigh_sublo[opt_sendproc[iswap]][i] - cut;
        opt_slabhi[iswap][i] = neigh_subhi[opt_sendproc[iswap]][i] + cut;
      }
    }
  }

  for(int iswap = 0; iswap < opt_maxswap; iswap++) {
    int rswap = iswap % 2 == 0 ? iswap + 1 : iswap - 1;
    opt_pbc_flag_recv[iswap] = opt_pbc_flag[rswap];
    opt_pbc_recv[iswap][0] = -1 * opt_pbc[rswap][0];
    opt_pbc_recv[iswap][1] = -1 * opt_pbc[rswap][1];
    opt_pbc_recv[iswap][2] = -1 * opt_pbc[rswap][2];
  }

  for(int i = 0; i < 3; i++){
    double tmp_cut = cut;
    if(tmp_cut > domain->lcl_prd[i]) tmp_cut -= domain->lcl_prd[i];

    bin_split_line[i][0] = sublo[i] + tmp_cut;
    bin_split_line[i][1] = subhi[i] - tmp_cut;
    if(bin_split_line[i][0] > bin_split_line[i][1]) {
      std::swap(bin_split_line[i][0], bin_split_line[i][1]);
    }

    // if(DEBUG_MSG){
    //   utils::logmesg(lmp,"bin_split_line {:.3f}:{:.3f} subbox {:.3f}:{:.3f}  cut {} {} prd {}\n", 
    //       bin_split_line[i][0], bin_split_line[i][1], sublo[i], subhi[i], tmp_cut, cut, domain->lcl_prd[i]);
    // }
  }

  double binsublo[27][3], binsubhi[27][3];

  for(int ibin = 0; ibin < 27; ibin++) {
    int ss[3];
    ss[2] = ibin / 9;
    ss[1] = (ibin / 3) % 3;
    ss[0] = ibin % 3;
    for(int i = 0; i < 3; i++) {
      if(ss[i] == 0) {
        binsublo[ibin][i] = sublo[i];
        binsubhi[ibin][i] = bin_split_line[i][0];
      } else if(ss[i] == 1){
        binsublo[ibin][i] = bin_split_line[i][0];
        binsubhi[ibin][i] = bin_split_line[i][1];
      } else if(ss[i] == 2) {
        binsublo[ibin][i] = bin_split_line[i][1];
        binsubhi[ibin][i] = subhi[i];
      }
    }
  }

  comm_step = full_flag ? 1 : 2;

  if(me == 0) utils::logmesg(lmp,"[info] comm_step  {}  full_flag {} \n", comm_step, full_flag);

  for(int tid = 0; tid < COMM_TNUM; tid++) {
    for(int i = tid * 2 ; i < opt_maxswap; i += TNI_NUM * 2) {
      for(int iswap = i; iswap <= i+1; iswap += comm_step) {
        opt_swap[tid].push_back(iswap);
      }
    }
  }

  if(DEBUG_MSG){
    for(int tid = 0; tid < COMM_TNUM; tid++) {
      utils::logmesg(lmp,"[INFO] opt_swap tid {} : ", tid);
      for(const auto iswap:opt_swap[tid]) {
        utils::logmesg(lmp," {} ", iswap);
      }
      utils::logmesg(lmp," \n");
    }
  }
  
  for(int ibin = 0; ibin < 27; ibin++) {
    bin2swap_ptr[ibin] = 0;
    for(int iswap = 0; iswap < opt_maxswap; iswap+=comm_step) {
      if(isEqualSmaller(opt_slablo[iswap][0], binsublo[ibin][0]) && isEqualBigger(opt_slabhi[iswap][0], binsubhi[ibin][0]) &&
          isEqualSmaller(opt_slablo[iswap][1], binsublo[ibin][1]) && isEqualBigger(opt_slabhi[iswap][1], binsubhi[ibin][1]) &&
          isEqualSmaller(opt_slablo[iswap][2], binsublo[ibin][2]) && isEqualBigger(opt_slabhi[iswap][2], binsubhi[ibin][2]) ) {
        bin2swap[ibin][bin2swap_ptr[ibin]++] = iswap;
      }
    }
  }


  if(DEBUG_MSG){
    for(int ibin = 0; ibin < 27; ibin++) {
      utils::logmesg(lmp," ibin {} coord {} {} {}, has {} swap", ibin, ibin % 3, (ibin / 3) % 3, ibin /9, bin2swap_ptr[ibin]);
      for(int i = 0; i < bin2swap_ptr[ibin]; i++) {
        utils::logmesg(lmp," {} ", bin2swap[ibin][i]);
      }
      utils::logmesg(lmp," \n");
    }
  }

  iswap = 0;

  if(DEBUG_MSG) {
    utils::logmesg(lmp," my_loc {} {} {} \n", myloc[0],myloc[1],myloc[2]);
    for(i = 0; i < opt_maxswap; i++) {
      utils::logmesg(lmp,"[info] iswap {} sendproc {} recvproc {}  sendborder {:.3f}:{:.3f}  {:.3f}:{:.3f}  {:.3f}:{:.3f},      sendborder_sla {:.3f}:{:.3f}  {:.3f}:{:.3f} {:.3f}:{:.3f}\n", 
                    i, opt_sendproc[i], opt_recvproc[i],
                    neigh_sublo[opt_sendproc[i]][0], neigh_subhi[opt_sendproc[i]][0],
                    neigh_sublo[opt_sendproc[i]][1], neigh_subhi[opt_sendproc[i]][1],
                    neigh_sublo[opt_sendproc[i]][2], neigh_subhi[opt_sendproc[i]][2],
                    opt_slablo[i][0], opt_slabhi[i][0],
                    opt_slablo[i][1], opt_slabhi[i][1],
                    opt_slablo[i][2], opt_slabhi[i][2]                    
                    );
    }
  }

  first_init_flag = true;
}

void CommBrick::setup_numa() {
  int i,j,dim;
  int ntypes = atom->ntypes;
  double *prd, *nusublo,*nusubhi;
  double cutneighmaxsq;
  int ***nugrid2proc;

  double cut = get_comm_cutoff();

  nundims = 0;

  prd = domain->prd;
  cutghost[0] = cutghost[1] = cutghost[2] = cut;

  FJMPI_Topology_get_shape(&nuprocgrid[0], &nuprocgrid[1], &nuprocgrid[2]);

  if(DEBUG_MSG ) utils::logmesg(lmp,"[NUMA] nuprocgrid    {} {} {} \n", nuprocgrid[0], nuprocgrid[1], nuprocgrid[2]);
  if(DEBUG_MSG ) utils::logmesg(lmp,"[NUMA] procgrid      {} {} {} \n", procgrid[0], procgrid[1], procgrid[2]);

  int **gridi;
  memory->create(gridi,nprocs,3, "gridi");
  memory->create(nugrid2proc, procgrid[0], procgrid[1],procgrid[2], "nugrid2proc");

  FJMPI_Topology_get_coords(world, me, FJMPI_LOGICAL, 3, numyloc);

  MPI_Allgather(numyloc,3,MPI_INT,gridi[0],3,MPI_INT,world);

  for (int i = 0; i < nprocs; i++)
    nugrid2proc[gridi[i][0]][gridi[i][1]][gridi[i][2]] = i / NUMA_NUM;

  memory->destroy(gridi);
  
  nusublo = domain->nusublo;
  nusubhi = domain->nusubhi;
  
  cutneighmaxsq = neighbor->cutneighmaxsq;

  memory->create(neigh_nusublo,nprocs,3,"comm:nuneigh_sublo");
  memory->create(neigh_nusubhi,nprocs,3,"comm:nuneigh_subhi");

  MPI_Allgather(domain->nusublo,3,MPI_DOUBLE,neigh_nusublo[0],3,MPI_DOUBLE,world);
  MPI_Allgather(domain->nusubhi,3,MPI_DOUBLE,neigh_nusubhi[0],3,MPI_DOUBLE,world);

  // if(DEBUG_MSG){
  //   auto mesg = fmt::format("[NUMA] neigh_nusub \n"); 
  //   for(i = 0; i < nprocs; i++) {
  //     mesg += fmt::format(" rankid {}  {:.3f}:{:.3f}, {:.3f}:{:.3f}, {:.3f}:{:.3f} \n", i, 
  //                 neigh_nusublo[i][0],neigh_nusubhi[i][0], neigh_nusublo[i][1],neigh_nusubhi[i][1],neigh_nusublo[i][2],neigh_nusubhi[i][2]); 
  //   }
  //   utils::logmesg(lmp,"{} \n", mesg);
  // }

  for(i = 0; i < DIM_NUM; i++){
    if(box_distance(con_direction[i], domain->lcl_nuprd) < cutneighmaxsq) {
      directions[nundims][0] = con_direction[i][0];
      directions[nundims][1] = con_direction[i][1];
      directions[nundims][2] = con_direction[i][2];
      nundims++;
    }
  }

  memory->create(numa_procneigh, nundims, 2,  "comm:numa_procneigh");

  init_buffers_numa();

  if(me == 0) utils::logmesg(lmp,"[NUMA] setup nundims {} nuopt_maxswap {}\n", nundims, numa_maxswap);

  int ncoords[3]; 

  for(dim = 0; dim < nundims; dim++) {
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 3; j++) {
        if(i%2 == 0)
          ncoords[j] = numyloc[j] - directions[dim][j];
        else
          ncoords[j] = numyloc[j] + directions[dim][j];
        
        if(ncoords[j] < 0) ncoords[j] += nuprocgrid[j];
        if(ncoords[j] >= nuprocgrid[j]) ncoords[j] -= nuprocgrid[j];
      }
      numa_procneigh[dim][i] = nugrid2proc[ncoords[0]][ncoords[1]][ncoords[2]];
      numa_procneigh[dim][i] = numa_procneigh[dim][i] * 4 + numa_id;
    }
  }

  // if(DEBUG_MSG) {
  //   auto mesg = fmt::format("[NUMA] numa_procneigh nundims {}:\n", nundims); 
  //   for(dim = 0; dim < nundims; dim++) {
  //     mesg += fmt::format("     dim {} proceneighbor {} {} \n", dim, numa_procneigh[dim][0],numa_procneigh[dim][1]); 
  //   }
  //   utils::logmesg(lmp,"{} \n", mesg);
  // }

  int ineed, iswap;

  iswap = 0;

  for(dim = 0; dim < nundims; dim++) {
    for(ineed = 0; ineed < 2; ineed++) {
      numa_pbc_flag[iswap] = 0;
      numa_pbc[iswap][0] = numa_pbc[iswap][1] = numa_pbc[iswap][2] =
        numa_pbc[iswap][3] = numa_pbc[iswap][4] = numa_pbc[iswap][5] = 0;
      if(ineed %2 == 0){
        numa_sendproc[iswap] = numa_procneigh[dim][0];
        numa_recvproc[iswap] = numa_procneigh[dim][1];
        swap_nudirect[iswap][0] = -directions[dim][0]; 
        swap_nudirect[iswap][1] = -directions[dim][1];
        swap_nudirect[iswap][2] = -directions[dim][2];
      } else{
        numa_sendproc[iswap] = numa_procneigh[dim][1];
        numa_recvproc[iswap] = numa_procneigh[dim][0];
        swap_nudirect[iswap][0] = directions[dim][0]; 
        swap_nudirect[iswap][1] = directions[dim][1]; 
        swap_nudirect[iswap][2] = directions[dim][2]; 
      }
      iswap++;
    }
  }

  
  // if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish create numa_sendproc numa_recvproc\n");

  for(i = 0; i < 3; i++) {
    for(iswap = 0; iswap < numa_maxswap; iswap++) {
      if(numyloc[i] + swap_nudirect[iswap][i] < 0) {
        numa_pbc_flag[iswap] = 1;
        numa_pbc[iswap][i]   = 1;
        numa_slablo[iswap][i] = neigh_nusublo[numa_sendproc[iswap]][i] - cut - domain->prd[i];
        numa_slabhi[iswap][i] = neigh_nusubhi[numa_sendproc[iswap]][i] + cut - domain->prd[i];
      }
      else if(numyloc[i] + swap_nudirect[iswap][i] >= nuprocgrid[i]) {
        numa_pbc_flag[iswap] = 1;
        numa_pbc[iswap][i]   = -1;
        numa_slablo[iswap][i] = neigh_nusublo[numa_sendproc[iswap]][i] - cut + domain->prd[i];
        numa_slabhi[iswap][i] = neigh_nusubhi[numa_sendproc[iswap]][i] + cut + domain->prd[i];
      } else {
        numa_slablo[iswap][i] = neigh_nusublo[numa_sendproc[iswap]][i] - cut;
        numa_slabhi[iswap][i] = neigh_nusubhi[numa_sendproc[iswap]][i] + cut;
      }
    }
  }

  for(int iswap = 0; iswap < numa_maxswap; iswap++) {
    int rswap = iswap % 2 == 0 ? iswap + 1 : iswap - 1;
    numa_pbc_flag_recv[iswap] = numa_pbc_flag[rswap];
    numa_pbc_recv[iswap][0] = -1 * numa_pbc[rswap][0];
    numa_pbc_recv[iswap][1] = -1 * numa_pbc[rswap][1];
    numa_pbc_recv[iswap][2] = -1 * numa_pbc[rswap][2];
  }

  if(DEBUG_MSG) {
    auto mesg = fmt::format("[NUMA] numa_procneigh numa_maxswap {} me {} :\n", numa_maxswap, me); 
    for(iswap = 0; iswap < numa_maxswap; iswap++) {
      mesg += fmt::format("[NUMA] iswap {} sendproc {} recvproc {} pbc {} {} {} {} \n ", iswap, numa_sendproc[iswap],numa_recvproc[iswap],
      numa_pbc_flag[iswap], numa_pbc[iswap][0],numa_pbc[iswap][1],numa_pbc[iswap][2]); 
    }
    utils::logmesg(lmp,"{} \n", mesg);
  }

  // for(int i = 0; i < 3; i++){
  //   double tmp_cut = cut;
  //   if(tmp_cut > domain->lcl_nuprd[i]) tmp_cut -= domain->lcl_nuprd[i];

  //   nubin_split_line[i][0] = nusublo[i] + tmp_cut;
  //   nubin_split_line[i][1] = nusubhi[i] - tmp_cut;
  //   if(nubin_split_line[i][0] > nubin_split_line[i][1]) {
  //     std::swap(nubin_split_line[i][0], nubin_split_line[i][1]);
  //   }

    // if(DEBUG_MSG){
    //   utils::logmesg(lmp,"nubin_split_line {:.3f}:{:.3f} subbox {:.3f}:{:.3f}  cut {} {} prd {}\n", 
    //       nubin_split_line[i][0], nubin_split_line[i][1], nusublo[i], nusubhi[i], tmp_cut, cut, domain->lcl_nuprd[i]);
    // }
  // }
  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish numa splitline\n");

  // double nubinsublo[27][3], nubinsubhi[27][3];

  // for(int ibin = 0; ibin < 27; ibin++) {
  //   int ss[3];
  //   ss[2] = ibin / 9;
  //   ss[1] = (ibin / 3) % 3;
  //   ss[0] = ibin % 3;
  //   for(int i = 0; i < 3; i++) {
  //     if(ss[i] == 0) {
  //       nubinsublo[ibin][i] = nusublo[i];
  //       nubinsubhi[ibin][i] = nubin_split_line[i][0];
  //     } else if(ss[i] == 1){
  //       nubinsublo[ibin][i] = nubin_split_line[i][0];
  //       nubinsubhi[ibin][i] = nubin_split_line[i][1];
  //     } else if(ss[i] == 2) {
  //       nubinsublo[ibin][i] = nubin_split_line[i][1];
  //       nubinsubhi[ibin][i] = nusubhi[i];
  //     }
  //   }
  // }

  // for(int ibin = 0; ibin < 27; ibin++) {
  //   nubin2swap_ptr[ibin] = 0;
  //   for(int iswap = 0; iswap < numa_maxswap; iswap+=COMM_STEP) {
  //     if(isEqualSmaller(numa_slablo[iswap][0], nubinsublo[ibin][0]) && isEqualBigger(numa_slabhi[iswap][0], nubinsubhi[ibin][0]) &&
  //         isEqualSmaller(numa_slablo[iswap][1], nubinsublo[ibin][1]) && isEqualBigger(numa_slabhi[iswap][1], nubinsubhi[ibin][1]) &&
  //         isEqualSmaller(numa_slablo[iswap][2], nubinsublo[ibin][2]) && isEqualBigger(numa_slabhi[iswap][2], nubinsubhi[ibin][2]) ) {
  //       nubin2swap[ibin][nubin2swap_ptr[ibin]++] = iswap;
  //     }
  //   }
  // }

  if(DEBUG_MSG) utils::logmesg(lmp," [NUMA] finish create numa ibin\n");

  thr_maxswap = (numa_maxswap / NUMA_NUM) ;
  if((numa_maxswap) % NUMA_NUM > numa_id) thr_maxswap++; 

  thr_maxswap = MIN(COMM_TNUM, thr_maxswap);
    
  int nid = 0;
  for(int i = 0, nid = 0; i < numa_maxswap; i+=2) {
    numa_swap_all[nid].push_back(i);
    numa_swap_all[nid].push_back(i + 1);

    numa_swap_full_all[nid].push_back(i);
    numa_swap_full_all[nid].push_back(i+1);
    nid = (nid + 1) % NUMA_NUM;
  }

  numa_swap = numa_swap_all[numa_id];
  numa_swap_full = numa_swap_full_all[numa_id];

  int tid = 0;
  for(i = 0; i < numa_swap.size(); i++) {
    thr_swap[tid].push_back(numa_swap[i]);
    if(i % 2 == 1) {
      tid = (tid + 1) % thr_maxswap;
    }
  }

  tid = 0;
  for(i = 0; i < numa_swap_full.size(); i+=2) {
    thr_swap_full[tid].push_back(numa_swap_full[i]);
    thr_swap_full[tid].push_back(numa_swap_full[i+1]);
    tid = (tid + 1) % thr_maxswap;
  }
  // int tid = 0;
  // for(i = 0; i < numa_swap.size(); i++) {
  //   thr_swap[tid].push_back(numa_swap[i]);
  //   // if(i % 2 == 1) {
  //     tid = (tid + 1) % thr_maxswap;
  //   // }
  // }

  // tid = 0;
  // for(i = 0; i < numa_swap_full.size(); i+=1) {
  //   thr_swap_full[tid].push_back(numa_swap_full[i]);
  //   // thr_swap_full[tid].push_back(numa_swap_full[i+1]);
  //   tid = (tid + 1) % thr_maxswap;
  // }

  for(int iswap = 0; iswap < numa_maxswap; iswap++) {
    for(int nu = 0; nu < NUMA_NUM; nu++) {
      for(auto _i : numa_swap_full_all[nu]) {
        if(iswap == _i) swap2numa_swap[iswap] = nu;
      }
    }
  }

  if(DEBUG_MSG) {
    for(int nid = 0; nid < NUMA_NUM; nid++)
      utils::logmesg_arry(lmp, fmt::format("[NUMA] numa_swap_all: numa_swap.size {} thr_maxswap {} ", numa_swap.size(), thr_maxswap), numa_swap_all[nid].data(), numa_swap_all[nid].size(), 1); 
    for(int tid = 0; tid < thr_maxswap; tid++) 
      utils::logmesg_arry(lmp, fmt::format("[NUMA] thr_swap tid {} ", tid), thr_swap[tid].data(), thr_swap[tid].size(), 1); 
    for(int nid = 0; nid < NUMA_NUM; nid++) 
      utils::logmesg_arry(lmp, fmt::format("[NUMA] numa_swap_all:  ", numa_swap_full.size()), numa_swap_full_all[nid].data(), numa_swap_full_all[nid].size(), 1); 
    for(int tid = 0; tid < thr_maxswap; tid++) 
      utils::logmesg_arry(lmp, fmt::format("[NUMA] thr_swap_full tid {} ", tid), thr_swap_full[tid].data(), thr_swap_full[tid].size(), 1); 

    utils::logmesg_arry(lmp, fmt::format("[NUMA] swap2numa_swap "), swap2numa_swap,numa_maxswap, 1); 
  }
 
  if(DEBUG_MSG) utils::logmesg(lmp," [NUMA] finish setup numa\n");



  first_init_flag = true;
  
}

void CommBrick::setup_init_shdmem_region() {

  // memory->create(opt_recvnum, opt_maxswap,"comm:opt_recvnum");

  // all_recv_buffer = (double *) memory->smalloc(total_buffer_size*sizeof(double),"comm:all_recv_buffer");

  // uintptr_t cur_offset = 0;
  // for(int i = 0; i < VCQ_NUM; i++) {      
  //   for(int j = 0; j < opt_maxswap; j++) {
  //     opt_buf_send[i][j] = all_recv_buffer + cur_offset; opt_stadd_send_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxsend[j];
  //     opt_buf_recv[i][j] = all_recv_buffer + cur_offset; opt_stadd_recv_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxrecv[j];
  //   }
  // }

  // for(int ii = 0; ii < opt_maxswap; ii++) {
  //   shm_len += sizeof(double) * opt_maxsend[ii];
  // }
  // shm_len *= VCQ_NUM * 2;
  int pagesize = sysconf(_SC_PAGESIZE);

  shm_len = 0;
  shm_len += total_buffer_size * sizeof(double) * 2 + pagesize;
  shm_len += sizeof(std::atomic<uint64_t>) * 4 * T_THREAD + 
                        sizeof(std::bitset<BIT_SET_DEEPTH>) * 2 +
                        sizeof(uint64_t) * atom->nmax * 4 * 2 + 
                        sizeof(double) * atom->nmax * 3 * (2 + nthreads) * 2 +
                        sizeof(double) * SHARE_DATA_LENGTH + 
                        sizeof(uintptr_t) * 2048;

  
  init_shdmem(numa_id, keys[numa_id], shmids[numa_id], shm_data[numa_id], shm_len, VFILE_0, numa_id+1);

  if(DEBUG_MSG || me == 0) utils::logmesg(lmp,"[NUMA] begin init_shdmem numa_id {} addr {} to {} buffersize {} nmax {} cacheline_size {}\n",
              numa_id, (void *)shm_data[numa_id], (void *)((uintptr_t)shm_data[numa_id] + shm_len), shm_len, atom->nmax, sysconf(_SC_LEVEL1_ICACHE_LINESIZE));

  uintptr_t shm_bias = (uintptr_t)shm_data[numa_id];

  for(int i = 0; i < T_THREAD; i++) { a_written[i]    = (std::atomic<int>*)shm_bias;       shm_bias = align_cache_size(shm_bias, sizeof(std::atomic<int>));}
  for(int i = 0; i < T_THREAD; i++) { a_written_s0[i] = (std::atomic<uint64_t>*)shm_bias;       shm_bias = align_cache_size(shm_bias, sizeof(std::atomic<uint64_t>));}
  for(int i = 0; i < T_THREAD; i++) { a_written_tt[i] = (std::atomic<int>*)shm_bias;       shm_bias = align_cache_size(shm_bias, sizeof(std::atomic<int>));}
  for(int i = 0; i < RPROC;    i++) { a_written_reverse[i] = (std::atomic<int>*)shm_bias;  shm_bias = align_cache_size(shm_bias, sizeof(std::atomic<int>));}

  // atom_bit_share    = (void*)shm_bias;   shm_bias = align_cache_size(shm_bias, sizeof(std::bitset<BIT_SET_DEEPTH>));
  // atom_bit          = (void*)shm_bias;   shm_bias = align_cache_size(shm_bias, sizeof(std::bitset<BIT_SET_DEEPTH>));

  // std::bitset<BIT_SET_DEEPTH> _bit_set0, _bit_set1;
  // memcpy(atom_bit_share, &_bit_set0, sizeof(std::bitset<BIT_SET_DEEPTH>));
  // memcpy(atom_bit      , &_bit_set1, sizeof(std::bitset<BIT_SET_DEEPTH>));

  // *(std::bitset<BIT_SET_DEEPTH>*)atom_bit_share = 0;
  // *(std::bitset<BIT_SET_DEEPTH>*)atom_bit       = 0;

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish atomic assignment {}\n", (void*)shm_bias);

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format("[NUMA] before atom->tag "), atom->tag, atom->nlocal, 1); 
  //   utils::logmesg_arry(lmp, fmt::format("[NUMA] before atom->type "), atom->type, atom->nlocal, 1); 
  //   utils::logmesg_arry(lmp, fmt::format("[NUMA] before atom->mask "), atom->mask, atom->nlocal, 1); 
  //   utils::logmesg_arry(lmp, fmt::format("[NUMA] before atom->image "), atom->image, atom->nlocal, 1); 
  // }

  memcpy((void*)shm_bias, atom->tag,   sizeof(tagint) *  atom->nmax);   memory->destroy(atom->tag);
  atom->tag    = (tagint*)shm_bias;    shm_bias = align_cache_size(shm_bias, sizeof(tagint) * atom->nmax);
  memcpy((void*)shm_bias, atom->type,  sizeof(int) *    atom->nmax);    memory->destroy(atom->type);
  atom->type   = (int*)shm_bias;       shm_bias = align_cache_size(shm_bias, sizeof(int) * atom->nmax);
  memcpy((void*)shm_bias, atom->mask,  sizeof(int) *    atom->nmax);    memory->destroy(atom->mask);
  atom->mask   = (int*)shm_bias;       shm_bias = align_cache_size(shm_bias,  sizeof(int) *    atom->nmax);
  // memcpy((void*)shm_bias, atom->image, sizeof(imageint) * atom->nmax);   memory->destroy(atom->image);
  // atom->image  = (imageint*)shm_bias;  shm_bias = align_cache_size(shm_bias, sizeof(imageint) * atom->nmax); 

  // memcpy((void*)shm_bias, atom->v[0],   sizeof(double) *  atom->nmax * 3);   memory->destroy(atom->v);
  // atom->v = new double*[atom->nmax];
  // atom->v[0]  = (double*)shm_bias;    shm_bias = align_cache_size(shm_bias, sizeof(double) * atom->nmax * 3);

  memcpy((void*)shm_bias, atom->x[0],   sizeof(double) *  atom->nmax * 3);   memory->destroy(atom->x);
  atom->x = new double*[atom->nmax];
  atom->x[0]  = (double*)shm_bias;    shm_bias = align_cache_size(shm_bias, sizeof(double) * atom->nmax * 3);

  // memcpy((void*)shm_bias, atom->f[0],   sizeof(double) *  atom->nmax * 3 * nthreads);  memory->destroy(atom->f);
  // atom->f = new double*[atom->nmax * nthreads];
  // atom->f[0]  = (double*)shm_bias;    shm_bias = align_cache_size(shm_bias, sizeof(double) * atom->nmax * 3 * nthreads);

  // for(int i = 1; i < atom->nmax; i++) {
  //   atom->v[i] = atom->v[i-1] + 3;
  // }
  for(int i = 1; i < atom->nmax; i++) {
    atom->x[i] = atom->x[i-1] + 3;
  }

  atom->avec->tag   =  atom->tag;
  atom->avec->type  =  atom->type;
  atom->avec->mask  =  atom->mask;
  // atom->avec->image =  atom->image;
  atom->avec->x =  atom->x;
  // atom->avec->v =  atom->v;

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] sh_init x address {}\n", (void*)atom->x[0]);

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish atom v x f assignment {}\n", (uintptr_t)shm_bias);

  shm_bias = ((shm_bias + pagesize - 1) / pagesize) * pagesize;

  all_recv_buffer = (double *) shm_bias;  

  // all_recv_buffer = (double *) memory->smalloc(total_buffer_size*sizeof(double),"comm:all_recv_buffer");

  uintptr_t cur_offset = 0;
  for(int i = 0; i < VCQ_NUM; i++) {
    for(int j = 0; j < opt_maxswap; j++) {
      opt_buf_send[i][j] = all_recv_buffer + cur_offset; opt_stadd_send_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxsend[0];
      opt_buf_recv[i][j] = all_recv_buffer + cur_offset; opt_stadd_recv_offset[i][j] = cur_offset * sizeof(double); cur_offset += opt_maxsend[0];
    }
  }
  shm_bias = align_cache_size(shm_bias, cur_offset * sizeof(double));
  if(DEBUG_MSG || me == 0) utils::logmesg(lmp,"[NUMA] total_buffer_size {} cur_offset {} delta {} \n", total_buffer_size, cur_offset, total_buffer_size-cur_offset);
  total_buffer_size = cur_offset;


  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] opt_maxsend[0] {} \n", opt_maxsend[0]);

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[NUMA] opt_maxsend "), opt_maxsend, opt_maxswap, 1); 
  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[NUMA] opt_maxrecv "), opt_maxrecv, opt_maxswap, 1); 

  for(int i = 0; i < NUMA_NUM; i++) {
    opt_force_recv[i] = new double*[numa_maxswap+1];
    for(int j = 0; j < numa_maxswap+1; j++) {
      opt_force_recv[i][j] = (double *) shm_bias; shm_bias = align_cache_size(shm_bias, opt_maxsend[0] * sizeof(double));
      // if(DEBUG_MSG)  utils::logmesg(lmp,"[NUMA] opt_force_recv {} {} {} {} \n", i, j, opt_maxsend[0] * sizeof(double), (void*)opt_force_recv[i][j]);
    }
  }

  // memory->create(opt_recvnum, opt_maxswap,"comm:opt_recvnum");
  // memory->create(opt_sendnum, opt_maxswap,"comm:opt_sendnum");
 
  opt_sendnum = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, sizeof(uint64_t) * opt_maxswap);
  opt_recvnum = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, sizeof(uint64_t) * opt_maxswap);
  shm_sendnum = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, sizeof(uint64_t) * (opt_maxswap + 1));
  shm_nlocal  = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, sizeof(uint64_t) * 4);
  shm_recvnum_numa    = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, NUMA_NUM * sizeof(uint64_t) * opt_maxswap * NUMA_NUM + 1);
  shm_firstrecv_numa  = (uint64_t*)shm_bias; shm_bias = align_cache_size(shm_bias, sizeof(uint64_t) * opt_maxswap * NUMA_NUM + 1);

  for(int i = 0; i < NUMA_NUM; i++) {
    shm_numa_recvnum_numa[i] = (uint64_t*)shm_bias;  shm_bias += sizeof(uint64_t) * numa_maxswap;
  }
  shm_bias = align_cache_size(shm_bias, 0);

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] shm_nlocal addr assignment addr {} offset {}\n", (void*)shm_nlocal, (uintptr_t)shm_nlocal-(uintptr_t)shm_data[numa_id]);

  normal_data  = (double*)shm_bias;

  intptr_t err =  (intptr_t)shm_data[numa_id] + (intptr_t)shm_len - (intptr_t)normal_data;
  if(err < 0) error->one(FLERR, "shm data not enough {} normal_data addr {} end addr {}", err, (uintptr_t)normal_data, (uintptr_t)shm_data[numa_id] + shm_len);

  if(DEBUG_MSG || me == 0) utils::logmesg(lmp,"[NUMA] finish normal_data assignment addr {} remind {}\n",
                          (void*)normal_data, err);

  std::atomic<uint64_t> tmp_a_written;
  std::atomic<int> tmp_a_written_32;
  for(int i = 0; i < T_THREAD; i++) {
    memcpy(a_written_s0[i], &tmp_a_written, sizeof(std::atomic<uint64_t>));
    *a_written_s0[i] = 0;
    memcpy(a_written[i], &tmp_a_written_32, sizeof(std::atomic<int>));
    *a_written[i] = 0;
    memcpy(a_written_tt[i], &tmp_a_written_32, sizeof(std::atomic<int>));
    *a_written_tt[i] = 0;
  }

  for(int i = 0; i < RPROC; i++) {
    memcpy(a_written_reverse[i], &tmp_a_written_32, sizeof(std::atomic<int>));
    *a_written_reverse[i] = 0;
  }
  // // int shm_ptr = 0;
  // // std::string vpfile = "/vol0001/hp230257/lijianxiong/numa_aware_lammps/lammps/src/BIN_threadpool_lj/shm_map/test.";


  if(me == 0) utils::logmesg(lmp,"[NUMA] finish init_shdmem \n");

  MPI_Barrier(nuworld);
}

void CommBrick::numa_shm_init() {

  setup_init_shdmem_region();
  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish setup_init_shdmem_region\n");

  for(int nu = 0; nu < NUMA_NUM; nu++) {
    if(numa_id != nu)
      init_shdmem(nu, keys[nu], shmids[nu], shm_data[nu], shm_len, VFILE_0, nu+1);

    for(int i = 0; i < T_THREAD; i++) { shm_st[nu].a_written_s0[i]  = (std::atomic<uint64_t>*)((uintptr_t)(a_written_s0[i]) - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]); }
    for(int i = 0; i < T_THREAD; i++) { shm_st[nu].a_written[i]     = (std::atomic<int>*)((uintptr_t)(a_written[i])    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]); }
    for(int i = 0; i < T_THREAD; i++) { shm_st[nu].a_written_tt[i]  = (std::atomic<int>*)((uintptr_t)(a_written_tt[i]) - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]); }
    for(int i = 0; i < RPROC;    i++) { shm_st[nu].a_written_reverse[i]  = (std::atomic<int>*)((uintptr_t)(a_written_reverse[i]) - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]); }

    // shm_st[nu].atom_bit_share    = (void*)((uintptr_t)atom_bit_share   - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    // shm_st[nu].atom_bit          = (void*)((uintptr_t)atom_bit         - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);

    shm_st[nu].tag      = (tagint*)((uintptr_t)atom->tag      - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].type     = (int*)((uintptr_t)atom->type        - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].mask     = (int*)((uintptr_t)atom->mask        - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    // shm_st[nu].image    = (imageint*)((uintptr_t)atom->image  - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    
    // shm_st[nu].v    = new double*[atom->nmax];
    // shm_st[nu].v[0] = (double*)((uintptr_t)atom->v[0]    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].x    = new double*[atom->nmax];
    shm_st[nu].x[0] = (double*)((uintptr_t)atom->x[0]    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    // shm_st[nu].f    = new double*[atom->nmax * nthreads];
    // shm_st[nu].f[0] = (double*)((uintptr_t)atom->f[0]    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);

    // for(int i = 1; i < atom->nmax; i++) {
    //   shm_st[nu].v[i] = shm_st[nu].v[i-1] + 3;
    // }
    for(int i = 1; i < atom->nmax; i++) {
      shm_st[nu].x[i] = shm_st[nu].x[i-1] + 3;
    }
    // for(int i = 1; i < atom->nmax * nthreads; i++) {
    //   shm_st[nu].f[i] = shm_st[nu].f[i-1] + 3;
    // }

    for(int viq = 0; viq < VCQ_NUM; viq++) {
      shm_st[nu].opt_buf_send[viq] = new double*[opt_maxswap];
      shm_st[nu].opt_buf_recv[viq] = new double*[opt_maxswap];
      for(int iswap = 0; iswap < opt_maxswap; iswap++) {
        shm_st[nu].opt_buf_send[viq][iswap] = (double*)((uintptr_t)opt_buf_send[viq][iswap] - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
        shm_st[nu].opt_buf_recv[viq][iswap] = (double*)((uintptr_t)opt_buf_recv[viq][iswap] - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
      }
    }

    for(int viq = 0; viq < NUMA_NUM; viq++) {
      shm_st[nu].opt_force_recv[viq] = new double*[numa_maxswap+1];
      for(int iswap = 0; iswap < numa_maxswap+1; iswap++) {
        shm_st[nu].opt_force_recv[viq][iswap] = (double*)((uintptr_t)opt_force_recv[viq][iswap] - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
      }
    }

    for(int viq = 0; viq < NUMA_NUM; viq++) {
      shm_st[nu].shm_numa_recvnum_numa[viq] = (uint64_t*)((uintptr_t)shm_numa_recvnum_numa[viq] - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    }

    shm_st[nu].normal_data     = (double*)((uintptr_t)normal_data     - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].opt_recvnum    = (uint64_t*)((uintptr_t)opt_recvnum    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].opt_sendnum    = (uint64_t*)((uintptr_t)opt_sendnum    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].shm_sendnum    = (uint64_t*)((uintptr_t)shm_sendnum    - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].shm_nlocal     = (uint64_t*)((uintptr_t)shm_nlocal     - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].shm_recvnum_numa       = (uint64_t*)((uintptr_t)shm_recvnum_numa     - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);
    shm_st[nu].shm_firstrecv_numa     = (uint64_t*)((uintptr_t)shm_firstrecv_numa     - (uintptr_t)shm_data[numa_id] + (uintptr_t)shm_data[nu]);

    int shm_ptr = 0;

    if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish setup_init_shdmem_region nu {}\n", nu);

  }

  a_written_id = 0;  
  for(int i = 0; i < NUMA_NUM; i++) a_written_nu_id[i] = a_written_tt_id[i] = 0;
}

void CommBrick::init_shdmem(int &nu, key_t &key, int &shmid, void *&share_mem, int memlen, std::string vpfile, int __proj_id) {
  int ret, pol;
  int pagesize = sysconf(_SC_PAGESIZE);
  struct bitmask *nmask = numa_allocate_nodemask();

  memlen = (memlen + pagesize - 1) / pagesize;
  memlen *= pagesize;

  if((key = ftok(vpfile.c_str(), __proj_id)) < 0) {
    error->one(FLERR, "ftok() get key failure: {} {} \n", key, strerror(errno));
    return ; 
  }

  if(numa_id == nu)
    shmid = shmget(key, memlen, IPC_CREAT | 0666);	//创建共享内存
  
  // MPI_Barrier(nuworld);

  if(numa_id != nu)  
    shmid = shmget(key, memlen,  0666);	//创建共享内存

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] get key_t {} and shmid  {} memlen {} \n", key, shmid, memlen);  

  if(shmid < 0) {
    error->one(FLERR, "shmget() creat shared memroy failure: {} \n", strerror(errno));
    return ; 
  }

  share_mem = shmat(shmid, NULL, 0);	//将共享内存连接到当前进程的地址空间
  if((void *) -1 == share_mem) {
    error->one(FLERR, "shmat() alloc share_mem memroy failure: {} \n", strerror(errno));
    return ;
  }

  if(numa_id == nu) {
    numa_bitmask_clearall(nmask);
    numa_bitmask_setbit(nmask, nu+4);
    ret = mbind(share_mem, memlen, MPOL_BIND, nmask->maskp, nmask->size, MPOL_MF_STRICT);
    if (ret < 0) {
      error->one(FLERR, "mbind failure: {} \n", strerror(errno));
    }

    if (get_mempolicy(&pol, nmask->maskp, nmask->size, share_mem, MPOL_F_ADDR) < 0) {
      printf("get_mempolicy pol %d mask %lx failure:%s\n",pol, *nmask->maskp & 0x3ff,strerror(errno)); fflush(stdout);
      error->one(FLERR, "get_mempolicy pol {} mask {} failure:{} \n", pol, *nmask->maskp & 0x3ff,strerror(errno));
    }
    memset(share_mem, 0, memlen);
  }

  if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] init_shdmem me {} memlen {} page size {} share_mem {} \n", me, memlen, sysconf(_SC_PAGESIZE), share_mem);   

  // if(DEBUG_MSG) utils::logmesg(lmp,"[NUMA] finish init_shdmem \n");   
}

void CommBrick::delete_shdmem(int shmid, void * share_mem) {
  if(shmdt(share_mem) == -1) {		//把共享内存从当前进程中分离
    error->one(FLERR, "shmdt() failure: {} \n", strerror(errno));
    return ;
  }
  shmctl(shmid, IPC_RMID, NULL);		//删除共享内存
  return ;
}

void CommBrick::buildMPIType() {
    int block_lengths[4];
    MPI_Aint displacements[4];
    MPI_Aint addresses[4], add_start;
    MPI_Datatype typelist[4];
    
    Utofu_comm utofu_comm;

    typelist[0] = MPI_UINT64_T;
    block_lengths[0] = 1;
    MPI_Get_address(&utofu_comm.vcq_id, &addresses[0]);

    typelist[1] = MPI_UINT64_T;
    block_lengths[1] = 1;
    MPI_Get_address(&utofu_comm.utofu_stadd, &addresses[1]);

    typelist[2] = MPI_UINT64_T;
    block_lengths[2] = 1;
    MPI_Get_address(&utofu_comm.utofu_f_stadd, &addresses[2]);

    typelist[3] = MPI_UINT64_T;
    block_lengths[3] = 1;
    MPI_Get_address(&utofu_comm.utofu_x_stadd, &addresses[3]);

    MPI_Get_address(&utofu_comm, &add_start);
    for (int i = 0; i < 4; i++) displacements[i] = addresses[i] - add_start;

    MPI_Type_create_struct(4, block_lengths , displacements, typelist, &utofu_comm_type);
    MPI_Type_commit(&utofu_comm_type);
}

void CommBrick::warp_utofu_put(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t rmt_vcq_id,
 utofu_stadd_t lcl_send_stadd, utofu_stadd_t rmt_recv_stadd, size_t length,
 uint64_t edata, uintptr_t cbvalue, unsigned long int post_flags, void *piggydata)
{
  int rc;
  // instruct the TNI to perform a Put communication
  if(comm_piggy_flag) {
    if(length <= 8) {
      rc = utofu_put_piggyback8(vcq_hdl, rmt_vcq_id, *(uint64_t*)piggydata, rmt_recv_stadd, length,
                                  edata, post_flags, (void *)cbvalue);
    } else if(length <= 32){
      rc = utofu_put_piggyback(vcq_hdl, rmt_vcq_id, piggydata, rmt_recv_stadd, length,
                                    edata, post_flags, (void *)cbvalue);    
    } else{
      rc = utofu_put(vcq_hdl, rmt_vcq_id, lcl_send_stadd, rmt_recv_stadd, length,
                                    edata, post_flags, (void *)cbvalue);    
    }
  } else {
    rc = utofu_put(vcq_hdl, rmt_vcq_id, lcl_send_stadd, rmt_recv_stadd, length,
                                  edata, post_flags, (void *)cbvalue);   
  }

  // int rc = utofu_put(vcq_hdl, rmt_vcq_id, lcl_send_stadd, rmt_recv_stadd, length,
  //                                 edata, post_flags, (void *)cbvalue);   
  if(rc != UTOFU_SUCCESS){
    error->one(FLERR, "warp_utofu_put fail\n");
  } 
}


// send data and confirm its completion
void CommBrick::warp_utofu_poll_tcq(utofu_vcq_hdl_t vcq_hdl, 
              uintptr_t &cbvalue, unsigned long int post_flags)
{
  int rc;

  // confirm the TCQ notification
  if (post_flags & UTOFU_ONESIDED_FLAG_TCQ_NOTICE) {
    void *cbdata;
    do {
      rc = utofu_poll_tcq(vcq_hdl, 0, &cbdata);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    if(rc != UTOFU_SUCCESS){
      error->one(FLERR, "warp_utofu_poll_tcq fail {}\n", rc);
    }
    // assert((uintptr_t)cbdata == cbvalue);
    cbvalue = (uintptr_t)cbdata;
  }
}


// send data and confirm its completion
void CommBrick::warp_utofu_poll_mrq(utofu_vcq_hdl_t vcq_hdl, 
      uint64_t &edata, unsigned long int post_flags,struct utofu_mrq_notice &in_notice )
{
  int rc;

  // confirm the local MRQ notification
  struct utofu_mrq_notice notice;
  if (post_flags & UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE) {
    do {
      rc = utofu_poll_mrq(vcq_hdl, 0, &notice);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    if(rc != UTOFU_SUCCESS){
      error->one(FLERR, "warp_utofu_poll_mrq fail {}\n", rc);
    }
    // assert(rc == UTOFU_SUCCESS);
    // assert(notice.notice_type == UTOFU_MRQ_TYPE_LCL_PUT);
    // assert(notice.edata == edata);
    edata = notice.edata;
  }
  in_notice = notice;
}

// confirm receiving data
void CommBrick::utofu_recv(utofu_vcq_hdl_t vcq_hdl, uint64_t &edata, unsigned long int post_flags, struct utofu_mrq_notice &in_notice)
{
  int rc;
  // confirm the remote MRQ notification or the memory update
  struct utofu_mrq_notice notice;
  if (post_flags & UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE) {
    do {
      rc = utofu_poll_mrq(vcq_hdl, 0, &notice);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    if(rc != UTOFU_SUCCESS){
      error->one(FLERR, "utofu_recv fail {} \n", rc);
    }
    // utils::logmesg(lmp, "in_notice rmt_stadd {} lcl_stadd_add {} edata {} vcq_id {}\n",
    //                       notice.rmt_stadd, notice.lcl_stadd, notice.edata, notice.vcq_id);
    // if(notice.notice_type != UTOFU_MRQ_TYPE_RMT_PUT) {
    //   error->one(FLERR, "notice.notice_type != UTOFU_MRQ_TYPE_RMT_PUT \n");
    // }
    // assert(rc == UTOFU_SUCCESS);
    // assert(notice.notice_type == UTOFU_MRQ_TYPE_RMT_PUT);
    // assert(notice.edata == edata);
    edata = notice.edata;
  }
  in_notice = notice;
}

void CommBrick::utofu_init_opt() {
  int rc, i;

  utofu_init_flag = true;

  if(DEBUG_MSG || me == 0) utils::logmesg(lmp," utofu_init_opt opt_maxswap {} \n", opt_maxswap);


  for(i = 0; i < VCQ_NUM; i++){
    lcl_send_stadd[i]         = new utofu_stadd_t[opt_maxswap];
    lcl_recv_stadd[i]         = new utofu_stadd_t[opt_maxswap];
    lcl_send_f_stadd[i]         = new utofu_stadd_t[opt_maxswap];
    lcl_recv_x_stadd[i]         = new utofu_stadd_t[opt_maxswap];
    lcl_comms[i]              = new Utofu_comm[opt_maxswap];
    lcl_recv_comms[i]         = new Utofu_comm[opt_maxswap];
    rmt_comms[i]              = new Utofu_comm[opt_maxswap];
  }

  edata                 = new uint8_t[opt_maxswap];
  cbvalue               = new uintptr_t[opt_maxswap];
  cbvalue_send          = new uint64_t[opt_maxswap];
  edata_send            = new uint64_t[opt_maxswap];
  edata_recvs            = new uint64_t[opt_maxswap];

  for(int i = 0; i < opt_maxswap; i++) {
    cbvalue[i] = edata[i] = i % 256;
  }

  // post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE |
  //               UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE |
  //               UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE ;
  post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE |
                UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE;


  for(int k = 0; k < VCQ_NUM; k++){
    for(int i = 0; i < TNI_NUM; i++) {
      tni_id = i;
      rc = utofu_query_onesided_caps(tni_id, &onesided_caps_send[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_query_onesided_caps error %d \n", rc);
      }
      rc = utofu_create_vcq(tni_id, 0, &vcq_hdl_send[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_create_vcq error %d tni_ids %d\n", rc, tni_id); 
      }
      rc = utofu_query_vcq_id(vcq_hdl_send[k][i], &lcl_vcq_id_send[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_query_vcq_id error %d \n", rc);
      }
    }
  }

  for(int k = 0; k < VCQ_NUM; k++){
    for(int i = 0; i < TNI_NUM; i++) {
      tni_id = i;
      rc = utofu_query_onesided_caps(tni_id, &onesided_caps_recv[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_query_onesided_caps error %d \n", rc);
      }
      rc = utofu_create_vcq(tni_id, 0, &vcq_hdl_recv[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_create_vcq error %d tni_ids %d\n", rc, tni_id); 
      }
      rc = utofu_query_vcq_id(vcq_hdl_recv[k][i], &lcl_vcq_id_recv[k][i]);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_query_vcq_id error %d \n", rc);
      }
    }
  }

  if(DEBUG_MSG || me == 0) utils::logmesg(lmp," utofu init nmax {} total_buffer_size {} \n", atom->nmax, total_buffer_size);

  for(int k = 0; k < VCQ_NUM; k++){
    for(int i = 0; i < TNI_NUM; i++) {
      rc  |= utofu_reg_mem(vcq_hdl_send[k][i], (void *)all_recv_buffer, sizeof(double) * total_buffer_size, 0, &all_send_stadd[k][i]);  
      rc  |= utofu_reg_mem(vcq_hdl_send[k][i], (void *)atom->f[0], sizeof(double) * atom->nmax * 3, 0, &all_f_stadd[k][i]);

      rc  |= utofu_reg_mem(vcq_hdl_recv[k][i], (void *)all_recv_buffer, sizeof(double) * total_buffer_size, 0, &all_recv_stadd[k][i]);   
      rc  |= utofu_reg_mem(vcq_hdl_recv[k][i], (void *)atom->x[0], sizeof(double) * atom->nmax * 3, 0, &all_x_stadd[k][i]); 
    }
  }

  for(int k = 0; k < VCQ_NUM; k++) {
    for(int i = 0; i < opt_maxswap; i++) {
      int vcq_ptr = ((i / 2) % TNI_NUM);
      lcl_send_stadd[k][i] = all_send_stadd[k][vcq_ptr];
      lcl_send_f_stadd[k][i] = all_f_stadd[k][vcq_ptr];
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_reg_mem vcq_hdl_send error k {} i {} vcq_ptr {} rc {}\n", k, i, vcq_ptr, rc); 
      }
      lcl_comms[k][i].vcq_id        = lcl_vcq_id_send[k][vcq_ptr];
      lcl_comms[k][i].utofu_stadd   = lcl_send_stadd[k][i];
      lcl_comms[k][i].utofu_f_stadd = lcl_send_f_stadd[k][i];
    }
  }
  for(int k = 0; k < VCQ_NUM; k++) {
    for(int i = 0; i < opt_maxswap; i++) {
      int vcq_ptr = ((i / 2) % TNI_NUM);
      lcl_recv_stadd[k][i] = all_recv_stadd[k][vcq_ptr];
      lcl_recv_x_stadd[k][i] = all_x_stadd[k][vcq_ptr];
      if(rc != UTOFU_SUCCESS) {
         error->one(FLERR,"utofu_reg_mem vcq_hdl_recv error k {} i {} vcq_ptr {} rc {}\n", k, i, vcq_ptr, rc); 
      }
      lcl_recv_comms[k][i].vcq_id         = lcl_vcq_id_recv[k][vcq_ptr];
      lcl_recv_comms[k][i].utofu_stadd    = lcl_recv_stadd[k][i];
      lcl_recv_comms[k][i].utofu_x_stadd  = lcl_recv_x_stadd[k][i];
    }
  }

  MPI_Request request;
  for(int k = 0; k < VCQ_NUM; k++){
    for(int i = 0; i < opt_maxswap; i++) {
      MPI_Irecv(&rmt_comms[k][i], 1, utofu_comm_type, opt_sendproc[i], 0, MPI_COMM_WORLD, &request);  
      MPI_Send(&lcl_recv_comms[k][i], 1, utofu_comm_type, opt_recvproc[i], 0, MPI_COMM_WORLD); 
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
  }

  for(int k = 0; k < VCQ_NUM; k++){
    for(int i = 0; i < opt_maxswap; i++) {
      rc = utofu_set_vcq_id_path(&rmt_comms[k][i].vcq_id, NULL);
      if(rc != UTOFU_SUCCESS) {
        error->one(FLERR,"utofu_set_vcq_id_path error\n"); 
      }
    }
  }

  if(comm->numa_flag) utofu_init_numa();
}

void CommBrick::utofu_init_numa() {
  int rc;

  if(DEBUG_MSG || me == 0) utils::logmesg(lmp," utofu_init_numa thr_maxswap {} numa_maxswap {} \n", thr_maxswap,numa_maxswap);

  for(int i = 0; i < VCQ_NUM; i++){
    lcl_recv_numa_comms[i]          = new Utofu_comm[numa_maxswap];
    nu_rmt_comms[i]                 = new Utofu_comm[numa_maxswap];
  }

  for(int k = 0; k < VCQ_NUM; k++) {
    for(int tid = 0; tid < thr_maxswap; tid++) {
      for(auto &iswap : thr_swap_full[tid]) {
        lcl_recv_numa_comms[k][iswap].vcq_id              = lcl_vcq_id_recv[k][tid];
        lcl_recv_numa_comms[k][iswap].utofu_stadd         = all_recv_stadd[k][tid];
      }
    }
  }

  MPI_Request request;

  for(int k = 0; k < VCQ_NUM; k++) {
    for(auto &iswap : numa_swap_full) {
      MPI_Irecv(&nu_rmt_comms[k][iswap], 1, utofu_comm_type, numa_sendproc[iswap], 0, MPI_COMM_WORLD, &request);  
      MPI_Send(&lcl_recv_numa_comms[k][iswap], 1, utofu_comm_type, numa_recvproc[iswap], 0, MPI_COMM_WORLD); 
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
  }
  
  for(int k = 0; k < VCQ_NUM; k++){
    for(auto &iswap : numa_swap_full) {
      rc |= utofu_set_vcq_id_path(&nu_rmt_comms[k][iswap].vcq_id, NULL);
    }
    if(rc != UTOFU_SUCCESS) {
      error->one(FLERR,"utofu_set_vcq_id_path error\n"); 
    }
  }

  MPI_Barrier(world);  

  // if(DEBUG_MSG || me == 0) utils::logmesg(lmp,"[INFO] finish init utofu numa\n");



}

void CommBrick::borders_one_parrral_numa(int tid) {
  int i,n,itype,iswap,dim,ineed, k, rswap;
  double **x;
  std::string mesg;
  AtomVec *avec = atom->avec;
  uint64_t edata_recv, r_edata_recv;
  int nlocal = atom->nlocal;

  struct utofu_mrq_notice in_notice;
  int nbuf_len;
  int pkt_size[NUMA_NUM];
  int pkt_offset[BORDER_ELEMENT][NUMA_NUM] = {{sizeof(double)}};
  
  x       = atom->x;

  if(tid == 11) {
    MPI_Allgather(&atom->nlocal,1,MPI_INT,numa_nlocal,1,MPI_INT,nuworld);

    for(int i = 0; i < BORDER_ELEMENT; i++) {
      int _len = i == 0 ? 24 : 4;
      for(int nu = 0; nu < NUMA_NUM; nu++) {
        if((i == BORDER_ELEMENT - 1) && (nu == NUMA_NUM -1)) break;
        int _o = i * NUMA_NUM + nu;
        *(pkt_offset[0] + _o + 1) =  *(pkt_offset[0] + _o) + _len * numa_nlocal[nu];
      }
    }

    nlocal_offset[0] = 0; 
    numa_nlocal_all = numa_nlocal[0];
    for(int nu = 1; nu < NUMA_NUM; nu++) {
      nlocal_offset[nu] = numa_nlocal[nu-1]+nlocal_offset[nu-1];
      numa_nlocal_all += numa_nlocal[nu];
    }

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] border nlocal "),  numa_nlocal, NUMA_NUM, 1);
    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] border pkt_offset "),  pkt_offset[0], NUMA_NUM * BORDER_ELEMENT, 1);

    // int ptr_offset = 0;
    // for(int nu = 1; nu <= numa_id; nu++) {
    //   ptr_offset += get_align_border_ptr(numa_nlocal[nu-1]) + 1;
    // }

    opt_buf_send[c_vcq][0][0] = numa_nlocal_all;
    for(int nu = 0; nu < NUMA_NUM; nu++) {
      // shm_st[nu].opt_buf_send[c_vcq][0][ptr_offset] = nlocal;
      // avec->pack_border_numa(shm_st[nu].opt_buf_send[c_vcq][0]+ptr_offset+1);
      memcpy((char*)(shm_st[nu].opt_buf_send[c_vcq][0]) + pkt_offset[0][numa_id], atom->x[0], sizeof(double) * nlocal * 3);
      memcpy((char*)(shm_st[nu].opt_buf_send[c_vcq][0]) + pkt_offset[1][numa_id], atom->tag,  sizeof(tagint) * nlocal);
      memcpy((char*)(shm_st[nu].opt_buf_send[c_vcq][0]) + pkt_offset[2][numa_id], atom->type, sizeof(int) * nlocal);
      memcpy((char*)(shm_st[nu].opt_buf_send[c_vcq][0]) + pkt_offset[3][numa_id], atom->mask, sizeof(int) * nlocal);
      *(shm_st[nu].a_written_tt[2]) += 1;
    }

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      lb_offset[nu] = nlocal_offset[nu];
      if(nu < numa_id) lb_offset[nu] += nlocal;
      else if(nu == numa_id) lb_offset[nu] = 0;
    }

    // load balance divide the atoms
    {
      if(lb_flag) {
        pair_len = 0;
        int ifrom, ito;
        int idelta_i = numa_nlocal_all / NUMA_NUM;
        int idelta_j = numa_nlocal_all % NUMA_NUM;
        int _bias    = idelta_j == 0 ? 0 : 1;
        
        if(numa_id >= idelta_j) {
          ifrom = (idelta_i + _bias) * idelta_j + idelta_i * (numa_id - idelta_j);
          ito = ifrom + idelta_i; 
        } else {
          ifrom = (idelta_i + _bias) * (numa_id);
          ito = ifrom + idelta_i + _bias; 
        }
        ito = (ito > numa_nlocal_all) ? numa_nlocal_all : ito; 
        
        for(int i = ifrom; i < ito; i++) {
          pair_index[pair_len] = i;
          if(pair_index[pair_len] < nlocal_offset[numa_id]) pair_index[pair_len] += nlocal;
          else if((pair_index[pair_len] >= nlocal_offset[numa_id]) && (pair_index[pair_len] < nlocal_offset[numa_id] + nlocal)) pair_index[pair_len] -= nlocal_offset[numa_id];

          pair_len++;
        }
      } else {
        pair_len = nlocal;
        for(int i = 0; i < nlocal; i++) pair_index[i] = i;  
      }


      if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] pair_index "),  pair_index, pair_len, 1);
    }


    memset(shm_numa_recvnum_numa[numa_id], 0, sizeof(uint64_t)*(numa_maxswap));
    numa_recvnum = shm_numa_recvnum_numa[numa_id];

    while((*(shm_st[numa_id].a_written_tt[2]) != NUMA_NUM));
    *(shm_st[numa_id].a_written_tt[2]) = 0;


    if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border finish atom memcpy \n"); 

    if(DEBUG_MSG)
      for(int nu = 0; nu < NUMA_NUM; nu++) {
          utils::logmesg_arry_x(lmp, fmt::format("[NUMA] selfsend iswap {}", 0), 
                                (double*)((char*)(opt_buf_send[c_vcq][0])+pkt_offset[0][nu]), numa_nlocal[nu]*3, 1); 
      }
  }

  lmp->parral_barrier(12, tid);

    // if(DEBUG_MSG) {
    //   utils::logmesg_arry_parral(lmp, fmt::format("[NUMA] border nbuf_len tid {} ", tid), nbuf_len, numa_swap[tid]); 
    // }
  if(tid < thr_maxswap) {
    nbuf_len = 0;
    nbuf_len = numa_nlocal_all * (8 * 3 + 4 + 4 + 4) + 8;
    // for(int nu = 0; nu < NUMA_NUM; nu++) {
    //   nbuf_len += recv_swap_mnt[numa_maxswap].pkt_size[nu];
    // }

    // if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border tid {} nbuflen {}\n", tid, nbuf_len); 


    for(auto &iswap : thr_swap[tid]) {
      if(nbuf_len > opt_maxsend[0]) {
        error->one(FLERR, "[ERROR]  buffer not enough maxsend {} buflen {}", opt_maxsend[0], nbuf_len );
      }
      warp_utofu_put(vcq_hdl_send[c_vcq][tid], nu_rmt_comms[c_vcq][iswap].vcq_id, all_send_stadd[c_vcq][tid]+opt_stadd_send_offset[c_vcq][0], 
              nu_rmt_comms[c_vcq][iswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][iswap], sizeof(double) * nbuf_len, edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][0]);  
    }
    for(auto &iswap : thr_swap[tid]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][tid], cbvalue_send[iswap], post_flags);
    }

    for(auto &iswap : thr_swap[tid]) {
      utofu_recv(vcq_hdl_recv[c_vcq][tid], edata_recv, post_flags, in_notice);
      numa_recvnum[edata_recv] = opt_buf_recv[c_vcq][edata_recv][0];        

      int _len = numa_recvnum[edata_recv];
      if(numa_pbc_flag_recv[edata_recv]) {
        double dx, dy, dz;
        dx = numa_pbc_recv[edata_recv][0] * domain->xprd;  dy = numa_pbc_recv[edata_recv][1] * domain->yprd;  dz = numa_pbc_recv[edata_recv][2] * domain->zprd;
        for(int i = 0; i < _len; i++) {
          opt_buf_recv[c_vcq][edata_recv][1 + i * 3 + 0] += dx;
          opt_buf_recv[c_vcq][edata_recv][1 + i * 3 + 1] += dy;
          opt_buf_recv[c_vcq][edata_recv][1 + i * 3 + 2] += dz;
        }

        if(DEBUG_MSG) utils::logmesg_arry_x(lmp, fmt::format("[NUMA] recve {} flag {} {} {} {}",
                                  edata_recv, numa_pbc_flag_recv[edata_recv], numa_pbc_recv[edata_recv][0],numa_pbc_recv[edata_recv][1],numa_pbc_recv[edata_recv][2]), 
                                       &opt_buf_recv[c_vcq][edata_recv][1], _len * 3, 1); 

      }
    }
  }

  if(tid == 11) {
    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      char* buf = (char*)opt_buf_send[c_vcq][0];
      int len = numa_nlocal[nu];
      memcpy(atom->x[0] + lb_offset[nu] * 3,  (void*)(buf + pkt_offset[0][nu]), sizeof(double) * len * 3); 
      memcpy(atom->tag  + lb_offset[nu],      (void*)(buf + pkt_offset[1][nu]), sizeof(tagint) * len);     
      memcpy(atom->type + lb_offset[nu],      (void*)(buf + pkt_offset[2][nu]), sizeof(int) * len);
      memcpy(atom->mask + lb_offset[nu],      (void*)(buf + pkt_offset[3][nu]), sizeof(int) * len);
    }
  }

  lmp->parral_barrier(12, tid);

  //  通信 recvnum  firstrecv
  if(tid == 11) {

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] shm_numa_recvnum_numa "),  numa_recvnum, (numa_maxswap), 1); 

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      memcpy(shm_st[nu].shm_numa_recvnum_numa[numa_id],  numa_recvnum, sizeof(uint64_t)*(numa_maxswap));
      *(shm_st[nu].a_written_tt[3]) += 1;
    }

    while(*(a_written_tt[3]) != 3);
    *(a_written_tt[3]) = 0;

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      for(int i = 0; i < numa_maxswap; i++) {
        if(*(numa_recvnum+i) == 0) *(numa_recvnum+i) = *(shm_numa_recvnum_numa[nu]+i);
      }
    }
    // for(int nu = 0; nu < NUMA_NUM; nu++) {
    //   if(nu == numa_id) continue;
    //   if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] shm_numa_recvnum_numa nu {} ", nu),  shm_numa_recvnum_numa[nu], numa_maxswap, 1); 
    // }

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] shm_numa_recvnum_numa "),  numa_recvnum, numa_maxswap, 1); 

    numa_firstrecv[0] = numa_nlocal_all;
    for(int iswap = 1; iswap < numa_maxswap; iswap++) {
      numa_firstrecv[iswap] = numa_recvnum[iswap-1] + numa_firstrecv[iswap-1];
    }
    for(int iswap = 0; iswap < numa_maxswap; iswap++) {
      if(numa_recvnum[iswap] <= 0 || numa_recvnum[iswap] > opt_maxsend[0]) {
        error->one(FLERR, "[ERROR]  numa_recvnum error {}\n", numa_recvnum[iswap]);
      }
    }

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] numa_firstrecv "), numa_firstrecv, numa_maxswap, 1); 

    atom->nghost = 0;

    for(int iswap = 0; iswap < numa_maxswap; iswap++) {
      atom->nghost += numa_recvnum[iswap];
    }

    atom->nunlocal = numa_nlocal_all;
    atom->nunghost = atom->nghost;

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      atom->nghost += numa_nlocal[nu];
    }

    if(atom->nghost + atom->nlocal > atom->nmax ) {
      error->one(FLERR, "[ERROR]  atom exceed nmax {} atom nlocl {} nghost {} \n", atom->nmax, atom->nlocal, atom->nghost);
    }

    if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border begin to copy buffer \n"); 
  }

  lmp->parral_barrier(12, tid);

  if(tid < thr_maxswap) {
    for(auto &iswap : thr_swap[tid]) {
      for(int nu = 0; nu < NUMA_NUM; nu++) {
          char* buf = (char*)opt_buf_recv[c_vcq][iswap];
          int offset = sizeof(double);
          int len = numa_recvnum[iswap];
            // if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border unpack nu {} ptr {} first_recv {} len {} offset {} \n", nu, ptr, _first_recv, len, recv_swap_mnt[iswap].pkt_offset[ii]+1);

            // if(DEBUG_MSG) utils::logmesg_arry_x(lmp, fmt::format("[NUMA] recied x nu {} iswap {} ", ), buf,  * 3, 1); 
          memcpy(shm_st[nu].x[0] + numa_firstrecv[iswap] * 3,  (void*)(buf + offset), sizeof(double) * len * 3);  offset += sizeof(double) * len * 3;
          memcpy(shm_st[nu].tag  + numa_firstrecv[iswap],      (void*)(buf + offset), sizeof(tagint) * len);      offset += sizeof(tagint) * len;
          memcpy(shm_st[nu].type + numa_firstrecv[iswap],      (void*)(buf + offset), sizeof(int) * len);         offset += sizeof(int) * len;
          memcpy(shm_st[nu].mask + numa_firstrecv[iswap],      (void*)(buf + offset), sizeof(int) * len);         offset += sizeof(int) * len;
      }
    }
  }
  
  // __asm__ __volatile__ ("" ::: "memory");

  lmp->parral_barrier(12, tid);

  if(tid == 11) {
    // for(int iswap = 0; iswap < numa_maxswap; iswap++)
    //   if(tid == 11)  if(DEBUG_MSG) utils::logmesg_arry_x(lmp, fmt::format("[NUMA] atom->x iswap {} recvporc {} ", iswap, numa_recvproc[iswap]), 
    //                                     atom->x[0] + numa_firstrecv[iswap] * 3, numa_recvnum[iswap] * 3, 1); 

    if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border finish copy buf, begin to sync  \n"); 


    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      *(shm_st[nu].a_written_tt[0]) += 1;
    }

    while(1) {
      int _t = *(a_written_tt[0]);
      if(_t == 3) break;
    }
    *(a_written_tt[0]) = 0;

    // MPI_Barrier(nuworld);

    if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border finish snyc  \n"); 


    // if(DEBUG_MSG) {
    //   std::string tmp;
    //   tmp +=  fmt::format("[info] pair_index len {} ", pair_len);
    //   for(int i = 0; i < pair_len; i++) {
    //       tmp += fmt::format("  {}:{}:{}", i, pair_index[i], atom->tag[pair_index[i]]);
    //   }
    //   tmp += "\n";
    //   utils::logmesg(lmp, tmp);
    // }  
  }

  if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] border in finish tid {}  \n", tid); 



}

void CommBrick::forward_comm_parral_numa(int tid) {
  int nbuf_len;

  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;
  int iswap;
  struct utofu_mrq_notice in_notice;
  int i;

  std::string mesg;
  uint64_t edata_recv;

  if(tid < NUMA_NUM) {
    memcpy(shm_st[tid].opt_buf_send[c_vcq][0] + nlocal_offset[numa_id] * 3 , atom->x[0], sizeof(double) * atom->nlocal * 3);
    *(shm_st[tid].a_written_tt[1]) += 1;

    if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] forward finish atom memcpy nlocal tid {} \n", tid);
  }

  lmp->parral_barrier(12, tid);

  if(tid < thr_maxswap) {
    while((*(a_written_tt[1])) != NUMA_NUM);

    nbuf_len = numa_nlocal_all * 3;

    for(auto &iswap : thr_swap[tid]) {
      if(nbuf_len > opt_maxsend[0]) {
        error->one(FLERR, "[ERROR]  buffer not enough maxsend {} buflen {}", opt_maxsend[0], nbuf_len );
      }
      warp_utofu_put(vcq_hdl_send[c_vcq][tid], nu_rmt_comms[c_vcq][iswap].vcq_id, all_send_stadd[c_vcq][tid]+opt_stadd_send_offset[c_vcq][0], 
                nu_rmt_comms[c_vcq][iswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][iswap], sizeof(double) * nbuf_len, edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][0]);  

      // if(DEBUG_MSG)  utils::logmesg_arry_x(lmp, fmt::format("[NUMA] send x iswap {}", 0), opt_buf_send[c_vcq][0], numa_nlocal_all * 3, 1); 
    }
    for(auto &iswap : thr_swap[tid]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][tid], cbvalue_send[iswap], post_flags);
    }

    for(auto &iswap : thr_swap[tid]) {
      utofu_recv(vcq_hdl_recv[c_vcq][tid], edata_recv, post_flags, in_notice);

      int _len = numa_recvnum[edata_recv];
      if(numa_pbc_flag_recv[edata_recv]) {
        double dx, dy, dz;
        dx = numa_pbc_recv[edata_recv][0] * domain->xprd;  dy = numa_pbc_recv[edata_recv][1] * domain->yprd;  dz = numa_pbc_recv[edata_recv][2] * domain->zprd;
        for(int i = 0; i < _len; i++) {
          opt_buf_recv[c_vcq][edata_recv][i * 3 + 0] += dx;
          opt_buf_recv[c_vcq][edata_recv][i * 3 + 1] += dy;
          opt_buf_recv[c_vcq][edata_recv][i * 3 + 2] += dz;
        }
      }
      for(int nu = 0; nu < NUMA_NUM; nu++) {
        memcpy(shm_st[nu].x[0] + numa_firstrecv[edata_recv] * 3,  opt_buf_recv[c_vcq][edata_recv], sizeof(double) * _len * 3); 
      }
    }
  }

  if(tid == 11) {
    while((*(a_written_tt[1])) != NUMA_NUM);
    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      int len = numa_nlocal[nu];
      memcpy(atom->x[0] + lb_offset[nu] * 3,  opt_buf_send[c_vcq][0]+ nlocal_offset[nu]*3, sizeof(double) * len * 3); 
    }
  }

  lmp->parral_barrier(12, tid);

  if(tid == 11) {
    *(a_written_tt[1]) = 0;

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      *(shm_st[nu].a_written_tt[0]) += 1;
    }

    while(*(a_written_tt[0]) != 3);
    *(a_written_tt[0]) = 0;
  }


  // if(DEBUG_MSG)  utils::logmesg(lmp," opt_forward_parral finish parral tid {} \n", tid);  
  
  #ifdef THR_TIME_TEST
    td->timer(Timer::FORWARD);
  #endif
}

void CommBrick::reverse_comm_parral_numa(int tid) {
  int n, i;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap, rswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv, r_edata_recv;
  int nbuf_len[RPROC];

  if(tid == 11) {
    self_timer->stamp();
  }

  for(int iswap = tid; iswap < numa_maxswap; iswap += T_THREAD) {
    int _to_numa_id = swap2numa_swap[iswap];

    if(_to_numa_id != numa_id)
      memcpy(shm_st[_to_numa_id].opt_force_recv[numa_id][iswap], atom->f[0]+numa_firstrecv[iswap]*3, numa_recvnum[iswap]*3*sizeof(double));
    else
      memcpy(opt_buf_send[c_vcq][iswap], atom->f[0]+numa_firstrecv[iswap]*3, numa_recvnum[iswap]*3*sizeof(double));
    // if(DEBUG_MSG && (numa_recvproc[iswap] / 4 == 0)) 
    //       utils::logmesg_arry_x(lmp,fmt::format("[info] send nu 2 "), atom->f[0]+(numa_firstrecv[iswap] + 108*2)*3, 3 * 108, 1);
    
    __asm__ __volatile__ ("" ::: "memory");
    *(shm_st[_to_numa_id].a_written_reverse[iswap]) += 1;

    // if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] reverse finish copy opt iswap {} numa_id {} swap {} \n", 
    //                                             iswap, _to_numa_id, iswap);
  }

  // lmp->parral_barrier(12, tid);

  // if(tid == 11) {
  //   MPI_Barrier(nuworld);
  //   while(*(a_written_reverse[numa_maxswap]) != reverse_target_bits[numa_maxswap]);
  //   self_timer->stamp(Timer::REVERSE_CPY);
  // }

  // lmp->parral_barrier(12, tid);


  if(tid < thr_maxswap) {
    for(auto &iswap : thr_swap[tid]) {
      nbuf_len[iswap] = numa_recvnum[iswap] * 3;
    }

    for(auto &iswap : thr_swap[tid]) {
      int rswap = iswap % 2 == 0 ? iswap + 1 : iswap -1;
      
      while(*(a_written_reverse[iswap]) != NUMA_NUM);
      *(a_written_reverse[iswap]) = 0;   

      for(int nu = 0; nu < NUMA_NUM; nu++) {
        if(nu == numa_id) continue;
        for(int atom_i = 0; atom_i < numa_recvnum[iswap] * 3; atom_i++) {
          opt_buf_send[c_vcq][iswap][atom_i] += opt_force_recv[nu][iswap][atom_i];
        }
      }
      
      // if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] send iswap {} proc {} nbuf_len {} ", rswap, numa_sendproc[rswap], nbuf_len[iswap]), 
      //                 opt_buf_send[c_vcq][iswap], (recv_swap_mnt[iswap].atom_num[3] + recv_swap_mnt[iswap].atom_offset[3]) * 3, 1);
      if(nbuf_len[iswap] > opt_maxsend[0]) {
        error->one(FLERR, "[ERROR]  buffer not enough maxsend {} buflen {}", opt_maxsend[0], nbuf_len[iswap] );
      }

      warp_utofu_put(vcq_hdl_send[c_vcq][tid], nu_rmt_comms[c_vcq][rswap].vcq_id, all_send_stadd[c_vcq][tid]+opt_stadd_send_offset[c_vcq][iswap], 
        nu_rmt_comms[c_vcq][rswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][iswap],  nbuf_len[iswap] * sizeof(double), edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
    }

    for(auto &iswap : thr_swap[tid]) {
      int rswap = iswap % 2 == 0 ? iswap + 1 : iswap -1;
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][tid], cbvalue_send[rswap], post_flags);
    }

    if(thr_swap[tid].size()) {
      int _swap = thr_swap[tid][0];
      memset(opt_buf_send[c_vcq][_swap], 0, sizeof(double) * 3 * numa_nlocal_all);
    }
    
    for(auto &iswap : thr_swap[tid]) {
      int _swap = thr_swap[tid][0];
      int rswap = iswap % 2 == 0 ? iswap + 1 : iswap -1;

      utofu_recv(vcq_hdl_recv[c_vcq][tid], edata_recv, post_flags, in_notice);

      int r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv -1;
      // if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] reverse recv {} \n", edata_recv);

      // if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] recv iswap {} recvproc {}", edata_recv, numa_recvproc[edata_recv]), opt_buf_recv[c_vcq][edata_recv], 3 * numa_nlocal_all, 1);
      // if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] recv nu 2 iswap {} recvproc {}", edata_recv, numa_recvproc[edata_recv]), opt_buf_recv[c_vcq][edata_recv]+nlocal_offset[2]*3, 3 * numa_nlocal[2], 1);

      for(int atom_i = 0; atom_i < 3 * numa_nlocal_all; atom_i++) {
        opt_buf_send[c_vcq][_swap][atom_i] += opt_buf_recv[c_vcq][edata_recv][atom_i];
      }
    }
  }

  if(tid == 11) {
    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      int atom_i, atom_j, i;
      for(atom_i = nlocal_offset[nu], atom_j = lb_offset[nu], i = 0; i < numa_nlocal[nu]; atom_i++, atom_j++, i++) {
        opt_force_recv[numa_id][numa_maxswap][atom_i*3+0] += atom->f[atom_j][0];
        opt_force_recv[numa_id][numa_maxswap][atom_i*3+1] += atom->f[atom_j][1];
        opt_force_recv[numa_id][numa_maxswap][atom_i*3+2] += atom->f[atom_j][2];
      }
    }
  }

  lmp->parral_barrier(12, tid);

  if(tid == 11) {
    // if(DEBUG_MSG) utils::logmesg(lmp,"[info] begin reduce all thread data thr_maxswap {} \n", thr_maxswap);

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] lb_offset "),  lb_offset, NUMA_NUM, 1); 
    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("[INFO] nlocal_offset "),  nlocal_offset, NUMA_NUM, 1); 

    for(int _tid = 0; _tid < thr_maxswap; _tid++) {
      if(thr_swap[_tid].size()) {
        int _swap = thr_swap[_tid][0];
        for(int atom_i = 0; atom_i < 3 * numa_nlocal_all; atom_i++) {
          opt_force_recv[numa_id][numa_maxswap][atom_i] += opt_buf_send[c_vcq][_swap][atom_i];
        }
      }
    }

    // if(DEBUG_MSG) utils::logmesg(lmp,"[info] finish reduce all thread data\n");

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      memcpy(shm_st[nu].opt_force_recv[numa_id][numa_maxswap], opt_force_recv[numa_id][numa_maxswap] + nlocal_offset[nu] * 3, sizeof(double)*numa_nlocal[nu]*3);
      *(shm_st[nu].a_written_tt[4]) += 1;
    }

    for(int atom_i = 0; atom_i < atom->nlocal; atom_i++) {
      atom->f[atom_i][0] += opt_force_recv[numa_id][numa_maxswap][(atom_i + nlocal_offset[numa_id])*3+0];
      atom->f[atom_i][1] += opt_force_recv[numa_id][numa_maxswap][(atom_i + nlocal_offset[numa_id])*3+1];
      atom->f[atom_i][2] += opt_force_recv[numa_id][numa_maxswap][(atom_i + nlocal_offset[numa_id])*3+2];
    }

    while((*(a_written_tt[4])) != 3);
    *(a_written_tt[4]) = 0;

    for(int nu = 0; nu < NUMA_NUM; nu++) {
      if(nu == numa_id) continue;
      for(int atom_i = 0; atom_i < atom->nlocal; atom_i++) {
        atom->f[atom_i][0] += opt_force_recv[nu][numa_maxswap][atom_i*3+0];
        atom->f[atom_i][1] += opt_force_recv[nu][numa_maxswap][atom_i*3+1];
        atom->f[atom_i][2] += opt_force_recv[nu][numa_maxswap][atom_i*3+2];
      }
    }

    memset(opt_force_recv[numa_id][numa_maxswap], 0, numa_nlocal_all * 3 * sizeof(double));
    // self_timer->stamp(Timer::REVERSE_REDUCE);
  }


  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] finish opt_reverse_comm_parrel over tid {}\n", tid);
  // }




  // #ifdef THR_TIME_TEST
  //   td->timer(Timer::REVERSE);
  // #endif
}

void CommBrick::reverse_comm_parral_unpack_numa(){
  int iswap, rswap;
  AtomVec *avec = atom->avec;

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm_parral_unpack remaind_iswap "), remaind_iswap, opt_maxswap, COMM_STEP); 
  // }

  for(iswap = 0 ; iswap < opt_maxswap; iswap += comm_step) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_recv[rswap]) {
      if(remaind_iswap[rswap]){
        avec->unpack_reverse(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_recv[c_vcq][rswap]);
      }
    }
  }
}


#ifdef OPT_COMM_TEST
void CommBrick::borders() {
  int i,n,itype,iswap,dim,ineed, k, rswap;  
  double **x;
  std::string mesg;
  
  AtomVec *avec = atom->avec;
  uint64_t edata_recv;
  uint64_t r_edata_recv;

  x       = atom->x;

  for(i = 0; i < opt_maxswap; i+=COMM_STEP) {
    opt_sendnum[i] = 0;
  }

  if(DEBUG_MSG) utils::logmesg(lmp,"[info] begin opt_border maxiswap {}  local {}\n", opt_maxswap, atom->nlocal);   
 
  int ibin;
  for(i = 0; i < atom->nlocal; i++) {
    ibin = atom2bin(x[i], bin_split_line);

    for(int j = 0; j < bin2swap_ptr[ibin]; j++) {
      iswap = bin2swap[ibin][j];
      opt_sendlist[iswap][opt_sendnum[iswap]++] = i;
      if (opt_sendnum[iswap] == opt_maxsendlist[iswap]) {
        error->one(FLERR, "sendlist not enough iswap {} opt_sendnum {} maxsendlist {}\n", iswap, opt_sendnum[iswap], opt_maxsendlist[iswap]);
      }
    }
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_sendnum "), opt_sendnum, opt_maxswap, COMM_STEP); 
  }

  struct utofu_mrq_notice in_notice;
  int nbuf_len[SWAP_NUM];
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]*size_border > opt_maxsend[iswap]) {
      error->one(FLERR, "[info] opt_maxsend not enough  me = {} opt_sendnum {} maxsend {}\n", 
                                                me, opt_sendnum[iswap]*size_border, opt_maxsend[iswap]);
    }

    nbuf_len[iswap] = 1 + avec->pack_border(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap]+1,opt_pbc_flag[iswap],opt_pbc[iswap]);
    opt_buf_send[c_vcq][iswap][0] = opt_sendnum[iswap];
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one nbuf_len "), nbuf_len, opt_maxswap, COMM_STEP); 
  }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
            rmt_comms[c_vcq][iswap].utofu_stadd, sizeof(double) * nbuf_len[iswap], edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
  }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
  }
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
  }
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);
    opt_recvnum[edata_recv] =  opt_buf_recv[c_vcq][edata_recv][0];
  }
  
  opt_firstrecv[0] = atom->nlocal;
  for (iswap = COMM_STEP; iswap < opt_maxswap; iswap += COMM_STEP) {
    opt_firstrecv[iswap] = opt_firstrecv[iswap-COMM_STEP] + opt_recvnum[iswap-COMM_STEP];
  }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    avec->unpack_border(opt_recvnum[iswap], opt_firstrecv[iswap], opt_buf_recv[c_vcq][iswap]+1);
  }


  c_vcq = (c_vcq + 1) % VCQ_NUM;

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_recvnum "), opt_recvnum, opt_maxswap, COMM_STEP); 
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_firstrecv "), opt_firstrecv, opt_maxswap, COMM_STEP); 
  }

  for (iswap = 0; iswap < opt_maxswap; iswap += COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    opt_buf_send[c_vcq][rswap][0] = opt_firstrecv[iswap];
  }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_stadd[c_vcq][rswap], 
            rmt_comms[c_vcq][rswap].utofu_stadd, sizeof(double), edata[rswap], cbvalue[rswap], post_flags, (void*)opt_buf_send[c_vcq][rswap]);  
  }  

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
  }
  
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
  }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);   
    r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;
    opt_forw_pos[r_edata_recv] =  opt_buf_recv[c_vcq][edata_recv][0];
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_forw_pos "), opt_forw_pos, opt_maxswap, COMM_STEP); 
  }

  for(iswap = 0; iswap < opt_maxswap; iswap += COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    opt_size_forward_recv[iswap] = opt_recvnum[iswap]*size_forward;
    opt_size_reverse_send[rswap] = opt_recvnum[iswap]*size_reverse;
    opt_size_reverse_recv[rswap] = opt_sendnum[iswap]*size_reverse;
    opt_reverse_send_pos[rswap] = opt_firstrecv[iswap]*sizeof(double)*3;
    opt_forward_send_pos[iswap] = opt_forw_pos[iswap]*sizeof(double)*3;
    atom->nghost += opt_recvnum[iswap];
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_size_forward_recv "), opt_size_forward_recv, opt_maxswap, COMM_STEP); 
    utils::logmesg_arry_r(lmp, fmt::format("opt_border_one opt_size_reverse_send "), opt_size_reverse_send, opt_maxswap, COMM_STEP); 
    utils::logmesg_arry_r(lmp, fmt::format("opt_border_one opt_size_reverse_recv "), opt_size_reverse_recv, opt_maxswap, COMM_STEP); 
    utils::logmesg_arry_r(lmp, fmt::format("opt_border_one opt_reverse_send_pos "), opt_reverse_send_pos, opt_maxswap, COMM_STEP); 
    utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_forward_send_pos "), opt_forward_send_pos, opt_maxswap, COMM_STEP); 
  }

  c_vcq = (c_vcq + 1) % VCQ_NUM;

  if ((atom->molecular != Atom::ATOMIC)
      && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,"Per-processor number of atoms is too large for "
               "molecular neighbor lists");
};

/* ----------------------------------------------------------------------
   walk up/down the extent of nearby processors in dim and dir
   loc = myloc of proc to start at
   dir = 0/1 = walk to left/right
   do not cross non-periodic boundaries
   is not called for z dim in 2d
   return how many procs away are needed to encompass cutghost away from loc
------------------------------------------------------------------------- */



void CommBrick::exchange() {
  int i,j,m,nlocal, nrecv[SWAP_NUM], nsend[SWAP_NUM], dim;
  double value[3];
  double **x;
  double *sublo,*subhi;
  AtomVec *avec = atom->avec;
  int iswap;
  std::string mesg;
  struct utofu_mrq_notice in_notice;
  uint64_t edata_recv;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // if(DEBUG_MSG) utils::logmesg(lmp, "[info] exchange begin cur c_vcq: {} \n", c_vcq);


  // insure send buf has extra space for a single atom
  // only need to reset if a fix can dynamically add to size of single atom

  if (maxexchange_fix_dynamic) {
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
  }

  // subbox bounds for orthogonal or triclinic

  if (triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  x = atom->x;
  nlocal = atom->nlocal;
  i = 0;
  for (iswap = 0; iswap < opt_maxswap; iswap++) {
    nsend[iswap] = 1;
  }

  bool belong;
   
  while (i < nlocal) {
    if(x[i][0] >= sublo[0] && x[i][0] < subhi[0] &&
        x[i][1] >= sublo[1] && x[i][1] < subhi[1] &&
        x[i][2] >= sublo[2] && x[i][2] < subhi[2]) {
          i++;
          continue;
        }
    belong = false;
      
    for(int k = 0; k < 3; k++){
      if(x[i][k] < sublo[k] || x[i][k] >= subhi[k]) {
        for(iswap = 0; iswap < EXCH_SWAP_NUM; iswap++){
          if(swap_direct[iswap][k] != 0) {
            belong = in_neighbor_box(x[i], neigh_sublo[opt_sendproc[iswap]],neigh_subhi[opt_sendproc[iswap]]);      
            if(belong) break;
          }
        }
        if(belong) break;
      }
    }

    if(belong) {
      nsend[iswap] += avec->pack_exchange(i,&opt_buf_send[c_vcq][iswap][nsend[iswap]]);
      // utils::logmesg(lmp, " echange x {:.2f} {:.2f} {:.2f} iswap {} current nsend {} \n", x[i][0], x[i][1], x[i][2], iswap, nsend[iswap]);

      if(nsend[iswap] > opt_maxsend[iswap]) {
        error->one(FLERR, "[info] me {} opt_cexchange overflow iswap {} recvnum {} \n", me, iswap, nsend[iswap]);
      }
      avec->copy(nlocal-1,i,1);
      nlocal--;

    } else {
      error->one(FLERR, "[info] me {} opt_exchange atom not find node iswap {} x {} {} {}, neibor {}:{} {}:{} {}:{}\n", 
            me, iswap,x[i][0],x[i][1],x[i][2],
            neigh_sublo[opt_sendproc[iswap]][0], neigh_subhi[opt_sendproc[iswap]][0],
            neigh_sublo[opt_sendproc[iswap]][1], neigh_subhi[opt_sendproc[iswap]][1],
            neigh_sublo[opt_sendproc[iswap]][2], neigh_subhi[opt_sendproc[iswap]][2]);
      i++;
    }
  }

  atom->nlocal = nlocal;

  
  for (i = 0; i < EXCH_SWAP_NUM; i+=1) {
    opt_buf_send[c_vcq][i][0] = nsend[i];
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("[info] exchange nsend "), nsend, EXCH_SWAP_NUM, 1);
  }

  for (iswap = EXCH_SWAP_NUM - 1; iswap >= 0; iswap--) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap]+opt_stadd_send_offset[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][iswap], sizeof(double) * nsend[iswap], edata[iswap], cbvalue[iswap], post_flags, opt_buf_send[c_vcq][iswap]);  
  }

  for (iswap = EXCH_SWAP_NUM - 1; iswap >= 0; iswap--) {
    warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
  }

  // if(DEBUG_MSG) utils::logmesg(lmp, "exchange recv ");
  for (iswap = 0; iswap < EXCH_SWAP_NUM; iswap+=1) {
    utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice); 
    // if(DEBUG_MSG) utils::logmesg(lmp," {}:{} ", edata_recv, (int)opt_buf_recv[c_vcq][edata_recv][0]);
    nrecv[edata_recv] = opt_buf_recv[c_vcq][edata_recv][0];

    if(nrecv[edata_recv] > 1) {
      m = 1;
      while(m < nrecv[edata_recv]) {
        for(j = 0; j < 3; j++){
          value[j] = opt_buf_recv[c_vcq][edata_recv][m+j+1];
        }
        if (value[0] >= sublo[0] && value[0] < subhi[0] &&
              value[1] >= sublo[1] && value[1] < subhi[1] && 
              value[2] >= sublo[2] && value[2] < subhi[2]) {
              m += avec->unpack_exchange(&opt_buf_recv[c_vcq][edata_recv][m]);
        }
        else {
          utils::logmesg(lmp,"recv not in this node iswap {} x {} {} {} \n", edata_recv, value[0],value[1],value[2]);
          m += static_cast<int> (opt_buf_recv[c_vcq][edata_recv][m]);
        }
      }
    }
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format("[info] exchange nlocal {} nrecvc ", atom->nlocal), nrecv, EXCH_SWAP_NUM, 1);
  }

  c_vcq = (c_vcq + 1) % VCQ_NUM;


  if (atom->firstgroupname) atom->first_reorder();  
}

void CommBrick::forward_comm(int) {

  int nbuf_len[SWAP_NUM];

  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;
  int iswap;
  struct utofu_mrq_notice in_notice;
  int i;
  uint64_t edata_recv;

  std::string mesg;

  // if(c_vcq % 2 != FORWARD_PTR) {
  //   c_vcq = (c_vcq + 1) % VCQ_NUM; 
  // }

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format("opt_forward_comm opt_sendnum "), opt_sendnum, opt_maxswap, COMM_STEP); 
  // }
  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format("opt_forward_comm opt_forward_send_atom "), opt_forward_send_pos, opt_maxswap, COMM_STEP); 
  // }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    nbuf_len[iswap] = avec->pack_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap],opt_pbc_flag[iswap],opt_pbc[iswap]);
    if (nbuf_len[iswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_x_stadd+opt_forward_send_pos[iswap], sizeof(double) * nbuf_len[iswap], edata[iswap], cbvalue[iswap], post_flags, opt_buf_send[c_vcq][iswap]);  
    }
  }
  

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format("opt_forward_comm nbuf_len "), nbuf_len, opt_maxswap, COMM_STEP); 
  // }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] finish forward pack\n");
  // }

  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (nbuf_len[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
    }
  }
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (nbuf_len[iswap]) {
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
    }
  }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] finish forward send\n");
  // }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] recv  ");
  // }
  for(iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if(opt_size_forward_recv[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice); 
      // if(DEBUG_MSG) {
      //   utils::logmesg(lmp," {}:{}:{}  ", edata_recv, opt_size_forward_recv[edata_recv]/size_forward, opt_firstrecv[edata_recv]);
      // }
    }
  }
  // if(DEBUG_MSG) {
  //     utils::logmesg(lmp," \n");
  // }

  c_vcq = (c_vcq + 1) % VCQ_NUM;
}

void CommBrick::reverse_comm()
{
  int n, i;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv;
  uint64_t r_edata_recv;

  int rswap;

  // if(c_vcq % 2 != REVERSE_PTR) {
  //   c_vcq = (c_vcq + 1) % VCQ_NUM; 
  // }

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm opt_size_reverse_send "), opt_size_reverse_send, opt_maxswap, COMM_STEP); 
  // }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_send[rswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_f_stadd[c_vcq][rswap]+opt_reverse_send_pos[rswap],
                  rmt_comms[c_vcq][rswap].utofu_stadd, sizeof(double) * opt_size_reverse_send[rswap], 
                  edata[rswap], cbvalue[rswap], post_flags,atom->f[opt_firstrecv[iswap]]);  
    }
  }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] finish reverse send\n");
  // }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_send[rswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_send[rswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
    }
  }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp,"[info] reverse recv  ");
  // }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_recv[rswap]) { 
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);
      r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;

      // if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);
      
      avec->unpack_reverse(opt_sendnum[r_edata_recv],opt_sendlist[r_edata_recv],opt_buf_recv[c_vcq][edata_recv]);
    }
  }

  // if(DEBUG_MSG) utils::logmesg(lmp," \n");
  c_vcq = (c_vcq + 1) % VCQ_NUM;
  
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Pair *pair)
{
  int iswap;
  double *buf;
  MPI_Request request;
  int n[SWAP_NUM];
  struct utofu_mrq_notice in_notice;

  uint64_t edata_recv;

  int nsize = pair->comm_forward;

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    n[iswap] = pair->pack_forward_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap],opt_pbc_flag[iswap],opt_pbc[iswap]);
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry(lmp, fmt::format(" pair forward_comm n "), n, opt_maxswap, COMM_STEP); 
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_stadd, sizeof(double) * n[iswap] , edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
    }
  }

  if(DEBUG_MSG) utils::logmesg(lmp," pair forward_comm ");

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if(opt_recvnum[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);    
      pair->unpack_forward_comm(opt_recvnum[edata_recv],opt_firstrecv[edata_recv],opt_buf_recv[c_vcq][edata_recv]);  
      if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);
    }
  }

  if(DEBUG_MSG) utils::logmesg(lmp," end \n");

  c_vcq = (c_vcq + 1) % VCQ_NUM;

}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Pair *pair)
{
  int i;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap, rswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv;
  uint64_t r_edata_recv;
  int n[SWAP_NUM];

  int nsize = MAX(pair->comm_reverse,pair->comm_reverse_off);

  if(DEBUG_MSG) {
    utils::logmesg(lmp,"[info] pair opt_reverse_comm {} \n ", c_vcq);
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    n[rswap] = pair->pack_reverse_comm(opt_recvnum[iswap],opt_firstrecv[iswap],opt_buf_send[c_vcq][rswap]);
  }



  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;      
    if(opt_recvnum[iswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_stadd[c_vcq][rswap], 
              rmt_comms[c_vcq][rswap].utofu_stadd, sizeof(double) * n[rswap] , edata[rswap], cbvalue[rswap], post_flags, opt_buf_send[c_vcq][rswap]);  
    }
  }
 

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[iswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
    }
  }

  if(DEBUG_MSG) {
    utils::logmesg(lmp," pair reverse begin recv ");
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_sendnum[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);
      r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;

      if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);

      pair->unpack_reverse_comm(opt_sendnum[r_edata_recv],opt_sendlist[r_edata_recv],opt_buf_recv[c_vcq][edata_recv]);   
    }
  }

  if(DEBUG_MSG) {
      utils::logmesg(lmp," end \n");
  }
  c_vcq = (c_vcq + 1) % VCQ_NUM;

}

void CommBrick::forward_comm_parral(Pair *pair, int tid)
{
  int iswap;
  double *buf;
  MPI_Request request;
  int n[SWAP_NUM];
  struct utofu_mrq_notice in_notice;

  uint64_t edata_recv;

  int nsize = pair->comm_forward;

  if(DEBUG_MSG) {
    utils::logmesg(lmp,"[info] pair forward_comm_parral tid {} c_vcq {} \n ", tid, c_vcq);
  }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    n[iswap] = pair->pack_forward_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap],opt_pbc_flag[iswap],opt_pbc[iswap]);
  }

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format(" pair forward_comm n "), n, opt_maxswap, COMM_STEP); 
  // }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    if (opt_sendnum[iswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_stadd, sizeof(double) * n[iswap] , edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
    }
  }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    if (opt_sendnum[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
    }
  }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    if (opt_sendnum[iswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
    }
  }

  // if(DEBUG_MSG) utils::logmesg(lmp," pair forward_comm ");

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    if(opt_recvnum[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);    
      pair->unpack_forward_comm(opt_recvnum[edata_recv],opt_firstrecv[edata_recv],opt_buf_recv[c_vcq][edata_recv]);  
    }
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm_parral(Pair *pair, int tid)
{
  int i;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap, rswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv;
  uint64_t r_edata_recv;
  int n[SWAP_NUM];

  int nsize = MAX(pair->comm_reverse,pair->comm_reverse_off);

  if(DEBUG_MSG) {
    utils::logmesg(lmp,"[info] pair opt_reverse_comm_parral tid {} c_vcq {} \n ", tid, c_vcq);
  }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    n[rswap] = pair->pack_reverse_comm(opt_recvnum[iswap],opt_firstrecv[iswap],opt_buf_send[c_vcq][rswap]);
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry_parral_r(lmp, fmt::format("opt_reverse_comm_parrel pair n tid {} ", tid), n, opt_maxswap, tid * 2, TNI_NUM * 2); 
  }



  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;      
    if(opt_recvnum[iswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_stadd[c_vcq][rswap], 
              rmt_comms[c_vcq][rswap].utofu_stadd, sizeof(double) * n[rswap] , edata[rswap], cbvalue[rswap], post_flags, opt_buf_send[c_vcq][rswap]);  
    }
  }
 

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }
  }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[iswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
    }
  }

  // if(DEBUG_MSG) {
  //   utils::logmesg(lmp," pair reverse begin recv \n");
  // }

  for(iswap = tid * 2 ; iswap < opt_maxswap; iswap += TNI_NUM * 2) {  
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_sendnum[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);

      if(mtx_reverse.try_lock()){
        r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;
        // if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);
        pair->unpack_reverse_comm(opt_sendnum[r_edata_recv],opt_sendlist[r_edata_recv],opt_buf_recv[c_vcq][edata_recv]);   
        remaind_iswap[edata_recv] = 0;
        mtx_reverse.unlock();
      } else {
        remaind_iswap[edata_recv] = 1;
      }
    }
  }
}

void CommBrick::forward_comm_parral_unpack(Pair *pair){
  int iswap, rswap;
  AtomVec *avec = atom->avec;

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_r(lmp, fmt::format("opt_forward_comm_parral_unpack remaind_iswap "), remaind_iswap, opt_maxswap, COMM_STEP); 
  // }

  for(iswap = 0 ; iswap < opt_maxswap; iswap += COMM_STEP) {
    if (opt_recvnum[iswap]) {
      if(remaind_iswap[iswap]){
        pair->unpack_forward_comm(opt_recvnum[iswap],opt_firstrecv[iswap],opt_buf_recv[c_vcq][iswap]);  
      }
    }
  }
}

void CommBrick::reverse_comm_parral_unpack(Pair *pair){
  int iswap, rswap;
  AtomVec *avec = atom->avec;

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm_parral_unpack pair remaind_iswap "), remaind_iswap, opt_maxswap, COMM_STEP); 
  // }

  for(iswap = 0 ; iswap < opt_maxswap; iswap += COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_sendnum[iswap]) {
      if(remaind_iswap[rswap]) {
        pair->unpack_reverse_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_recv[c_vcq][rswap]);   
      }
    }
  }
}

void CommBrick::forward_comm(Fix *fix, int size)
{
  int iswap,nsize;
  double *buf;
  MPI_Request request;
  int n[SWAP_NUM];
  struct utofu_mrq_notice in_notice;
  uint64_t edata_recv;

  if(DEBUG_MSG) utils::logmesg(lmp,"[info] fix forward begin {} \n", c_vcq);


  if (size) nsize = size;
  else nsize = fix->comm_forward;

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    n[iswap] = fix->pack_forward_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap],opt_pbc_flag[iswap],opt_pbc[iswap]);
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
     warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_stadd, sizeof(double) * n[iswap] , edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if (opt_sendnum[iswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
    }
  }

  if(DEBUG_MSG) utils::logmesg(lmp," fix forward_comm ");

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    if(opt_recvnum[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);    
      fix->unpack_forward_comm(opt_recvnum[edata_recv],opt_firstrecv[edata_recv],opt_buf_recv[c_vcq][edata_recv]);  
      if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);
    }
  }

  if(DEBUG_MSG) utils::logmesg(lmp," end \n");
  

  c_vcq = (c_vcq + 1) % VCQ_NUM;

}

/* ----------------------------------------------------------------------

   reverse communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Fix *fix, int size)
{
  int i, nsize;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap, rswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv;
  uint64_t r_edata_recv;
  int n[SWAP_NUM];

  if (size) nsize = size;
  else nsize = fix->comm_reverse;

  if(DEBUG_MSG) {
    utils::logmesg(lmp,"[info] fix opt_reverse_comm {} \n ", c_vcq);
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    n[rswap] = fix->pack_reverse_comm(opt_recvnum[rswap],opt_firstrecv[iswap],opt_buf_send[c_vcq][rswap]);
  }

  if(DEBUG_MSG) {
    utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm n "), n, opt_maxswap, COMM_STEP); 
    utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm opt_recvnum "), opt_recvnum, opt_maxswap, COMM_STEP); 
  }


  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;      
    if(opt_recvnum[rswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_stadd[c_vcq][rswap], 
              rmt_comms[c_vcq][rswap].utofu_stadd, sizeof(double) * n[rswap] , edata[rswap], cbvalue[rswap], post_flags, opt_buf_send[c_vcq][rswap]);  
    }
  }
 

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[rswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_recvnum[rswap]) { 
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
    }
  }

  if(DEBUG_MSG) {
    utils::logmesg(lmp," fix reverse begin recv ");
  }

  for (iswap = 0; iswap < opt_maxswap; iswap+=COMM_STEP) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_sendnum[rswap]) { 
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);
      r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;

      if(DEBUG_MSG) utils::logmesg(lmp," {} ", edata_recv);

      fix->unpack_reverse_comm(opt_sendnum[r_edata_recv],opt_sendlist[r_edata_recv],opt_buf_recv[c_vcq][edata_recv]);   
    }
  }

  if(DEBUG_MSG) {
      utils::logmesg(lmp," end \n");
  }
  c_vcq = (c_vcq + 1) % VCQ_NUM;

}

void CommBrick::borders_one_parrral(int tid) {
  int i,n,itype,iswap,dim,ineed, k, rswap;  
  double **x;
  std::string mesg;
  AtomVec *avec = atom->avec;
  uint64_t edata_recv, r_edata_recv;

  struct utofu_mrq_notice in_notice;
  int nbuf_len[SWAP_NUM];
  
  x       = atom->x;

  if(tid == 11) {
    for(i = 0; i < opt_maxswap; i+=comm_step) {
      opt_sendnum[i] = 0;
    }

    // if(DEBUG_MSG) utils::logmesg(lmp,"[info] begin opt_border local {}\n", atom->nlocal);   

    int ibin;
    for(i = 0; i < atom->nlocal; i++) {
      ibin = atom2bin(x[i], bin_split_line);

      for(int j = 0; j < bin2swap_ptr[ibin]; j++) {
        iswap = bin2swap[ibin][j];

        opt_sendlist[iswap][opt_sendnum[iswap]++] = i;
        // if(iswap == 0) utils::logmesg(lmp, "[info] opt_sendlist[iswap][opt_sendnum[iswap]++] {} \n", opt_sendlist[iswap][opt_sendnum[iswap]-1]);

        if (opt_sendnum[iswap] == opt_maxsendlist[iswap]) {
          error->one(FLERR, "sendlist not enough iswap {} opt_sendnum {} maxsendlist {}\n", iswap, opt_sendnum[iswap], opt_maxsendlist[iswap]);
        }
      }
      // for(iswap = 0; iswap < opt_maxswap; iswap+=comm_step) {
      //   opt_sendlist[iswap][opt_sendnum[iswap]++] = i;
      // }
    }

    if(DEBUG_MSG) {
      utils::logmesg_arry(lmp, fmt::format("[INFO] pack_border sendlist iswap {} ", 0), opt_sendlist[0], opt_sendnum[0], 1); 
      utils::logmesg_arry(lmp, fmt::format("[INFO] opt_border_one opt_sendnum "), opt_sendnum, opt_maxswap, comm_step); 
    }
  }

  lmp->parral_barrier(12, tid);

  // if(tid < 6) {
  if(tid == 11) {
    for(int tid = 0; tid < 6; tid++) {

      if(DEBUG_MSG) utils::logmesg(lmp, "[info] begin tid {} \n", tid);

      for(const auto iswap : opt_swap[tid]) {
        if (opt_sendnum[iswap]*size_border > opt_maxsend[iswap]) {
          error->one(FLERR, "[info] opt_maxsend not enough  me = {} opt_sendnum {} maxsend {}\n", 
                                                    me, opt_sendnum[iswap]*size_border, opt_maxsend[iswap]);
        }

        // if(DEBUG_MSG) utils::logmesg(lmp, "[info] pack tid {} iswap {} num {} addr {} {} {}\n", 
        //           tid, iswap, opt_sendnum[iswap], (void*)opt_buf_send[c_vcq][iswap], (void*)atom->avec->type, (void*)atom->avec->tag,(void*)atom->avec->mask,(void*)atom->avec->x[0]);
        nbuf_len[iswap] = 1 + avec->pack_border(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap]+1,opt_pbc_flag[iswap],opt_pbc[iswap]);
        opt_buf_send[c_vcq][iswap][0] = opt_sendnum[iswap];
        // if(DEBUG_MSG) utils::logmesg(lmp, "[info] finish pack tid {} iswap {} \n", tid, iswap);
      }

      if(DEBUG_MSG) {
        utils::logmesg_arry_parral(lmp, fmt::format("[INFO] opt_border_one nbuf_len tid {} ", tid), nbuf_len, opt_swap[tid]); 
      }

      for(const auto iswap : opt_swap[tid]) {
        // if(DEBUG_MSG) utils::logmesg(lmp, "[info] iswap {} opt_stadd_send_offset {} \n", iswap, opt_stadd_send_offset[c_vcq][iswap] - total_buffer_size*sizeof(double));
        // if(DEBUG_MSG) utils::logmesg(lmp, "[info] iswap {} opt_stadd_recv_offset {} \n", iswap, opt_stadd_recv_offset[c_vcq][iswap] - total_buffer_size*sizeof(double));
        warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap]+opt_stadd_send_offset[c_vcq][iswap], 
                rmt_comms[c_vcq][iswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][iswap], sizeof(double) * nbuf_len[iswap], edata[iswap], cbvalue[iswap], post_flags, (void*)opt_buf_send[c_vcq][iswap]);  
      }
      for(const auto iswap : opt_swap[tid]) {
        warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
      }

      for(const auto iswap : opt_swap[tid]) {
        utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);

        opt_recvnum[edata_recv] =  opt_buf_recv[c_vcq][edata_recv][0];
      }

      if(DEBUG_MSG) {
        utils::logmesg_arry_parral(lmp, fmt::format("[INFO] opt_border_one opt_recvnum tid {} ", tid), opt_recvnum, opt_swap[tid]); 
      }
    }
  }

  lmp->parral_barrier(12, tid);

  if(tid == 11) {
    // MPI_Barrier(world);

    c_vcq = (c_vcq + 1) % VCQ_NUM;

    if(DEBUG_MSG) {
      utils::logmesg(lmp,"[INFO]  opt_borders_one_parral_firstrecv vcq {} \n", c_vcq);
    }

    opt_buf_send[c_vcq][1][0] = opt_firstrecv[0] = atom->nlocal;
    for(iswap = comm_step; iswap < opt_maxswap; iswap += comm_step) {
      rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
      opt_buf_send[c_vcq][rswap][0] = opt_firstrecv[iswap] = opt_firstrecv[iswap-comm_step] + opt_recvnum[iswap-comm_step];
    }
  
    if(DEBUG_MSG) {
      utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_recvnum "),   opt_recvnum, opt_maxswap, comm_step); 
      utils::logmesg_arry(lmp, fmt::format("opt_border_one opt_firstrecv "), opt_firstrecv, opt_maxswap, comm_step); 
    }
    
    // if(DEBUG_MSG) utils::logmesg(lmp, "[info] borders_one_parral_xmit_pos begin cur c_vcq: {} \n", c_vcq);
  }

  lmp->parral_barrier(12, tid);

  if(tid < 6) {
    // if(DEBUG_MSG) {
    //   utils::logmesg(lmp,"[info] begin borders_one_parral_xmit_pos tid {}\n", tid);
    // }
  
    for(const auto iswap : opt_swap[tid]) {
      rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_stadd[c_vcq][rswap]+opt_stadd_send_offset[c_vcq][rswap], 
              rmt_comms[c_vcq][rswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][rswap], sizeof(double), edata[rswap], cbvalue[rswap], post_flags, (void*)opt_buf_send[c_vcq][rswap]);  
    }

    int tmp_vcq = c_vcq - 1 < 0 ? VCQ_NUM - 1 : c_vcq - 1;

    for(const auto iswap : opt_swap[tid]) {
      avec->unpack_border(opt_recvnum[iswap], opt_firstrecv[iswap], opt_buf_recv[tmp_vcq][iswap]+1);
    }

    for(const auto iswap : opt_swap[tid]) {
      rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }

    for(const auto iswap : opt_swap[tid]) {
      rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);   
      r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;
      opt_forw_pos[r_edata_recv] =  opt_buf_recv[c_vcq][edata_recv][0];
    }

    if(DEBUG_MSG) {
      utils::logmesg_arry_parral(lmp, fmt::format("opt_border_one opt_forw_pos tid {} ", tid), opt_forw_pos, opt_swap[tid]); 
    }
  }

  lmp->parral_barrier(12, tid);

  if(tid == 11) {
    c_vcq = (c_vcq + 1) % VCQ_NUM;

    if(DEBUG_MSG) {
      utils::logmesg(lmp,"[INFO]  opt_borders_one_parral_finish vcq {} comm_step {}  \n", c_vcq, comm_step);
    }

    for(int iswap = 0; iswap < opt_maxswap; iswap+=comm_step) {
      rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
      opt_size_forward_recv[iswap] = opt_recvnum[iswap]*size_forward;
      opt_size_reverse_send[rswap] = opt_recvnum[iswap]*size_reverse;
      opt_size_reverse_recv[rswap] = opt_sendnum[iswap]*size_reverse;
      opt_reverse_send_pos[rswap] = opt_firstrecv[iswap]*sizeof(double)*3;
      opt_forward_send_pos[iswap] = opt_forw_pos[iswap]*sizeof(double)*3;
      atom->nghost += opt_recvnum[iswap];
    }

    // if(DEBUG_MSG) {
    //   utils::logmesg_arry(lmp, fmt::format("borders_parral_finish opt_size_forward_recv "), opt_size_forward_recv, opt_maxswap, COMM_STEP); 
    //   utils::logmesg_arry_r(lmp, fmt::format("borders_parral_finish opt_size_reverse_send "), opt_size_reverse_send, opt_maxswap, COMM_STEP); 
    //   utils::logmesg_arry_r(lmp, fmt::format("borders_parral_finish opt_size_reverse_recv "), opt_size_reverse_recv, opt_maxswap, COMM_STEP); 
    //   utils::logmesg_arry_r(lmp, fmt::format("borders_parral_finish opt_reverse_send_pos "), opt_reverse_send_pos, opt_maxswap, COMM_STEP); 
    //   utils::logmesg_arry(lmp, fmt::format("borders_parral_finish opt_forward_send_pos "), opt_forward_send_pos, opt_maxswap, COMM_STEP); 
    // }

    // if(DEBUG_MSG) utils::logmesg(lmp," finish border nlocal {} nghost {} \n",atom->nlocal, atom->nghost);
    
    if ((atom->molecular != Atom::ATOMIC)
        && ((atom->nlocal + atom->nghost) > NEIGHMASK))
      error->one(FLERR,"Per-processor number of atoms is too large for "
                "molecular neighbor lists");
  }

  if(DEBUG_MSG) utils::logmesg(lmp,"[INFO] finish border tid  {} \n", tid);
}

void CommBrick::forward_comm_parral(int tid) {
  int nbuf_len[SWAP_NUM];

  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;
  int iswap;
  struct utofu_mrq_notice in_notice;
  int i;

  std::string mesg;
  uint64_t edata_recv;

  #ifdef THR_TIME_TEST
    FixThreadpool *fixThreadpool = dynamic_cast<FixThreadpool *>(modify->get_fix_by_id("threadpool"));
    ThrData *td = fixThreadpool->get_thr(tid);
    td->timer(Timer::START);
  #endif

  if(DEBUG_MSG) {
    utils::logmesg_arry_parral(lmp, fmt::format("opt_forward_parral opt_sendnum tid {} ", tid), opt_sendnum, opt_maxswap, tid * 2, TNI_NUM * 2); 
  }
  // if(DEBUG_MSG) {
  //   utils::logmesg_arry(lmp, fmt::format("opt_forward_comm opt_forward_send_atom "), opt_forward_send_pos, opt_maxswap, COMM_STEP); 
  // }

  for(const auto iswap : opt_swap[tid]) {
    nbuf_len[iswap] = avec->pack_comm(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_send[c_vcq][iswap],opt_pbc_flag[iswap],opt_pbc[iswap]);
    if (nbuf_len[iswap]) {
      // opt_buf_send[c_vcq][iswap][nbuf_len[iswap]] = update->ntimestep;
      // warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap], 
      //         rmt_comms[c_vcq][iswap].utofu_stadd, (nbuf_len[iswap] + 1) * sizeof(double), edata[iswap], cbvalue[iswap], 0, opt_buf_send[c_vcq][iswap]);  

      warp_utofu_put(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], rmt_comms[c_vcq][iswap].vcq_id, lcl_send_stadd[c_vcq][iswap]+opt_stadd_send_offset[c_vcq][iswap], 
              rmt_comms[c_vcq][iswap].utofu_x_stadd+opt_forward_send_pos[iswap], sizeof(double) * nbuf_len[iswap], edata[iswap], cbvalue[iswap], post_flags, opt_buf_send[c_vcq][iswap]);  
    }
  }

  for(const auto iswap : opt_swap[tid]) {
    if (nbuf_len[iswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], cbvalue_send[iswap], post_flags);
    }
  }
  for(const auto iswap : opt_swap[tid]) {
    if (nbuf_len[iswap]) {
      warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(iswap / 2) % TNI_NUM], edata_send[iswap], post_flags, in_notice);
    }
  }

  for(const auto iswap : opt_swap[tid]) {
    if(opt_size_forward_recv[iswap]) {
      utofu_recv(vcq_hdl_recv[c_vcq][(iswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice); 
      // if(DEBUG_MSG) {
      //   utils::logmesg(lmp," {}:{}:{}  ", edata_recv, opt_size_forward_recv[edata_recv]/size_forward, opt_firstrecv[edata_recv]);
      // }
      // while(opt_buf_recv[c_vcq][iswap][opt_size_forward_recv[iswap]] != update->ntimestep) ;
      // memcpy(atom->x[opt_forw_pos[iswap]], opt_buf_recv[c_vcq][iswap], opt_size_forward_recv[iswap] * sizeof(double));
    }
  }

  if(DEBUG_MSG)  utils::logmesg(lmp," opt_forward_parral finish parral tid {} \n", tid);  
  

  #ifdef THR_TIME_TEST
    td->timer(Timer::FORWARD);
  #endif
}

void CommBrick::reverse_comm_parral(int tid) {
  int n, i;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  int iswap, rswap;
  struct utofu_mrq_notice in_notice;
  std::string mesg;
  uint64_t edata_recv, r_edata_recv;

  #ifdef THR_TIME_TEST
    FixThreadpool *fixThreadpool = dynamic_cast<FixThreadpool *>(modify->get_fix_by_id("threadpool"));
    ThrData *td = fixThreadpool->get_thr(tid);
    td->timer(Timer::START);
  #endif
  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_parral_r(lmp, fmt::format("opt_reverse_comm_parrel opt_size_reverse_send tid {} ", tid), opt_size_reverse_send, opt_maxswap, tid * 2, TNI_NUM * 2); 
  //   utils::logmesg_arry_parral_r(lmp, fmt::format("opt_reverse_comm_parrel opt_size_reverse_recv tid {} ", tid), opt_size_reverse_recv, opt_maxswap, tid * 2, TNI_NUM * 2); 
  // }


  for(const auto iswap : opt_swap[tid]) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_send[rswap]) {
      warp_utofu_put(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], rmt_comms[c_vcq][rswap].vcq_id, lcl_send_f_stadd[c_vcq][rswap]+opt_reverse_send_pos[rswap],
                  rmt_comms[c_vcq][rswap].utofu_stadd+opt_stadd_recv_offset[c_vcq][rswap], sizeof(double) * opt_size_reverse_send[rswap], 
                  edata[rswap], cbvalue[rswap], post_flags,atom->f[opt_firstrecv[iswap]]);
    }
  }

  for(const auto iswap : opt_swap[tid]) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_send[rswap]) {
      warp_utofu_poll_tcq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], cbvalue_send[rswap], post_flags);
    }
  }

  // for(const auto iswap : opt_swap[tid]) {
  //   rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
  //   if (opt_size_reverse_send[rswap]) {
  //     warp_utofu_poll_mrq(vcq_hdl_send[c_vcq][(rswap / 2) % TNI_NUM], edata_send[rswap], post_flags, in_notice);
  //   }
  // }


  for(const auto iswap : opt_swap[tid]) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_recv[rswap]) { 
      utofu_recv(vcq_hdl_recv[c_vcq][(rswap / 2) % TNI_NUM], edata_recv, post_flags, in_notice);

      r_edata_recv = edata_recv % 2 == 0 ? edata_recv + 1 : edata_recv - 1;

      if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] recv iswap {} sendproc {} i {} ", 
                edata_recv, opt_sendproc[r_edata_recv], opt_sendlist[r_edata_recv][0]), 
                opt_buf_recv[c_vcq][edata_recv], (opt_sendnum[r_edata_recv]) * 3, 1);

      if(mtx_reverse.try_lock()){
        avec->unpack_reverse(opt_sendnum[r_edata_recv],opt_sendlist[r_edata_recv],opt_buf_recv[c_vcq][edata_recv]);
        remaind_iswap[edata_recv] = 0;
        mtx_reverse.unlock();
      } else {
        remaind_iswap[edata_recv] = 1;
      }
    }
  }

  // if(tid == 0) {
  //   if(DEBUG_MSG) {
  //     auto mesg = fmt::format("[NUMA] reverse_comm_parral ghost force :  \n"); 
  //     for(iswap = 0 ; iswap < opt_maxswap; iswap += COMM_STEP) {
  //       mesg += fmt::format("  iswap {}: ", iswap); 
  //       for(int i = opt_firstrecv[iswap]; i < opt_firstrecv[iswap] + opt_recvnum[iswap]; i++) {
  //         mesg += fmt::format(" {:<3.3g}:{:<3.3g}:{:<3.3g} |", f[i][0], f[i][1],f[i][2]); 
  //       }
  //       mesg += "\n";
  //     }
  //     utils::logmesg(lmp,"{}\n", mesg);
  //   }
  // }

  if(DEBUG_MSG) {
    utils::logmesg(lmp,"[info] finish opt_reverse_comm_parrel send tid {}\n", tid);
  }

  #ifdef THR_TIME_TEST
    td->timer(Timer::REVERSE);
  #endif
}

void CommBrick::reverse_comm_parral_unpack(){
  int iswap, rswap;
  AtomVec *avec = atom->avec;

  // if(DEBUG_MSG) {
  //   utils::logmesg_arry_r(lmp, fmt::format("opt_reverse_comm_parral_unpack remaind_iswap "), remaind_iswap, opt_maxswap, COMM_STEP); 
  // }

  for(iswap = 0 ; iswap < opt_maxswap; iswap += comm_step) {
    rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;
    if (opt_size_reverse_recv[rswap]) {
      if(remaind_iswap[rswap]){
        avec->unpack_reverse(opt_sendnum[iswap],opt_sendlist[iswap],opt_buf_recv[c_vcq][rswap]);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

#else
void CommBrick::forward_comm(int /*dummy*/)
{
  int n;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **x = atom->x;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_x_only set, exchange or copy directly to x, don't unpack

  for (int iswap = 0; iswap < nswap; iswap++) {
    if (sendproc[iswap] != me) {
      if (comm_x_only) {
        if (size_forward_recv[iswap]) {
          buf = x[firstrecv[iswap]];
          MPI_Irecv(buf,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
        }
        n = avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      } else if (ghost_velocity) {
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
        n = avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
        avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_recv);
      } else {
        if (size_forward_recv[iswap])
          MPI_Irecv(buf_recv,size_forward_recv[iswap],MPI_DOUBLE,
                    recvproc[iswap],0,world,&request);
        n = avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (size_forward_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
        avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_recv);
      }

    } else {
      if (comm_x_only) {
        if (sendnum[iswap])
          avec->pack_comm(sendnum[iswap],sendlist[iswap],
                          x[firstrecv[iswap]],pbc_flag[iswap],pbc[iswap]);
      } else if (ghost_velocity) {
        avec->pack_comm_vel(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        avec->unpack_comm_vel(recvnum[iswap],firstrecv[iswap],buf_send);
      } else {
        avec->pack_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
        avec->unpack_comm(recvnum[iswap],firstrecv[iswap],buf_send);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommBrick::reverse_comm()
{
  int n;
  MPI_Request request;
  AtomVec *avec = atom->avec;
  double **f = atom->f;
  double *buf;

  // exchange data with another proc
  // if other proc is self, just copy
  // if comm_f_only set, exchange or copy directly from f, don't pack

  for (int iswap = nswap-1; iswap >= 0; iswap--) {
    if (sendproc[iswap] != me) {
      if (comm_f_only) {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        if (size_reverse_send[iswap]) {
          buf = f[firstrecv[iswap]];
          MPI_Send(buf,size_reverse_send[iswap],MPI_DOUBLE,recvproc[iswap],0,world);
        }
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      } else {
        if (size_reverse_recv[iswap])
          MPI_Irecv(buf_recv,size_reverse_recv[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
        n = avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
        if (size_reverse_recv[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      }
      avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_recv);

    } else {
      if (comm_f_only) {
        if (sendnum[iswap])
          avec->unpack_reverse(sendnum[iswap],sendlist[iswap],f[firstrecv[iswap]]);
      } else {
        avec->pack_reverse(recvnum[iswap],firstrecv[iswap],buf_send);
        avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_send);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
   atoms exchanged with all 6 stencil neighbors
   send out atoms that have left my box, receive ones entering my box
   atoms will be lost if not inside a stencil proc's box
     can happen if atom moves outside of non-periodic boundary
     or if atom moves more than one proc away
   this routine called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before exchange is called
------------------------------------------------------------------------- */

void CommBrick::exchange()
{
  int i,m,nsend,nrecv,nrecv1,nrecv2,nlocal;
  double lo,hi,value;
  double **x;
  double *sublo,*subhi;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()
  // clear ghost count and any ghost bonus data internal to AtomVec

  if (map_style != Atom::MAP_NONE) atom->map_clear();
  atom->nghost = 0;
  atom->avec->clear_bonus();

  // insure send buf has extra space for a single atom
  // only need to reset if a fix can dynamically add to size of single atom

  if (maxexchange_fix_dynamic) {
    int bufextra_old = bufextra;
    init_exchange();
    if (bufextra > bufextra_old) grow_send(maxsend+bufextra,2);
  }

  // subbox bounds for orthogonal or triclinic

  if (triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  // loop over dimensions

  int dimension = domain->dimension;

  for (int dim = 0; dim < dimension; dim++) {

    // fill buffer with atoms leaving my box, using < and >=
    // when atom is deleted, fill it in with last atom

    x = atom->x;
    lo = sublo[dim];
    hi = subhi[dim];
    nlocal = atom->nlocal;
    i = nsend = 0;

    while (i < nlocal) {
      if (x[i][dim] < lo || x[i][dim] >= hi) {
        if (nsend > maxsend) grow_send(nsend,1);
        nsend += avec->pack_exchange(i,&buf_send[nsend]);
        avec->copy(nlocal-1,i,1);
        nlocal--;
      } else i++;
    }
    atom->nlocal = nlocal;

    // send/recv atoms in both directions
    // send size of message first so receiver can realloc buf_recv if needed
    // if 1 proc in dimension, no send/recv
    //   set nrecv = 0 so buf_send atoms will be lost
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    if (procgrid[dim] == 1) nrecv = 0;
    else {
      MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][0],0,
                   &nrecv1,1,MPI_INT,procneigh[dim][1],0,world,MPI_STATUS_IGNORE);
      nrecv = nrecv1;
      if (procgrid[dim] > 2) {
        MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][1],0,
                     &nrecv2,1,MPI_INT,procneigh[dim][0],0,world,MPI_STATUS_IGNORE);
        nrecv += nrecv2;
      }
      if (nrecv > maxrecv) grow_recv(nrecv);

      MPI_Irecv(buf_recv,nrecv1,MPI_DOUBLE,procneigh[dim][1],0,world,&request);
      MPI_Send(buf_send,nsend,MPI_DOUBLE,procneigh[dim][0],0,world);
      MPI_Wait(&request,MPI_STATUS_IGNORE);

      if (procgrid[dim] > 2) {
        MPI_Irecv(&buf_recv[nrecv1],nrecv2,MPI_DOUBLE,procneigh[dim][0],0,world,&request);
        MPI_Send(buf_send,nsend,MPI_DOUBLE,procneigh[dim][1],0,world);
        MPI_Wait(&request,MPI_STATUS_IGNORE);
      }
    }

    // check incoming atoms to see if they are in my box
    // if so, add to my list
    // box check is only for this dimension,
    //   atom may be passed to another proc in later dims

    m = 0;
    while (m < nrecv) {
      value = buf_recv[m+dim+1];
      if (value >= lo && value < hi) m += avec->unpack_exchange(&buf_recv[m]);
      else m += static_cast<int> (buf_recv[m]);
    }
  }

  if (atom->firstgroupname) atom->first_reorder();
}

/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a forward_comm(), so don't need to explicitly
     call forward_comm() on reneighboring timestep
   this routine is called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before borders is called
------------------------------------------------------------------------- */

void CommBrick::borders()
{
  int i,n,itype,icollection,iswap,dim,ineed,twoneed;
  int nsend,nrecv,sendflag,nfirst,nlast,ngroup,nprior;
  double lo,hi;
  int *type;
  int *collection;
  double **x;
  double *buf,*mlo,*mhi;
  MPI_Request request;
  AtomVec *avec = atom->avec;

  // After exchanging/sorting, need to reconstruct collection array for border communication
  if (mode == Comm::MULTI) neighbor->build_collection(0);

  // do swaps over all 3 dimensions

  iswap = 0;
  smax = rmax = 0;

  for (dim = 0; dim < 3; dim++) {
    nlast = 0;
    twoneed = 2*maxneed[dim];
    for (ineed = 0; ineed < twoneed; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in sendlist for use in future timesteps

      x = atom->x;
      if (mode == Comm::SINGLE) {
        lo = slablo[iswap];
        hi = slabhi[iswap];
      } else if (mode == Comm::MULTI) {
        collection = neighbor->collection;
        mlo = multilo[iswap];
        mhi = multihi[iswap];
      } else {
        type = atom->type;
        mlo = multioldlo[iswap];
        mhi = multioldhi[iswap];
      }
      if (ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom->nlocal + atom->nghost;
      }

      nsend = 0;

      // sendflag = 0 if I do not send on this swap
      // sendneed test indicates receiver no longer requires data
      // e.g. due to non-PBC or non-uniform sub-domains

      if (ineed/2 >= sendneed[dim][ineed % 2]) sendflag = 0;
      else sendflag = 1;

      // find send atoms according to SINGLE vs MULTI
      // all atoms eligible versus only atoms in bordergroup
      // can only limit loop to bordergroup for first sends (ineed < 2)
      // on these sends, break loop in two: owned (in group) and ghost

      if (sendflag) {
        if (!bordergroup || ineed >= 2) {
          if (mode == Comm::SINGLE) {
            for (i = nfirst; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            for (i = nfirst; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            for (i = nfirst; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }

        } else {
          if (mode == Comm::SINGLE) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            for (i = atom->nlocal; i < nlast; i++)
              if (x[i][dim] >= lo && x[i][dim] <= hi) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
          } else if (mode == Comm::MULTI) {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              icollection = collection[i];
              if (x[i][dim] >= mlo[icollection] && x[i][dim] <= mhi[icollection]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          } else {
            ngroup = atom->nfirst;
            for (i = 0; i < ngroup; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
            for (i = atom->nlocal; i < nlast; i++) {
              itype = type[i];
              if (x[i][dim] >= mlo[itype] && x[i][dim] <= mhi[itype]) {
                if (nsend == maxsendlist[iswap]) grow_list(iswap,nsend);
                sendlist[iswap][nsend++] = i;
              }
            }
          }
        }
      }

      // pack up list of border atoms

      if (nsend*size_border > maxsend) grow_send(nsend*size_border,0);
      if (ghost_velocity)
        n = avec->pack_border_vel(nsend,sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);
      else
        n = avec->pack_border(nsend,sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

      // swap atoms with other proc
      // no MPI calls except SendRecv if nsend/nrecv = 0
      // put incoming ghosts at end of my atom arrays
      // if swapping with self, simply copy, no messages

      if (sendproc[iswap] != me) {
        MPI_Sendrecv(&nsend,1,MPI_INT,sendproc[iswap],0,
                     &nrecv,1,MPI_INT,recvproc[iswap],0,world,MPI_STATUS_IGNORE);
        if (nrecv*size_border > maxrecv) grow_recv(nrecv*size_border);
        if (nrecv) MPI_Irecv(buf_recv,nrecv*size_border,MPI_DOUBLE,
                             recvproc[iswap],0,world,&request);
        if (n) MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
        if (nrecv) MPI_Wait(&request,MPI_STATUS_IGNORE);
        buf = buf_recv;
      } else {
        nrecv = nsend;
        buf = buf_send;
      }

      // unpack buffer

      if (ghost_velocity)
        avec->unpack_border_vel(nrecv,atom->nlocal+atom->nghost,buf);
      else
        avec->unpack_border(nrecv,atom->nlocal+atom->nghost,buf);

      // set all pointers & counters

      smax = MAX(smax,nsend);
      rmax = MAX(rmax,nrecv);
      sendnum[iswap] = nsend;
      recvnum[iswap] = nrecv;
      size_forward_recv[iswap] = nrecv*size_forward;
      size_reverse_send[iswap] = nrecv*size_reverse;
      size_reverse_recv[iswap] = nsend*size_reverse;
      firstrecv[iswap] = atom->nlocal + atom->nghost;
      nprior = atom->nlocal + atom->nghost;
      atom->nghost += nrecv;
      if (neighbor->style == Neighbor::MULTI) neighbor->build_collection(nprior);

      iswap++;
    }
  }

  // For molecular systems we lose some bits for local atom indices due
  // to encoding of special pairs in neighbor lists. Check for overflows.

  if ((atom->molecular != Atom::ATOMIC)
      && ((atom->nlocal + atom->nghost) > NEIGHMASK))
    error->one(FLERR,"Per-processor number of atoms is too large for "
               "molecular neighbor lists");

  // insure send/recv buffers are long enough for all forward & reverse comm

  int max = MAX(maxforward*smax,maxreverse*rmax);
  if (max > maxsend) grow_send(max,0);
  max = MAX(maxforward*rmax,maxreverse*smax);
  if (max > maxrecv) grow_recv(max);

  // reset global->local map

  if (map_style != Atom::MAP_NONE) atom->map_set();
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Fix
   size/nsize used only to set recv buffer limit
   size = 0 (default) -> use comm_forward from Fix
   size > 0 -> Fix passes max size per atom
   the latter is only useful if Fix does several comm modes,
     some are smaller than max stored in its comm_forward
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Fix *fix, int size)
{
  int iswap,n,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = fix->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

void CommBrick::reverse_comm(Fix *fix, int size)
{
  int iswap,n,nsize;
  double *buf;
  MPI_Request request;

  if (size) nsize = size;
  else nsize = fix->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Pair *pair)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = pair->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = pair->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    pair->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Pair
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Pair *pair)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(pair->comm_reverse,pair->comm_reverse_off);

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = pair->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    pair->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}
#endif

/* ----------------------------------------------------------------------
   forward communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Bond *bond)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = bond->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = bond->pack_forward_comm(sendnum[iswap],sendlist[iswap],buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    bond->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Bond
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Bond *bond)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = MAX(bond->comm_reverse,bond->comm_reverse_off);

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = bond->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    bond->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Fix with variable size data
   query fix for pack size to insure buf_send is big enough
   handshake sizes before each Irecv/Send to insure buf_recv is big enough
------------------------------------------------------------------------- */

void CommBrick::reverse_comm_variable(Fix *fix)
{
  int iswap,nsend,nrecv;
  double *buf;
  MPI_Request request;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    nsend = fix->pack_reverse_comm_size(recvnum[iswap],firstrecv[iswap]);
    if (nsend > maxsend) grow_send(nsend,0);
    nsend = fix->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      MPI_Sendrecv(&nsend,1,MPI_INT,recvproc[iswap],0,
                   &nrecv,1,MPI_INT,sendproc[iswap],0,world,MPI_STATUS_IGNORE);

      if (sendnum[iswap]) {
        if (nrecv > maxrecv) grow_recv(nrecv);
        MPI_Irecv(buf_recv,maxrecv,MPI_DOUBLE,sendproc[iswap],0,world,&request);
      }
      if (recvnum[iswap])
        MPI_Send(buf_send,nsend,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    fix->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Compute *compute)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = compute->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                   buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    compute->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Compute
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Compute *compute)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = compute->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = compute->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    compute->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::forward_comm(Dump *dump)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_forward;

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    n = dump->pack_forward_comm(sendnum[iswap],sendlist[iswap],
                                buf_send,pbc_flag[iswap],pbc[iswap]);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    dump->unpack_forward_comm(recvnum[iswap],firstrecv[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   reverse communication invoked by a Dump
   nsize used only to set recv buffer limit
------------------------------------------------------------------------- */

void CommBrick::reverse_comm(Dump *dump)
{
  int iswap,n;
  double *buf;
  MPI_Request request;

  int nsize = dump->comm_reverse;

  for (iswap = nswap-1; iswap >= 0; iswap--) {

    // pack buffer

    n = dump->pack_reverse_comm(recvnum[iswap],firstrecv[iswap],buf_send);

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (sendnum[iswap])
        MPI_Irecv(buf_recv,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world,&request);
      if (recvnum[iswap])
        MPI_Send(buf_send,n,MPI_DOUBLE,recvproc[iswap],0,world);
      if (sendnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    dump->unpack_reverse_comm(sendnum[iswap],sendlist[iswap],buf);
  }
}

/* ----------------------------------------------------------------------
   forward communication of N values in per-atom array
------------------------------------------------------------------------- */

void CommBrick::forward_comm_array(int nsize, double **array)
{
  int i,j,k,m,iswap,last;
  double *buf;
  MPI_Request request;

  // insure send/recv bufs are big enough for nsize
  // based on smax/rmax from most recent borders() invocation

  if (nsize > maxforward) {
    maxforward = nsize;
    if (maxforward*smax > maxsend) grow_send(maxforward*smax,0);
    if (maxforward*rmax > maxrecv) grow_recv(maxforward*rmax);
  }

  for (iswap = 0; iswap < nswap; iswap++) {

    // pack buffer

    m = 0;
    for (i = 0; i < sendnum[iswap]; i++) {
      j = sendlist[iswap][i];
      for (k = 0; k < nsize; k++)
        buf_send[m++] = array[j][k];
    }

    // exchange with another proc
    // if self, set recv buffer to send buffer

    if (sendproc[iswap] != me) {
      if (recvnum[iswap])
        MPI_Irecv(buf_recv,nsize*recvnum[iswap],MPI_DOUBLE,recvproc[iswap],0,world,&request);
      if (sendnum[iswap])
        MPI_Send(buf_send,nsize*sendnum[iswap],MPI_DOUBLE,sendproc[iswap],0,world);
      if (recvnum[iswap]) MPI_Wait(&request,MPI_STATUS_IGNORE);
      buf = buf_recv;
    } else buf = buf_send;

    // unpack buffer

    m = 0;
    last = firstrecv[iswap] + recvnum[iswap];
    for (i = firstrecv[iswap]; i < last; i++)
      for (k = 0; k < nsize; k++)
        array[i][k] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   exchange info provided with all 6 stencil neighbors
------------------------------------------------------------------------- */

int CommBrick::exchange_variable(int n, double *inbuf, double *&outbuf)
{
  int nsend,nrecv,nrecv1,nrecv2;
  MPI_Request request;

  nrecv = n;
  if (nrecv > maxrecv) grow_recv(nrecv);
  memcpy(buf_recv,inbuf,nrecv*sizeof(double));

  // loop over dimensions

  for (int dim = 0; dim < 3; dim++) {

    // no exchange if only one proc in a dimension

    if (procgrid[dim] == 1) continue;

    // send/recv info in both directions using same buf_recv
    // if 2 procs in dimension, single send/recv
    // if more than 2 procs in dimension, send/recv to both neighbors

    nsend = nrecv;
    MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][0],0,
                 &nrecv1,1,MPI_INT,procneigh[dim][1],0,world,MPI_STATUS_IGNORE);
    nrecv += nrecv1;
    if (procgrid[dim] > 2) {
      MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][1],0,
                   &nrecv2,1,MPI_INT,procneigh[dim][0],0,world,MPI_STATUS_IGNORE);
      nrecv += nrecv2;
    } else nrecv2 = 0;

    if (nrecv > maxrecv) grow_recv(nrecv);

    MPI_Irecv(&buf_recv[nsend],nrecv1,MPI_DOUBLE,procneigh[dim][1],0,world,&request);
    MPI_Send(buf_recv,nsend,MPI_DOUBLE,procneigh[dim][0],0,world);
    MPI_Wait(&request,MPI_STATUS_IGNORE);

    if (procgrid[dim] > 2) {
      MPI_Irecv(&buf_recv[nsend+nrecv1],nrecv2,MPI_DOUBLE,procneigh[dim][0],0,world,&request);
      MPI_Send(buf_recv,nsend,MPI_DOUBLE,procneigh[dim][1],0,world);
      MPI_Wait(&request,MPI_STATUS_IGNORE);
    }
  }

  outbuf = buf_recv;
  return nrecv;
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR and bufextra
   flag = 0, don't need to realloc with copy, just free/malloc w/ BUFFACTOR
   flag = 1, realloc with BUFFACTOR
   flag = 2, free/malloc w/out BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_send(int n, int flag)
{
  if (flag == 0) {
    maxsend = static_cast<int> (BUFFACTOR * n);
    memory->destroy(buf_send);
    memory->create(buf_send,maxsend+bufextra,"comm:buf_send");
  } else if (flag == 1) {
    maxsend = static_cast<int> (BUFFACTOR * n);
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");
  } else {
    memory->destroy(buf_send);
    memory->grow(buf_send,maxsend+bufextra,"comm:buf_send");
  }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_recv(int n)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  memory->destroy(buf_recv);
  memory->create(buf_recv,maxrecv,"comm:buf_recv");
}

/* ----------------------------------------------------------------------
   realloc the size of the iswap sendlist as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrick::grow_list(int iswap, int n)
{
  maxsendlist[iswap] = static_cast<int> (BUFFACTOR * n);
  memory->grow(sendlist[iswap],maxsendlist[iswap],"comm:sendlist[iswap]");
}

/* ----------------------------------------------------------------------
   realloc the buffers needed for swaps
------------------------------------------------------------------------- */

void CommBrick::grow_swap(int n)
{
  free_swap();
  allocate_swap(n);
  if (mode == Comm::MULTI) {
    free_multi();
    allocate_multi(n);
  }

  if (mode == Comm::MULTIOLD) {
    free_multiold();
    allocate_multiold(n);
  }


  sendlist = (int **)
    memory->srealloc(sendlist,n*sizeof(int *),"comm:sendlist");
  memory->grow(maxsendlist,n,"comm:maxsendlist");
  for (int i = maxswap; i < n; i++) {
    maxsendlist[i] = BUFMIN;
    memory->create(sendlist[i],BUFMIN,"comm:sendlist[i]");
  }
  maxswap = n;
}

/* ----------------------------------------------------------------------
   allocation of swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_swap(int n)
{
  memory->create(sendnum,n,"comm:sendnum");
  memory->create(recvnum,n,"comm:recvnum");
  memory->create(sendproc,n,"comm:sendproc");
  memory->create(recvproc,n,"comm:recvproc");
  memory->create(size_forward_recv,n,"comm:size");
  memory->create(size_reverse_send,n,"comm:size");
  memory->create(size_reverse_recv,n,"comm:size");
  memory->create(slablo,n,"comm:slablo");
  memory->create(slabhi,n,"comm:slabhi");
  memory->create(firstrecv,n,"comm:firstrecv");
  memory->create(pbc_flag,n,"comm:pbc_flag");
  memory->create(pbc,n,6,"comm:pbc");
}

/* ----------------------------------------------------------------------
   allocation of multi-collection swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multi(int n)
{
  multilo = memory->create(multilo,n,ncollections,"comm:multilo");
  multihi = memory->create(multihi,n,ncollections,"comm:multihi");
}

/* ----------------------------------------------------------------------
   allocation of multi/old-type swap info
------------------------------------------------------------------------- */

void CommBrick::allocate_multiold(int n)
{
  multioldlo = memory->create(multioldlo,n,atom->ntypes+1,"comm:multioldlo");
  multioldhi = memory->create(multioldhi,n,atom->ntypes+1,"comm:multioldhi");
}


/* ----------------------------------------------------------------------
   free memory for swaps
------------------------------------------------------------------------- */

void CommBrick::free_swap()
{
  memory->destroy(sendnum);
  memory->destroy(recvnum);
  memory->destroy(sendproc);
  memory->destroy(recvproc);
  memory->destroy(size_forward_recv);
  memory->destroy(size_reverse_send);
  memory->destroy(size_reverse_recv);
  memory->destroy(slablo);
  memory->destroy(slabhi);
  memory->destroy(firstrecv);
  memory->destroy(pbc_flag);
  memory->destroy(pbc);
}

/* ----------------------------------------------------------------------
   free memory for multi-collection swaps
------------------------------------------------------------------------- */

void CommBrick::free_multi()
{
  memory->destroy(multilo);
  memory->destroy(multihi);
  multilo = multihi = nullptr;
}

/* ----------------------------------------------------------------------
   free memory for multi/old-type swaps
------------------------------------------------------------------------- */

void CommBrick::free_multiold()
{
  memory->destroy(multioldlo);
  memory->destroy(multioldhi);
  multioldlo = multioldhi = nullptr;
}

/* ----------------------------------------------------------------------
   extract data potentially useful to other classes
------------------------------------------------------------------------- */

void *CommBrick::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"localsendlist") == 0) {
    int i, iswap, isend;
    dim = 1;
    if (!localsendlist)
      memory->create(localsendlist,atom->nlocal,"comm:localsendlist");
    else
      memory->grow(localsendlist,atom->nlocal,"comm:localsendlist");

    for (i = 0; i < atom->nlocal; i++)
      localsendlist[i] = 0;

    for (iswap = 0; iswap < nswap; iswap++)
      for (isend = 0; isend < sendnum[iswap]; isend++)
        if (sendlist[iswap][isend] < atom->nlocal)
          localsendlist[sendlist[iswap][isend]] = 1;

    return (void *) localsendlist;
  }

  return nullptr;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

double CommBrick::memory_usage()
{
  double bytes = 0;
  bytes += (double)nprocs * sizeof(int);    // grid2proc
  for (int i = 0; i < nswap; i++)
    bytes += memory->usage(sendlist[i],maxsendlist[i]);
  bytes += memory->usage(buf_send,maxsend+bufextra);
  bytes += memory->usage(buf_recv,maxrecv);
  return bytes;
}


int CommBrick::updown(int dim, int dir, int loc, double prd, int periodicity, double *split)
{
  int index,count;
  double frac,delta;

  if (dir == 0) {
    frac = cutghost[dim]/prd;
    index = loc - 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index < 0) {
        if (!periodicity) break;
        index = procgrid[dim] - 1;
      }
      count++;
      delta += split[index+1] - split[index];
      index--;
    }

  } else {
    frac = cutghost[dim]/prd;
    index = loc + 1;
    delta = 0.0;
    count = 0;
    while (delta < frac) {
      if (index >= procgrid[dim]) {
        if (!periodicity) break;
        index = 0;
      }
      count++;
      delta += split[index+1] - split[index];
      index++;
    }
  }

  return count;
}