/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_COMM_BRICK_H
#define LMP_COMM_BRICK_H

#include "comm.h"
#include <bitset>

#define BIT_SET_DEEPTH 10000
#define VFILE_0          "/dev/zero"
#define VFILE_1          "./"

#define NUMA_ALL
#define BORDER_ELEMENT 4

namespace LAMMPS_NS {

// struct Lcl_recv_param{
  
// }


struct Swap_pair{
  int numa_id;
  int nu_swap;
  int ptr;
  int offset_atom;
};

class Sh_swap_struct {
public:
  int enable = 1;
  int is_lcl = 0;
  int rmt_node_id = -1;

  int pkt_offset[NUMA_NUM] = {0, -1,-1,-1};
  int atom_offset[NUMA_NUM] = {0, -1,-1,-1};
  int pkt_size[NUMA_NUM]   = {-1,-1,-1,-1};
  int atom_num[NUMA_NUM]   = {0,0,0,0};
  // int lcl_rank_need[NUMA_NUM];
  int lcl_ptr[NUMA_NUM][NUMA_NUM] = {{-1,-1,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}};
  int lcl_offset[NUMA_NUM][NUMA_NUM] = {{-1,-1,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1}};
  int lcl_pbc_flag[NUMA_NUM][NUMA_NUM] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
  int lcl_pbc[NUMA_NUM][NUMA_NUM][3];

};

class CommBrick : public Comm {
 public:
  CommBrick(class LAMMPS *);
  CommBrick(class LAMMPS *, class Comm *);

  ~CommBrick() override;

  void init() override;
  void setup() override;                        // setup 3d comm pattern
  void setup_opt();                        // setup 3d comm pattern
  void setup_numa();                        // setup 3d comm pattern
  void forward_comm(int dummy = 0) override;    // forward comm of atom coords
  void reverse_comm() override;                 // reverse comm of forces
  void exchange() override;                     // move atoms to new procs
  void borders() override;                      // setup list of atoms to comm

  void numa_shm_init() override;



  void borders_one_parrral(int tid) override;                      // setup list of atoms to comm

  void forward_comm_parral(int tid) override;                 // forward comm from a Pair
  void reverse_comm_parral(int tid) override;                 // reverse comm from a Pair
  void reverse_comm_parral_unpack() override;                 // reverse comm of forces

  void borders_one_parrral_numa(int tid) override;                      // setup list of atoms to comm
  void forward_comm_parral_numa(int tid) override;                 // forward comm from a Pair
  void reverse_comm_parral_numa(int tid) override;                 // reverse comm from a Pair
  void reverse_comm_parral_unpack_numa() override;                 // reverse comm of forces
  

  void utofu_init_opt() override;
  void utofu_init_numa() override;

  double box_distance(const int*, double*);

  void forward_comm(class Pair *) override;                 // forward comm from a Pair
  void reverse_comm(class Pair *) override;                 // reverse comm from a Pair
  void forward_comm_parral(class Pair *, int tid) override;                 // forward comm from a Pair
  void reverse_comm_parral(class Pair *, int tid) override;                 // reverse comm from a Pair
  void forward_comm_parral_unpack(class Pair *) override;                 // forward comm from a Pair
  void reverse_comm_parral_unpack(class Pair *) override;                 // reverse comm from a Pair

  void forward_comm(class Bond *) override;                 // forward comm from a Bond
  void reverse_comm(class Bond *) override;                 // reverse comm from a Bond
  void forward_comm(class Fix *, int size = 0) override;    // forward comm from a Fix
  void reverse_comm(class Fix *, int size = 0) override;    // reverse comm from a Fix
  void reverse_comm_variable(class Fix *) override;         // variable size reverse comm from a Fix
  void forward_comm(class Compute *) override;              // forward from a Compute
  void reverse_comm(class Compute *) override;              // reverse from a Compute
  void forward_comm(class Dump *) override;                 // forward comm from a Dump
  void reverse_comm(class Dump *) override;                 // reverse comm from a Dump

  void forward_comm_array(int, double **) override;            // forward comm of array
  int exchange_variable(int, double *, double *&) override;    // exchange on neigh stencil
  void *extract(const char *, int &) override;
  double memory_usage() override;

 protected:
  int nswap;                            // # of swaps to perform = sum of maxneed
  int recvneed[3][2];                   // # of procs away I recv atoms from
  int sendneed[3][2];                   // # of procs away I send atoms to
  int maxneed[3];                       // max procs away any proc needs, per dim
  int maxswap;                          // max # of swaps memory is allocated for
  int *sendnum, *recvnum;               // # of atoms to send/recv in each swap
  int *sendproc, *recvproc;             // proc to send/recv to/from at each swap
  int *size_forward_recv;               // # of values to recv in each forward comm
  int *size_reverse_send;               // # to send in each reverse comm
  int *size_reverse_recv;               // # to recv in each reverse comm
  double *slablo, *slabhi;              // bounds of slab to send at each swap
  double **multilo, **multihi;          // bounds of slabs for multi-collection swap
  double **multioldlo, **multioldhi;    // bounds of slabs for multi-type swap
  double **cutghostmulti;               // cutghost on a per-collection basis
  double **cutghostmultiold;            // cutghost on a per-type basis
  int *pbc_flag;                        // general flag for sending atoms thru PBC
  int **pbc;                            // dimension flags for PBC adjustments

  int *firstrecv;        // where to put 1st recv atom in each swap
  int **sendlist;        // list of atoms to send in each swap
  int *localsendlist;    // indexed list of local sendlist atoms
  int *maxsendlist;      // max size of send list for each swap

  double *buf_send;        // send buffer for all comm
  double *buf_recv;        // recv buffer for all comm
  int maxsend, maxrecv;    // current size of send/recv buffer
  int smax, rmax;          // max size in atoms of single borders send/recv

  std::mutex mtx_reverse;

  bool first_init_flag;

  int comm_step;
  std::vector<int> opt_swap[COMM_TNUM];

  std::vector<int> numa_swap;

  std::vector<int> thr_swap[T_THREAD], thr_swap_full[T_THREAD];
  std::vector<int> numa_swap_full;
  std::vector<int> numa_swap_all[NUMA_NUM], numa_swap_full_all[NUMA_NUM];



  int bin2swap[27][SWAP_NUM];
  int bin2swap_ptr[27];

  int nubin2swap[27][SWAP_NUM];
  int nubin2swap_ptr[27];
  
  double bin_split_line[3][2]; 
  double nubin_split_line[3][2]; 


  int opt_maxdirct;
  int opt_maxswap, numa_maxswap;                          // max # of swaps memory is allocated for

  uint64_t *opt_sendnum, *opt_recvnum;               // # of atoms to send/recv in each swap 每个方向要发送的原子数量
  uint64_t *numa_sendnum, *numa_recvnum;               // # of atoms to send/recv in each swap 每个方向要发送的原子数量

  uint64_t *opt_forw_pos;
  int **opt_sendlist, **numa_sendlist;        // list of atoms to send in each swap. 二维数组，第一维是方向，第二维是该方向需要传输的原子的下标

  int *remaind_iswap;
  uint64_t *opt_firstrecv;
  int *opt_sendproc, *opt_recvproc;             // proc to send/recv to/from at each swap
  int *numa_sendproc, *numa_recvproc;             // proc to send/recv to/from at each swap
  int *opt_size_forward_recv;               // # of values to recv in each forward comm
  int opt_size_forward_recv_bcast[27][27];               // # of values to recv in each forward comm
  int *opt_size_reverse_send;               // # to send in each reverse comm
  int *opt_size_reverse_recv;               // # to recv in each reverse comm
  
  int *opt_reverse_send_pos;               // # to send in each reverse comm
  int *opt_forward_send_pos;               // # to send in each reverse comm
  double **opt_slablo, **opt_slabhi, **numa_slablo, **numa_slabhi;              // bounds of slab to send at each swap
  int *opt_pbc_flag, *numa_pbc_flag;                        // general flag for sending atoms thru PBC
  int **opt_pbc, **numa_pbc;                            // dimension flags for PBC 
  int *opt_pbc_flag_recv, **opt_pbc_recv;
  int *numa_pbc_flag_recv, **numa_pbc_recv;
  // int *opt_recv_pbc_flag, **opt_recv_pbc;
  double **opt_buf_send[VCQ_NUM];        // send buffer for all comm
  double **opt_buf_recv[VCQ_NUM];        // recv buffer for all comm  
  int *opt_stadd_send_offset[VCQ_NUM];        // send buffer for all comm
  int *opt_stadd_recv_offset[VCQ_NUM];        // recv buffer for all comm  
  double *all_recv_buffer;
  uint64_t total_buffer_size = 0; 
  uint64_t total_force_size = 0; 

  int *opt_maxsendlist, *numa_maxsendlist;      // max size of send list for each swap
  int opt_maxforward;
  std::vector<int> send_thr;
  int thr_maxswap;
  double **neigh_sublo, **neigh_subhi;
  double **neigh_nusublo, **neigh_nusubhi;

  int swap2numa_swap[RPROC];


  int *opt_maxsend, *numa_maxsend;
  int *opt_maxrecv, *numa_maxrecv;    // current size of send/recv buffer
  MPI_Datatype utofu_comm_type;
  utofu_tni_id_t  tni_id, *tni_ids;

  utofu_vcq_hdl_t             vcq_hdl_send[VCQ_NUM][TNI_NUM],         vcq_hdl_recv[VCQ_NUM][TNI_NUM];;
  utofu_vcq_id_t              lcl_vcq_id_send[VCQ_NUM][TNI_NUM],      lcl_vcq_id_recv[VCQ_NUM][TNI_NUM];  
  struct utofu_onesided_caps *onesided_caps_send[VCQ_NUM][TNI_NUM],   *onesided_caps_recv[VCQ_NUM][TNI_NUM];

  utofu_stadd_t all_send_stadd[VCQ_NUM][TNI_NUM], all_recv_stadd[VCQ_NUM][TNI_NUM];
  utofu_stadd_t all_x_stadd[VCQ_NUM][TNI_NUM], all_f_stadd[VCQ_NUM][TNI_NUM];

  unsigned long int post_flags;
  utofu_stadd_t *lcl_send_stadd[VCQ_NUM], *lcl_recv_stadd[VCQ_NUM];
  utofu_stadd_t *lcl_send_f_stadd[VCQ_NUM], *lcl_recv_x_stadd[VCQ_NUM];
  Utofu_comm *lcl_comms[VCQ_NUM], *lcl_recv_comms[VCQ_NUM];
  Utofu_comm *lcl_recv_numa_comms[VCQ_NUM];
  Utofu_comm *rmt_comms[VCQ_NUM], *nu_rmt_comms[VCQ_NUM];
  uint8_t   *edata;
  uintptr_t *cbvalue;
  uint64_t  *cbvalue_send;
  uint64_t  *edata_send;
  uint64_t  *edata_recvs;

  std::vector<Sh_swap_struct> send_swap_mnt;
  std::vector<Sh_swap_struct> recv_swap_mnt;
  Sh_swap_struct** recv_swap_mnt_numa;
  Swap_pair *opt2numa_swap;


  int directions[62][3];
  int swap_direct[124][3];

  int swap_nudirect[124][3];
  int ndims, nundims;

  key_t keys[NUMA_NUM];
  int shmids[NUMA_NUM];

  Share_numa_struct *shared_mems[NUMA_NUM];

  uint64_t shm_len;


  void* shm_data[NUMA_NUM];
  double *normal_data;

  std::atomic<uint64_t> *a_written_s0[T_THREAD];
  std::atomic<int> *a_written[T_THREAD];
  std::atomic<int> *a_written_tt[T_THREAD];
  std::atomic<int> *a_written_reverse[RPROC];

  void *atom_bit_share;
  void *atom_bit;

  uint64_t bit_nid    ;
  uint64_t ex_bit_nid ;

  std::vector<int> other_stages;
  int stage0, stage1, stage1_s;

  int a_written_id;
  int a_written_nu_id[NUMA_NUM];
  int a_written_tt_id[NUMA_NUM];

  uint64_t **opt_recvnum_numa;
  uint64_t **opt_firstrecv_numa;

  uint64_t *numa_firstrecv;

  uint64_t *shm_sendnum;
  int sync_shm_sendnum_ptr = 0;
  uint64_t *shm_nlocal;
  int sync_shm_nlocal_ptr = 0;
  uint64_t *shm_recvnum_numa;
  int sync_shm_recvnum_numa_ptr_s = 0;
  int sync_shm_recvnum_numa_ptr_r = 0;

  uint64_t *shm_firstrecv_numa;
  int sync_shm_firstrecv_numa_ptr = 0;

  int sync_shm_border_cpy_send_ptr = 0;

  int last_a_written_tt_0_ptr[T_THREAD] = {0};

  uint64_t reverse_target_bits[RPROC] = {0};

  uint64_t *shm_numa_recvnum_numa[NUMA_NUM];

  double **opt_force_recv[NUMA_NUM];    

  void init_shdmem(int &nu, key_t &key, int &shmid, void * &share_mem, int memlen, std::string vpfile, int __proj_id);
  void delete_shdmem(int shmid, void * share_mem);
  void setup_init_shdmem_region();
  // NOTE: init_buffers is called from a constructor and must not be made virtual
  void init_buffers();
  void init_buffers_opt();
  void init_buffers_numa();
  void buildMPIType();

  void warp_utofu_put(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t rmt_vcq_id,
      utofu_stadd_t lcl_send_stadd, utofu_stadd_t rmt_recv_stadd, size_t length,
      uint64_t edata, uintptr_t cbvalue, unsigned long int post_flags,void *piggydata);

  void warp_utofu_poll_tcq(utofu_vcq_hdl_t vcq_hdl, 
              uintptr_t &cbvalue, unsigned long int post_flags);

  void warp_utofu_poll_mrq(utofu_vcq_hdl_t vcq_hdl, 
      uint64_t &edata, unsigned long int post_flags,struct utofu_mrq_notice &in_notice);

  void utofu_recv(utofu_vcq_hdl_t vcq_hdl, uint64_t &edata, unsigned long int post_flags,struct utofu_mrq_notice &in_notice);

  int updown(int, int, int, double, int, double *);
  // compare cutoff to procs
  virtual void grow_send(int, int);       // reallocate send buffer
  virtual void grow_recv(int);            // free/allocate recv buffer
  virtual void grow_list(int, int);       // reallocate one sendlist
  virtual void grow_swap(int);            // grow swap, multi, and multi/old arrays
  virtual void allocate_swap(int);        // allocate swap arrays
  virtual void allocate_multi(int);       // allocate multi arrays
  virtual void allocate_multiold(int);    // allocate multi/old arrays
  virtual void free_swap();               // free swap arrays
  virtual void free_multi();              // free multi arrays
  virtual void free_multiold();           // free multi/old arrays

  const int con_direction[62][3] = {
    {0, 0, 1},    {0, 1, 0},    {1, 0, 0},
    {0, 1, 1},    {0, -1, 1},    {1, 0, 1},
    {-1, 0, 1},    {1, 1, 0},    {-1, 1, 0},
    {1, 1, 1},    {1, -1, 1},    {-1, 1, 1},
    {-1, -1, 1},  // 13 

    {0, 0, 2},    {0, 2, 0},    {2, 0, 0},  // 3

    {-2, -2, 2}, {-1, -2, 2}, {0, -2, 2}, {1, -2, 2}, {2, -2, 2}, 
    {-2, -1, 2}, {-1, -1, 2}, {0, -1, 2}, {1, -1, 2}, {2, -1, 2},  // 10
    {-2, 0, 2}, {-1, 0, 2},  {1, 0, 2}, {2, 0, 2},  // 4
    {-2, 1, 2}, {-1, 1, 2}, {0, 1, 2}, {1, 1, 2}, {2, 1, 2}, 
    {-2, 2, 2}, {-1, 2, 2}, {0, 2, 2}, {1, 2, 2}, {2, 2, 2},

    {-2, 2, 1}, {-1, 2, 1}, {0, 2, 1}, {1, 2, 1}, {2, 2, 1}, 
    {-2, -2, 1}, {-1, -2, 1}, {0, -2, 1}, {1, -2, 1}, {2, -2, 1},  // 20

    {-2, -1, 1}, {2, -1, 1}, 
    {-2, 0, 1}, {2, 0, 1}, 
    {-2, 1, 1}, {2, 1, 1},  

    {-2, 2, 0}, {-1, 2, 0},  {1, 2, 0}, {2, 2, 0},   
    {-2, 1, 0}, {2, 1, 0}   // 12
  };

  const int send_direction[26] = {
    24,25,22,23,20,21,18,19,
    16,17,14,15,12,13,10,11,8,9,6,7,
    0,1,4,5,2,3
  };

  void mem_barrier(int _cq_id, int _i, uint64_t _iter, uint64_t _pos = 0) {
    int iter = 0; int rc;
    while(1) {
      uint64_t result;
      __asm__ __volatile__(
          "ldr x0, %1;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "mov x1, %2;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "sub %0, x0, x1;"  // 执行减法 r0 - r1，并将结果存储在 result 中
          : "=r" (result)    // 输出操作数
          : "m" (opt_buf_recv[_cq_id][_i][_pos]), "r" (_iter) // 输入操作数（注意使用 "m" 约束来引用内存地址）
          : "x0", "x1"      // 破坏描述
      );

      if(!result) break;
    }; 
  };

  uint64_t test_barrier(int _cq_id, int _i, uint64_t _iter, uint64_t _pos = 0) {
    int iter = 0; int rc;
    uint64_t* tmp = (uint64_t*)opt_buf_recv[_cq_id][_i];
    while(1) {
      uint64_t result;
      __asm__ __volatile__(
          "ldr x0, %1;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "mov x1, %2;"      // 将 a 的地址加载到寄存器 r0，然后从该地址加载 a 的值
          "sub %0, x0, x1;"  // 执行减法 r0 - r1，并将结果存储在 result 中
          : "=r" (result)    // 输出操作数
          : "m" (opt_buf_recv[_cq_id][_i][_pos]), "r" (_iter) // 输入操作数（注意使用 "m" 约束来引用内存地址）
          : "x0", "x1"      // 破坏描述
      );

      return result;
    }; 
  };

};







}    // namespace LAMMPS_NS

#endif
