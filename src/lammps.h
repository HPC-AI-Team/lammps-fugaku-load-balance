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

#ifndef LMP_LAMMPS_H
#define LMP_LAMMPS_H

#include <cstdio>
#include <mpi.h>
#include <utofu.h>
#include <pthread.h>
#include <atomic>
#include <mpi-ext.h>  // Include header file
#include <unistd.h>
#include "utils.h"
#include <mutex>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/ipc.h>
#include <numa.h>
#include <numaif.h>

#define RPROC   124
#define DIM_NUM 62
#define EXCH_SWAP_NUM 26
#define SWAP_NUM DIM_NUM*2
#define TNI_NUM 6
#define VCQ_NUM 3
#define NUMA_GROUP_NUM 2
// #define XMIT_TYPE_NUM 4

#define THREAD_NUM 11
#define COMM_TNUM  6
#define THREAD_STRIDE 5
#define T_THREAD 12
#define NUMA_NUM 4
#define FORCE_OFFSET 108

#define SHARE_DATA_LENGTH 163840

#define OPT_NEWTON

#define DEBUG_MSG comm->debug_flag
// #define DEBUG_MSG 0

#define OPT_COMM_TEST
// #define DEBUG_MSG 1


namespace LAMMPS_NS {

class LAMMPS {
 public:
  // ptrs to fundamental LAMMPS classes
  class Memory *memory;            // memory allocation functions
  class Error *error;              // error handling
  class Universe *universe;        // universe of processors
  class Input *input;              // input script processing
                                   // ptrs to top-level LAMMPS-specific classes
  class Atom *atom;                // atom-based quantities
  class Update *update;            // integrators/minimizers
  class Neighbor *neighbor;        // neighbor lists
  class Comm *comm;                // inter-processor communication
  class Domain *domain;            // simulation box
  class Force *force;              // inter-particle forces
  class Modify *modify;            // fixes and computes
  class Group *group;              // groups of atoms
  class Output *output;            // thermo/dump/restart
  class Timer *timer;              // CPU timing info
  class Timer *self_timer;              // CPU timing info
                                   //
  class KokkosLMP *kokkos;         // KOKKOS accelerator class
  class AtomKokkos *atomKK;        // KOKKOS version of Atom class
  class MemoryKokkos *memoryKK;    // KOKKOS version of Memory class
  class Python *python;            // Python interface
  class CiteMe *citeme;            // handle citation info


  enum Fun_type{
    FORWARD_XMIT      = 0x01,
    REVERSE_XMIT      = 0x02,
    BORDER_NUMA       = 0x03,
    FORWARD_NUMA      = 0x04,
    REVERSE_NUMA      = 0x05,
    EXCHANGE_NUMA     = 0x06,
    NUMA_SHARE_ATOM   = 0x07,
    EXCHANGE_XMIT     = 0x09,
    NEIGHBOR_BUILD    = 0x0a,
    PAIR_COMPUTE      = 0x0b,
    PRE_FORCE_P       = 0x0c,
    BORDER_COMBINE    = 0x0d,
    PAIR_NUMA         = 0x0e,
    NUMA_ATOM_INIT    = 0x10,
    NEIGHBOR_NUMA     = 0x0f,
    INIT_INTEG_NUMA     = 0x11,
    FINI_INTEG_NUMA     = 0x12
  };

  uint64_t excutions[32] = {0};

  pthread_t lmp_threads[THREAD_NUM];
  std::atomic<uint64_t> is_excute;
  // std::atomic<int> is_finish;
  // std::atomic<int> is_last;
  // std::atomic<bool> is_release;
  // std::atomic<int> is_last;
  int is_finish[T_THREAD];
  std::atomic<bool> is_release[T_THREAD];
  int mtx_ptr[T_THREAD];
  std::mutex mtx[T_THREAD];
  std::atomic<int> is_barrier[T_THREAD];

  bool is_shutdown_ = false;
  static LAMMPS* curMy;

  int numa_id;

  bool temp_first = true;

  bool stop_flag = false;

  void parral_barrier(int t_num = 12, int tid = 0);


  static void* callback(void* arg){  
    // utils::logmesg(LAMMPS::curMy, " callback create tid {} \n", int(arg)); 
    LAMMPS::curMy->parral_fun((int)(size_t)(arg));  
    return NULL;  
  }  

  void init_excutions();

  void execute(enum Fun_type);
  void parral_fun(int);
  void pthread_pool_start();
  void pthread_test();

  class DeepPot **deep_pots;



  pthread_barrier_t barrier_12;


  void setCurMy()  
  {//设置当前对象为回调函数调用的对象  
      curMy = this;  
  }

  const char *version;    // LAMMPS version string = date
  int num_ver;            // numeric version id derived from *version*
                          // that is constructed so that will be greater
                          // for newer versions in numeric or string
                          // value comparisons
  int restart_ver;        // -1 or numeric version id of LAMMPS version in restart
                          // file, in case LAMMPS was initialized from a restart
                          //
  MPI_Comm world;         // MPI communicator
  MPI_Comm nuworld;
  MPI_Comm m_world;

  FILE *infile;           // infile
  FILE *screen;           // screen output
  FILE *logfile;          // logfile
                          //
  double initclock;       // wall clock at instantiation
  int skiprunflag;        // 1 inserts timer command to skip run and minimize loops

  char *suffix, *suffix2;    // suffixes to add to input script style names
  int suffix_enable;         // 1 if suffixes are enabled, 0 if disabled
  int pair_only_flag;        // 1 if only force field pair styles are accelerated, 0 if all
  const char *non_pair_suffix() const;
  char *exename;             // pointer to argv[0]

  char ***packargs;    // arguments for cmdline package commands
  int num_package;     // number of cmdline package commands

  MPI_Comm external_comm;    // MPI comm encompassing external programs
                             // when multiple programs launched by mpirun
                             // set by -mpicolor command line arg

  void *mdicomm;    // for use with MDI code coupling library

  const char *match_style(const char *style, const char *name);
  static const char *installed_packages[];
  static bool is_installed_pkg(const char *pkg);

  static bool has_git_info();
  static const char *git_commit();
  static const char *git_branch();
  static const char *git_descriptor();

  LAMMPS(int, char **, MPI_Comm);
  ~LAMMPS();
  void create();
  void post_create();
  void init();
  void destroy();
  void print_config(FILE *);    // print compile time settings

 private:
  struct package_styles_lists *pkg_lists;
  void init_pkg_lists();
  void help();
  /// Default constructor. Declared private to prohibit its use
  LAMMPS(){};
  /// Copy constructor. Declared private to prohibit its use
  LAMMPS(const LAMMPS &){};
};

}    // namespace LAMMPS_NS

#endif
