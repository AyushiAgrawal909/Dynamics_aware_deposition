/**
 * SPDX-License-Identifier: EUPL-1.2
 * SPDX-FileCopyrightText: 2025 Stefan Costea, LeCAD-PEG
 *
 * @file hexapic.hpp
 * @brief Core data structures, types, and global parameters for the HEXAPIC
 * Particle-in-Cell (PIC) code.
 */

/* HEXAPIC header file */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <openPMD/openPMD.hpp>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>
extern "C" {
#include "HYPRE.h"
#include "HYPRE_krylov.h"
#include "HYPRE_struct_ls.h"
}

using namespace openPMD;

/* Define convenience constants */
#ifndef PI
#define PI 3.141592653589793
#endif
#define TWOPI 6.28318530717959
#define EPS0 8.8542e-12
#define TRUE 1
#define FALSE 0
#define plot_3D FALSE
#define q_e 1.602e-19

// #define REAL float
#define REAL double

/* ... */
#define frand() ((REAL)rand() / (RAND_MAX + 1.0))
#define comm MPI_COMM_WORLD // parallel solver

struct Particle {
  REAL x, y, z;    ///< positions, in cell length units
  REAL vx, vy, vz; ///< velocities, in m/s
};

struct Species_load {
  REAL x1s, x1f; ///< X-coordinate interval to load in, in m
  REAL x2s, x2f; ///< Y-coordinate interval to load in, in m
  REAL density;  ///< initial load density, in particles/m^-3
  REAL x1s_full, x1f_full; ///< undecomposed X-coordinate interval
  REAL x2s_full, x2f_full; ///< undecomposed Y-coordinate interval
  REAL density_full;       ///< undecomposed initial load density
  int np2c;      ///< ratio of real to computed particles
  REAL v1thermal, v2thermal, v3thermal; ///< components of thermal velocity
  REAL v1drift, v2drift, v3drift;       ///< components of drift velocity
  int method;                           ///< load method, 0=uniform
};

struct Species_inject {
  REAL I;     ///< injection current in A
  int j1, j2; ///< start-end indexes of the boundary along X
  int k1, k2; ///< start-end indexes of the boundary along Y
  int normal; ///< direction of injection (+/-1)
  int np2c;   ///< ratio of real to computed particles
  REAL v1thermal, v2thermal, v3thermal; ///< components of thermal velocity
  REAL v1drift, v2drift, v3drift;       ///< components of drift velocity
  REAL f;                               ///< particle fraction
};

struct Source { // volumetric heating
  // rectangular region (maybe later add option with cell indexes)
  // particles are randomly selected and returned with v from maxwellian

  int active;      ///< if 0 not active
  int index;       ///< source index
  REAL n_inj;      ///< [s^-1 m^-3] /nc2p when injecting
  REAL n_inj_step; ///< injected computed particles per timestep

  int shape_x; ///< 0-uniform, 1-cos, 2-normal, details in implementation
  int shape_y; ///< 0-uniform, 1-cos, 2-normal

  int j1, k1;    ///< bottom left corner
  int j2, k2;    ///< top right corner
  int lx, ly;    ///< x, y local length
  int Lx, Ly;    ///< x, y full length (global)
  REAL cx, cy;   ///< center position of global source (in local coordinates)
  REAL Fx0, Fx1; ///< from cumulative distribution part on this rank
  REAL Fy0, Fy1; ///< same for y
  std::set<std::pair<REAL, int>> Fxcell, Fycell; ///< for normal distribution

  REAL Tpar, Tper; ///< temperature parallel and perpendicular to B [eV]
  // REAL vdpar; ///< drifting velocity along B (maybe add later)
  REAL heat_freq; ///< frequency of collisions of a particle inside region
  // the same for computed particle (no /nc2p)
  REAL n_heat; ///< number of heat collisions of a particle in timestep

  int t_dep; ///< time-dependend (source and heating)
  // for t_dep = 1:
  REAL t0, t1, t2, t3; ///< [t0, t1] source 0->1, [t2, t3] source 1->0 (linear)
};

struct SourceSpecie {
  int index;         ///< source index (ordering done after reading input file)
  REAL source_frac;  ///< n_inj * source_frac to get specie specific injecting
  REAL heat_frac;    ///< freq*heat_frac to get specie specific heating
  REAL vtpar, vtper; ///< parallel and perpendicular thermal velocity
};

typedef void (*reactionFunction)(int, int, int, int, int, int, int, REAL);

struct Collision {
  // describes possible colisions between 2 species:
  // first one colliding into the second one (crossections)
  std::vector<std::string> reaction_names; ///< collision reaction names
  int sp[2];                               ///< colliding species indexes
  int n_reactions; ///< number of reactions between this two species

  // n_reactions possible reactions:
  // vector of collision functions
  std::vector<reactionFunction> processCollision;
  // (CollisionID, ReactionID, CellID, sp1ID, sp2ID, part1ID, part2ID)
  // We can store additional data in Collision (new vector or something...)
  // and access it inside a collision function with CollisionID, ReactionID
  std::vector<std::vector<int>> sp_out; ///< new species created in collision

  // vector of (j) tabulated collision cross-sections
  std::vector<std::set<std::pair<REAL, REAL>>> sigma;
  // first dimension along different reactions
  // set ordered along energies [eV] (first in pair)
  // second in pair is sigma -> cross-section value
  std::vector<REAL> E_min; ///< minimal energy required for reaction
  // E_min = E_ionisation or  E_min = delta_E for ionisation or excitation
  REAL sigma_v_max; ///< maximal over all energies (sigma_total*v)
};

struct Species {
  std::string name; ///< specie name, e.g. electron
  unsigned int n;   ///< number of particles of a specie
  int charge;       ///< specie charge/elementary charge
  REAL mass, chom;  ///< specie mass and charge/mass ratio
  unsigned int subcycle;
  REAL tb[3], sb[3];              ///< specie coefficients for Boris mover
  std::vector<Species_load> load; ///< allows multiple loads with same species
  std::vector<Species_inject> inject; ///< particle injection at boundary
  std::vector<SourceSpecie> source;   ///< volumetric specie sources
  std::vector<REAL> n_store;          ///< particle density diagnostic storage
  std::vector<REAL> T_store;          ///< temperature diagnostic storage
  std::vector<REAL> v_store;          ///< fluid velocity diagnostic storage

  void correct_inject(int Nx, int Ny, int Nx_old, int Ny_old) {
    // in case of changing Nx, Ny inject must be corrected
    // stretch for factor N*/N*_old
    for (auto &in : inject) {
      if (in.j1 != 0) {
        in.j1 = (int)(Nx * ((double)in.j1 / Nx_old));
      }
      if (in.j2 != 0) {
        in.j2 = (int)(Nx * ((double)in.j2 / Nx_old));
      }
      if (in.k1 != 0) {
        in.k1 = (int)(Ny * ((double)in.k1 / Ny_old));
      }
      if (in.k2 != 0) {
        in.k2 = (int)(Ny * ((double)in.k2 / Ny_old));
      }
    }
  }
};

struct SecondaryEmission {
  int sp1;                     ///< primary specie
  std::vector<int> sp2;        ///< secondary species
  int sec_type;                ///< emission type
  int inj_type;                ///< injection type
  REAL Ew, E0, gamma0, T0, Ep; ///< coefficients for injection and gamma
                               /*
                               effect of different coefficients explained inside PSI routines
                               it is not necessary that E0 is allways energy for non 0 sec_type
                               something else of similar importance might be saved here
                               what is actualy saved is explained in PSI routines
                               units [eV], [m/s], ...
                               */
};

struct Boundary {
  std::string name; ///< boundary name, e.g. Equipotential
  REAL C;           ///< fixed potential on boundary, in V
  REAL reflection;  ///< part of reflected particles (for neumann=1)
  // later make it specie specific
  REAL QuseFlag; ///< charge accumulation (for neumann=0)
  int j1, j2;    ///< start-end indexes of the boundary along X
  int k1, k2;    ///< start-end indexes of the boundary along Y

  // coordinates remain in global grid for the whole time

  // for PSI
  std::vector<std::vector<SecondaryEmission>> sp_sec;
  // dim0: primary particle, dim1: different secondary emissions

  void add_parameters(std::string NAME, int J1, int J2, int K1, int K2) {
    // add basic parameters to boundary
    this->name = NAME;
    this->j1 = J1;
    this->j2 = J2;
    this->k1 = K1;
    this->k2 = K2;
  }
};

struct Neighbours {
  std::set<int> all; ///< set of all neighbours (to expect data)
  std::vector<int> rxmin, rxmax, rymin, rymax; ///< ranks of neighbours
  std::vector<int> corners;                    ///< ranks of corner neighbours
  // [bottom-left, top-left, bottom-right, top-right]

  std::vector<std::vector<REAL>> shift_xy; ///< shift between local grids
  // to get position in local grid (+) shift local position from other rank

  int max_recv_size;                        ///< maximal size of received data
  std::vector<std::vector<REAL>> send_data; ///< send buffer
  std::vector<std::vector<std::vector<REAL>>> send_data_3d;
  std::vector<std::vector<REAL>> recv_data; ///< receive buffer
};

struct Cell {
  int n0x, n0y; ///< position of bottom left corner in local grid
  std::vector<std::vector<Particle>> part; ///< particle of each specie in cell
  std::vector<int> neighbours;             ///< all neighbouring cells
  // left, right, bottom, top, b-l, t-l, b-r, t-r
  // negative neighbour index means sending to rank abs(index)-1
};

struct Hypre {
  // We are using Hypre Struct
  // No ghost values access -> MPI communication
  // Upper-right nodes on local domain edge
  // belong to neighbouring ranks (except at boundary)
  // Rho is send in one direction, phi in the other
  // Order of values in bottom->top, left->right order

  int ilower[2], iupper[2];
  int periodic[2];
  int size, size_extend; // number solving nodes, all needed nodes
  int nx, ny;            // along dimension
  std::vector<int> bcs;  // for each (local) boundary
  // 0/1/2 = no / dirichlet / open

  // for each rank (neighbours)
  // indexes in hexapic way (i*Ny + j)
  std::vector<int> size_s;
  std::vector<int> size_r;
  std::vector<std::vector<int>> V_s_idx;
  std::vector<std::vector<int>> V_r_idx;
  std::vector<std::vector<double>> send_buff;
  std::vector<std::vector<double>> recv_buff;
};

struct Grid {
  int Nx, Ny;    ///< number of local grid edges in X and Y directions
  int Nxg, Nyg;  ///< number of global grid edges in X and Y directions
  int Ncx, Ncy;  ///< number of local grid cells in X and Y directions
  int N0x, N0y;  ///< shift of grid cells of local grid in global grid
  REAL x1s, x1f; ///< start and end extension along X, in m
  REAL x2s, x2f; ///< start and end extension along Y, in m
  REAL dx, dy;   ///< cell dimensions along X and Y coordinates, in m
  std::vector<Boundary> boundaries; ///< also includes periodic boundaries
  Neighbours neighbours;
  std::vector<REAL> V_store; ///< electric potential diagnostic storage
  REAL p2n;                  ///< particle-to-density constant

  void sort_boundaries() {
    std::vector<Boundary> boundaries_sorted(4);
    // Loop through the boundaries and sort them based on their positions
    // order: bc_xmin, bc_xmax, bc_ymin, bc_ymax
    for (Boundary &bc : boundaries) {
      if (bc.j1 == 0 && bc.j2 == 0)
        boundaries_sorted[0] = bc; // Left boundary
      else if (bc.j1 == Nxg - 1 && bc.j2 == Nxg - 1)
        boundaries_sorted[1] = bc; // Right boundary
      else if (bc.k1 == 0 && bc.k2 == 0)
        boundaries_sorted[2] = bc; // Bottom boundary
      else if (bc.k1 == Nyg - 1 && bc.k2 == Nyg - 1)
        boundaries_sorted[3] = bc; // Top boundary
    }
    boundaries = boundaries_sorted;
  }

  void correct_boundaries() {
    // in case of changing Nx, Ny boundaries must be corrected
    int Nx_old, Ny_old;
    for (Boundary &bc : boundaries) {
      if (bc.j1 != 0)
        bc.j1 = Nx;
      if (bc.j2 != 0)
        bc.j2 = Nx;
      if (bc.k1 != 0)
        bc.k1 = Ny;
      if (bc.k2 != 0)
        bc.k2 = Ny;
    }
  }
};

/* Define external variables */
#if defined(EXTERNALS)
#define EXTERNAL
#else
#define EXTERNAL extern
#endif

EXTERNAL Grid grid;
EXTERNAL Hypre hypre;
EXTERNAL std::vector<Cell> cells;
EXTERNAL std::vector<Cell *> cells_interior;
EXTERNAL std::vector<Cell *> cells_edge;
EXTERNAL std::vector<Species> species;
EXTERNAL std::vector<Collision> collisions;
EXTERNAL std::vector<double> rho;
EXTERNAL std::vector<int> rho_index;
EXTERNAL std::vector<REAL> trajx, trajy, trajz;
EXTERNAL std::vector<REAL> t_store;
EXTERNAL std::vector<MeshRecordComponent> meshes;
EXTERNAL int tstep, boris, dsteps, davg;
EXTERNAL int FieldSolverFlag, ParticleDiagnosticFlag, QuietFlag;
EXTERNAL int nmpi, rmpi;
EXTERNAL REAL dt, hdt, dx, dy, dV, Sconst, np2c_global;
EXTERNAL bool digital_smoothing;
EXTERNAL int deposition_mode; ///< 0=cic, 1=cic_plus_filter
EXTERNAL int deposition_action; ///< -1=disabled, otherwise see action-library registry
EXTERNAL REAL deposition_sigma_cells; ///< <=0 uses action default sigma
EXTERNAL std::string compact_lookup_table_5x5;
EXTERNAL std::string compact_lookup_table_7x7;
EXTERNAL bool compact_skip_rho_halo_exchange;
EXTERNAL std::string controller_runtime_bundle_dir;
EXTERNAL std::string controller_runtime_csv_path;
EXTERNAL bool controller_runtime_enabled;
EXTERNAL bool controller_sigma_runtime_enabled;
EXTERNAL int controller_tile_nx;
EXTERNAL int controller_tile_ny;
EXTERNAL std::vector<int> controller_selected_action_tiles;
EXTERNAL std::vector<unsigned long long> controller_step_action_histogram;
EXTERNAL std::vector<unsigned long long> controller_total_action_histogram;
EXTERNAL unsigned long long controller_step_fallback_count;
EXTERNAL unsigned long long controller_total_fallback_count;
EXTERNAL unsigned long long controller_step_switch_count;
EXTERNAL unsigned long long controller_total_switch_count;
EXTERNAL double controller_step_entropy_mean;
EXTERNAL double controller_step_entropy_max;
EXTERNAL REAL Bf[3];    ///< external magnetic field
EXTERNAL REAL Bb[3][3]; ///< normalised basis for B
EXTERNAL std::vector<Source> sources;
EXTERNAL FILE *f;
EXTERNAL std::vector<std::vector<std::string>> MCC_init_data;
EXTERNAL std::vector<std::vector<std::string>> PSI_init_data;
EXTERNAL std::string runtime_energy_csv_path;
EXTERNAL std::string runtime_conservation_csv_path;
EXTERNAL std::string runtime_transport_csv_path;
EXTERNAL std::string runtime_interface_csv_path;
EXTERNAL double runtime_kinetic_energy_exact;
EXTERNAL double runtime_field_energy_exact;
EXTERNAL double runtime_total_energy_exact;
EXTERNAL std::vector<unsigned long long> runtime_initial_macro_count;
EXTERNAL std::vector<double> runtime_initial_charge_total;
EXTERNAL bool runtime_conservation_reference_set;
EXTERNAL unsigned long long transport_particles_sent_step;
EXTERNAL unsigned long long transport_particles_received_step;
EXTERNAL unsigned long long transport_particles_wrapped_step;
EXTERNAL unsigned long long transport_particles_multi_rank_step;
EXTERNAL std::vector<double> interface_deposition_interior_charge_step;
EXTERNAL std::vector<double> interface_deposition_edge_charge_step;
EXTERNAL std::vector<double> interface_deposition_interior_abs_charge_step;
EXTERNAL std::vector<double> interface_deposition_edge_abs_charge_step;
EXTERNAL std::vector<unsigned long long> interface_deposition_interior_particles_step;
EXTERNAL std::vector<unsigned long long> interface_deposition_edge_particles_step;
EXTERNAL std::vector<unsigned long long> interface_halo_stencil_particles_step;
EXTERNAL double interface_periodic_v_mismatch_x_step;
EXTERNAL double interface_periodic_v_mismatch_y_step;
EXTERNAL unsigned long long interface_periodic_v_pairs_x_step;
EXTERNAL unsigned long long interface_periodic_v_pairs_y_step;
EXTERNAL double interface_periodic_ex_mismatch_x_step;
EXTERNAL double interface_periodic_ey_mismatch_x_step;
EXTERNAL double interface_periodic_ex_mismatch_y_step;
EXTERNAL double interface_periodic_ey_mismatch_y_step;
EXTERNAL unsigned long long interface_periodic_field_pairs_x_step;
EXTERNAL unsigned long long interface_periodic_field_pairs_y_step;
EXTERNAL int LandauInitModeFlag;
EXTERNAL REAL LandauAlpha;
EXTERNAL int LandauModeIndex;
EXTERNAL REAL LandauAlphaY;
EXTERNAL int LandauModeIndexY;

std::string sanitize_csv_name(const std::string &name);

EXTERNAL Series series;
EXTERNAL std::vector<std::vector<REAL>> chunks_float;

EXTERNAL void particles_init(), inject_particle(int, REAL, REAL),
    grid_init(int, char **), part2grid_edge(), part2grid_interior(),
    particle_mover_boris(const std::vector<Cell *> &), save_output(),
    num_param_init(), create_output(), grid2part(REAL, REAL, double *, REAL *),
    particle_boundaries(), read_input_file(char **), initial_particle_load(),
    inject_particles(), plot_profiling(), decompose_domain(),
    send_receive_particles(), add_received_particles(), cells_init(),
    collisions_init(), MCC(), remove_particle(int, int, int),
    particle_wall_interaction(size_t, int, int, int), PSI_init(), source_init(),
    source(), source_heating(), hypre_init(), hypre_cleanup(),
    hypre_source_update(), hypre_field_solver(), hypre_get_V(),
    hypre_post_V_halo(), hypre_finish_V_halo(), hypre_post_rho_halo(),
    hypre_finish_rho_halo(), hypre_rho_update(), hypre_update_Vlocal(),
    reset_interface_diagnostics_step(), controller_runtime_init(),
    controller_prepare_step(), momentum_audit_reset(), momentum_audit_report();

EXTERNAL int controller_action_for_global_cell(int, int);
EXTERNAL bool controller_sigma_runtime_active();
EXTERNAL double controller_sigma_x_for_global_cell(int, int);
EXTERNAL double controller_sigma_y_for_global_cell(int, int);

EXTERNAL int specie_index(std::string);
EXTERNAL std::vector<int> rank_domain(int, int, int, int);

EXTERNAL REAL normvel_MaxBol(), normvel_vFv();

EXTERNAL double et[10]; ///< elapsed time for profiling

EXTERNAL HYPRE_StructGrid grid_h;
EXTERNAL HYPRE_StructStencil stencil_h;
EXTERNAL HYPRE_StructMatrix A_h;
EXTERNAL HYPRE_StructVector b_h, x_h;
// EXTERNAL HYPRE_StructSolver solver_h, precond_h;
EXTERNAL HYPRE_StructSolver pfmg;
EXTERNAL double *V_h, *rho_h; // only for hypre node indexes!
EXTERNAL double *V_local;     // hexapic indexing
EXTERNAL double dx2_inv, dy2_inv;

// --- HEXAPIC / HYPRE halo exchange persistent MPI setup ---

// Persistent requests for V and rho halo exchanges
EXTERNAL std::vector<MPI_Request> hypre_persist_v_reqs;
EXTERNAL std::vector<MPI_Request> hypre_persist_rho_reqs;

/* .... */

#undef EXTERNAL
