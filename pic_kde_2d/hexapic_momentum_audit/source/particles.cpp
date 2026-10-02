/**
 * SPDX-License-Identifier: EUPL-1.2
 * SPDX-FileCopyrightText: 2025 Stefan Costea, LeCAD-PEG
 *
 * @file particles.cpp
 * @brief Particle pusher, weighting/interpolation, and particle management
 * (injection, movement, deposition).
 */

/* HEXAPIC particles section */

#include "hexapic.hpp"
#include "deposition_action_library.hpp"
#include "kde_deposition.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>

namespace {
std::vector<std::array<int, 4>> particle_rank_boxes;
std::vector<int> particle_exchange_ranks;
std::vector<char> particle_halo_node_mask;
std::vector<char> particle_cell_halo_mask;

double momentum_audit_force_local[2] = {0.0, 0.0};
double momentum_audit_force_abs_local[2] = {0.0, 0.0};
double momentum_audit_before_local[2] = {0.0, 0.0};
double momentum_audit_initial_global[2] = {0.0, 0.0};
double momentum_audit_impulse_global[2] = {0.0, 0.0};
bool momentum_audit_reference_set = false;

bool momentum_audit_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("HEXAPIC_MOMENTUM_AUDIT");
    return value != nullptr && std::string(value) != "0";
  }();
  return enabled;
}

void momentum_audit_local_particle_momentum(double *momentum) {
  momentum[0] = 0.0;
  momentum[1] = 0.0;
  for (const Cell &cell : cells) {
    for (size_t species_index = 0; species_index < species.size();
         ++species_index) {
      const Species &sp = species[species_index];
      const double macro_mass = np2c_global * static_cast<double>(sp.mass);
      for (const Particle &particle : cell.part[species_index]) {
        momentum[0] += macro_mass * static_cast<double>(particle.vx);
        momentum[1] += macro_mass * static_cast<double>(particle.vy);
      }
    }
  }
}

constexpr int kDepositionModeCIC = 0;
constexpr int kDepositionModeCICPlusFilter = 1;
constexpr int kDepositionModeGaussian5x5 = 2;
constexpr int kDepositionModeGaussian7x7 = 3;

constexpr int kStencilSize5 = 5;
constexpr int kStencilSize7 = 7;
constexpr int kStencilRadius5 = kStencilSize5 / 2;
constexpr int kStencilRadius7 = kStencilSize7 / 2;
constexpr int kMaxGaussianLookupWeights = kStencilSize7 * kStencilSize7;
constexpr REAL kGaussianLookupSigmaOverDx = static_cast<REAL>(0.5);

struct Phase1Lookup {
  bool loaded = false;
  std::string loaded_path;
  int stencil_size = 0;
  int weight_count = 0;
  int Nu = 0;
  int Nv = 0;
  int Na = 0;
  REAL u_min = -0.5;
  REAL u_max = 0.5;
  REAL v_min = -0.5;
  REAL v_max = 0.5;
  REAL a_min = 0.0;
  REAL a_max = 1.0;
  std::vector<float> data;

  void clear() {
    loaded = false;
    loaded_path.clear();
    stencil_size = 0;
    weight_count = 0;
    Nu = 0;
    Nv = 0;
    Na = 0;
    data.clear();
  }

  bool load_binary(const std::string &path, int expected_stencil) {
    if (loaded && loaded_path == path)
      return true;
    clear();

    std::ifstream in(path, std::ios::binary);
    if (!in.good())
      return false;

    char magic[12] = {0};
    in.read(magic, sizeof(magic));
    if (!in.good())
      return false;

    std::string magic_s(magic, sizeof(magic));
    int file_stencil = 0;
    if (magic_s.rfind("P1LUT5X5V1", 0) == 0)
      file_stencil = kStencilSize5;
    else if (magic_s.rfind("P1LUT7X7V1", 0) == 0)
      file_stencil = kStencilSize7;
    else
      return false;
    if (file_stencil != expected_stencil)
      return false;

    std::uint32_t nu = 0, nv = 0, na = 0;
    float umin = 0.0f, umax = 0.0f, vmin = 0.0f, vmax = 0.0f, amin = 0.0f,
          amax = 0.0f;
    in.read(reinterpret_cast<char *>(&nu), sizeof(nu));
    in.read(reinterpret_cast<char *>(&nv), sizeof(nv));
    in.read(reinterpret_cast<char *>(&na), sizeof(na));
    in.read(reinterpret_cast<char *>(&umin), sizeof(umin));
    in.read(reinterpret_cast<char *>(&umax), sizeof(umax));
    in.read(reinterpret_cast<char *>(&vmin), sizeof(vmin));
    in.read(reinterpret_cast<char *>(&vmax), sizeof(vmax));
    in.read(reinterpret_cast<char *>(&amin), sizeof(amin));
    in.read(reinterpret_cast<char *>(&amax), sizeof(amax));
    if (!in.good() || nu < 2 || nv < 2 || na < 2)
      return false;

    int weights = file_stencil * file_stencil;
    size_t total =
        static_cast<size_t>(nu) * static_cast<size_t>(nv) *
        static_cast<size_t>(na) * static_cast<size_t>(weights);
    data.resize(total);
    in.read(reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(total * sizeof(float)));
    if (!in.good()) {
      clear();
      return false;
    }

    loaded = true;
    loaded_path = path;
    stencil_size = file_stencil;
    weight_count = weights;
    Nu = static_cast<int>(nu);
    Nv = static_cast<int>(nv);
    Na = static_cast<int>(na);
    u_min = static_cast<REAL>(umin);
    u_max = static_cast<REAL>(umax);
    v_min = static_cast<REAL>(vmin);
    v_max = static_cast<REAL>(vmax);
    a_min = static_cast<REAL>(amin);
    a_max = static_cast<REAL>(amax);
    return true;
  }

  inline float at(int iu, int iv, int ia, int k) const {
    size_t idx =
        (((static_cast<size_t>(iu) * static_cast<size_t>(Nv) +
           static_cast<size_t>(iv)) *
              static_cast<size_t>(Na) +
          static_cast<size_t>(ia)) *
             static_cast<size_t>(weight_count) +
         static_cast<size_t>(k));
    return data[idx];
  }

  bool sample_trilinear(REAL u, REAL v, REAL a, REAL *weights_out) const {
    if (!loaded || u < u_min || u > u_max || v < v_min || v > v_max ||
        a < a_min || a > a_max)
      return false;

    REAL du = (u_max - u_min) / static_cast<REAL>(Nu - 1);
    REAL dv = (v_max - v_min) / static_cast<REAL>(Nv - 1);
    REAL da = (a_max - a_min) / static_cast<REAL>(Na - 1);
    if (du <= 0.0 || dv <= 0.0 || da <= 0.0)
      return false;

    REAL tu = std::min(std::max((u - u_min) / du, static_cast<REAL>(0.0)),
                       static_cast<REAL>(Nu - 1));
    REAL tv = std::min(std::max((v - v_min) / dv, static_cast<REAL>(0.0)),
                       static_cast<REAL>(Nv - 1));
    REAL ta = std::min(std::max((a - a_min) / da, static_cast<REAL>(0.0)),
                       static_cast<REAL>(Na - 1));

    int iu0 = static_cast<int>(std::floor(tu));
    int iv0 = static_cast<int>(std::floor(tv));
    int ia0 = static_cast<int>(std::floor(ta));
    if (iu0 >= Nu - 1)
      iu0 = Nu - 2;
    if (iv0 >= Nv - 1)
      iv0 = Nv - 2;
    if (ia0 >= Na - 1)
      ia0 = Na - 2;

    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    int ia1 = ia0 + 1;
    REAL fu = tu - static_cast<REAL>(iu0);
    REAL fv = tv - static_cast<REAL>(iv0);
    REAL fa = ta - static_cast<REAL>(ia0);

    for (int k = 0; k < weight_count; k++) {
      REAL c000 = at(iu0, iv0, ia0, k);
      REAL c100 = at(iu1, iv0, ia0, k);
      REAL c010 = at(iu0, iv1, ia0, k);
      REAL c110 = at(iu1, iv1, ia0, k);
      REAL c001 = at(iu0, iv0, ia1, k);
      REAL c101 = at(iu1, iv0, ia1, k);
      REAL c011 = at(iu0, iv1, ia1, k);
      REAL c111 = at(iu1, iv1, ia1, k);

      REAL c00 = c000 * (1.0 - fu) + c100 * fu;
      REAL c10 = c010 * (1.0 - fu) + c110 * fu;
      REAL c01 = c001 * (1.0 - fu) + c101 * fu;
      REAL c11 = c011 * (1.0 - fu) + c111 * fu;
      REAL c0 = c00 * (1.0 - fv) + c10 * fv;
      REAL c1 = c01 * (1.0 - fv) + c11 * fv;
      weights_out[k] = c0 * (1.0 - fa) + c1 * fa;
    }
    return true;
  }
};

Phase1Lookup compact_lookup_5x5;
Phase1Lookup compact_lookup_7x7;
std::vector<double> compact_rho_send;
std::vector<double> compact_rho_sum;
std::vector<double> compact_phi_send;
std::vector<double> compact_phi_sum;
std::vector<double> compact_ex_sum;
std::vector<double> compact_ey_sum;
int compact_phi_tstep = -1;
bool warned_lookup_5x5 = false;
bool warned_lookup_7x7 = false;
} // namespace

// Forward declarations needed by the KDE bridge functions below.
static void ensure_compact_global_potential();
static bool map_global_node_index(int ii_global, int jj_global, int &out_idx);

// Bridge functions called by kde_deposition.cpp.  Defined here because they
// need access to the static helpers and anonymous-namespace data above.
namespace hexapic::kde {
void ensure_phi_global() { ensure_compact_global_potential(); }
double get_phi_global(int ig, int jg) {
  int idx = -1;
  if (!map_global_node_index(ig, jg, idx))
    return 0.0;
  return compact_phi_sum[static_cast<size_t>(idx)];
}
} // namespace hexapic::kde

inline static bool is_edge(int i, int j, int Nx, int Ny) {
  return i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1;
}

void particles_init() {
  REAL tb2;
  unsigned int seed = static_cast<unsigned int>(time(NULL) * (rmpi + 1));
  const char *seed_env = getenv("HEXAPIC_RANDOM_SEED");

  for (auto &sp : species) {
    sp.chom = q_e * sp.charge / sp.mass;
    tb2 = 0.0;
    for (int j = 0; j < 3; j++) {
      sp.tb[j] = hdt * sp.chom * Bf[j];
      tb2 += sp.tb[j] * sp.tb[j];
    }
    for (int j = 0; j < 3; j++)
      sp.sb[j] = 2 * sp.tb[j] / (1 + tb2);
  }
  if (seed_env && seed_env[0] != '\0')
    seed = static_cast<unsigned int>(strtoull(seed_env, nullptr, 10) + rmpi);
  srand(seed);

  particle_rank_boxes.clear();
  particle_rank_boxes.resize(nmpi);
  for (int rank = 0; rank < nmpi; rank++) {
    std::vector<int> domain = rank_domain(grid.Nxg - 1, grid.Nyg - 1, nmpi, rank);
    particle_rank_boxes[rank] = {domain[0], domain[1], domain[2], domain[3]};
  }

  particle_exchange_ranks.clear();
  for (int rank = 0; rank < nmpi; rank++)
    if (rank != rmpi)
      particle_exchange_ranks.push_back(rank);

  particle_halo_node_mask.assign(static_cast<size_t>(hypre.size_extend), 0);
  for (int rank = 0; rank < nmpi; rank++)
    for (int node_idx : hypre.V_r_idx[rank])
      if (node_idx >= 0 && node_idx < hypre.size_extend)
        particle_halo_node_mask[static_cast<size_t>(node_idx)] = 1;

  particle_cell_halo_mask.assign(cells.size(), 0);
  for (size_t cell_i = 0; cell_i < cells.size(); cell_i++) {
    const Cell &cell = cells[cell_i];
    int J = cell.n0x * grid.Ny + cell.n0y;
    if (particle_halo_node_mask[static_cast<size_t>(J)] ||
        particle_halo_node_mask[static_cast<size_t>(J + 1)] ||
        particle_halo_node_mask[static_cast<size_t>(J + grid.Ny)] ||
        particle_halo_node_mask[static_cast<size_t>(J + grid.Ny + 1)])
      particle_cell_halo_mask[cell_i] = 1;
  }

  transport_particles_sent_step = 0;
  transport_particles_received_step = 0;
  transport_particles_wrapped_step = 0;
  transport_particles_multi_rank_step = 0;
  compact_phi_tstep = -1;
  hexapic::kde::kde_init();
}

static int decompose_expected_particle_count(REAL expected_local) {
  double prefix_expected = 0.0;
  double local_expected = expected_local > 0.0 ? expected_local : 0.0;

  MPI_Exscan(&local_expected, &prefix_expected, 1, MPI_DOUBLE, MPI_SUM, comm);
  if (!rmpi)
    prefix_expected = 0.0;

  long long count_before =
      static_cast<long long>(std::floor(prefix_expected + 1e-12));
  long long count_after =
      static_cast<long long>(std::floor(prefix_expected + local_expected + 1e-12));

  long long local_count = count_after - count_before;
  return static_cast<int>(local_count > 0 ? local_count : 0);
}

static uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static REAL deterministic_unit_sample(int sp_idx, int load_idx, int sample_idx,
                                      int component) {
  uint64_t key = (static_cast<uint64_t>(sample_idx) << 32) ^
                 (static_cast<uint64_t>(sp_idx) << 16) ^
                 (static_cast<uint64_t>(load_idx) << 4) ^
                 static_cast<uint64_t>(component);
  return static_cast<REAL>((mix64(key) >> 11) *
                           (1.0 / 9007199254740992.0));
}

static REAL deterministic_gaussian_sample(int sp_idx, int load_idx,
                                          int sample_idx, int component) {
  REAL u1 = deterministic_unit_sample(sp_idx, load_idx, sample_idx,
                                      8 + component * 2);
  REAL u2 = deterministic_unit_sample(sp_idx, load_idx, sample_idx,
                                      9 + component * 2);
  if (u1 < 1e-12)
    u1 = 1e-12;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWOPI * u2);
}

static REAL wrap_to_interval(REAL value, REAL start, REAL end) {
  REAL length = end - start;
  if (length <= 0.0)
    return start;
  REAL wrapped = std::fmod(value - start, length);
  if (wrapped < 0.0)
    wrapped += length;
  REAL result = start + wrapped;
  if (result >= end)
    result = std::nextafter(end, start);
  return result;
}

static int quiet_layout_x_count(int particles_per_cell) {
  int guess = static_cast<int>(std::floor(std::sqrt(particles_per_cell)));
  for (int div = guess; div >= 1; div--)
    if (particles_per_cell % div == 0)
      return div;
  return 1;
}

static int quiet_particles_per_cell(const Species_load &ld) {
  double particles_per_cell = ld.density_full * grid.dx * grid.dy / np2c_global;
  long long rounded = static_cast<long long>(std::llround(particles_per_cell));
  double tol =
      std::max(1e-6, 1e-3 * std::max(1.0, fabs(static_cast<double>(rounded))));
  if (rounded <= 0 ||
      fabs(particles_per_cell - static_cast<double>(rounded)) > tol) {
    std::cerr << "ERROR: quiet_displacement requires an integer particles-per-cell "
              << "load. density=" << ld.density_full
              << ", particle_weight=" << np2c_global
              << ", dx=" << grid.dx << ", dy=" << grid.dy
              << ", particles_per_cell=" << particles_per_cell << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }
  return static_cast<int>(rounded);
}

static void store_loaded_particle(int sp_idx, const Particle &p) {
  int cell_x = static_cast<int>(p.x);
  int cell_y = static_cast<int>(p.y);
  if (cell_x < 0 || cell_x >= grid.Ncx || cell_y < 0 || cell_y >= grid.Ncy) {
    std::cerr << "ERROR: Initial particle mapped outside local cells for species "
              << species[sp_idx].name << " on rank " << rmpi << ": local=("
              << p.x << ", " << p.y << "), grid=(" << grid.Ncx << ", "
              << grid.Ncy << ")\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }
  int cell_i = cell_y + cell_x * grid.Ncy;
  cells[cell_i].part[sp_idx].push_back(p);
  species[sp_idx].n += 1;
}

static void initial_particle_load_landau_quiet(int sp_idx, int load_idx,
                                               Species &sp, Species_load &ld) {
  double global_x_origin = grid.x1s - grid.N0x * grid.dx;
  double global_y_origin = grid.x2s - grid.N0y * grid.dy;
  int cell_x0_global = static_cast<int>(
      std::llround((ld.x1s_full - global_x_origin) / grid.dx));
  int cell_x1_global = static_cast<int>(
      std::llround((ld.x1f_full - global_x_origin) / grid.dx));
  int cell_y0_global = static_cast<int>(
      std::llround((ld.x2s_full - global_y_origin) / grid.dy));
  int cell_y1_global = static_cast<int>(
      std::llround((ld.x2f_full - global_y_origin) / grid.dy));
  int particles_per_cell = quiet_particles_per_cell(ld);
  int layout_x = quiet_layout_x_count(particles_per_cell);
  int layout_y = particles_per_cell / layout_x;
  bool perturb_density_x =
      (sp.name == "electron" && LandauAlpha != 0.0 && LandauModeIndex > 0);
  bool perturb_density_y =
      (sp.name == "electron" && LandauAlphaY != 0.0 && LandauModeIndexY > 0);
  double kx = 0.0;
  double ky = 0.0;
  if (perturb_density_x) {
    double load_length_x = ld.x1f_full - ld.x1s_full;
    if (load_length_x <= 0.0) {
      std::cerr << "ERROR: quiet_displacement requires positive x-length for "
                << "species " << sp.name << "\n";
      MPI_Abort(comm, EXIT_FAILURE);
    }
    kx = TWOPI * LandauModeIndex / load_length_x;
  }
  if (perturb_density_y) {
    double load_length_y = ld.x2f_full - ld.x2s_full;
    if (load_length_y <= 0.0) {
      std::cerr << "ERROR: quiet_displacement requires positive y-length for "
                << "species " << sp.name << "\n";
      MPI_Abort(comm, EXIT_FAILURE);
    }
    ky = TWOPI * LandauModeIndexY / load_length_y;
  }

  for (int cell_x = cell_x0_global; cell_x < cell_x1_global; cell_x++)
    for (int cell_y = cell_y0_global; cell_y < cell_y1_global; cell_y++) {
      int local_cell_x = cell_x - cell_x0_global;
      int local_cell_y = cell_y - cell_y0_global;
      int cell_linear =
          local_cell_x * (cell_y1_global - cell_y0_global) + local_cell_y;
      for (int qx = 0; qx < layout_x; qx++)
        for (int qy = 0; qy < layout_y; qy++) {
          int particle_idx = qx * layout_y + qy;
          int sample_idx = cell_linear * particles_per_cell + particle_idx;

          double x_m =
              global_x_origin + (cell_x + (qx + 0.5) / layout_x) * grid.dx;
          double y_m =
              global_y_origin + (cell_y + (qy + 0.5) / layout_y) * grid.dy;

          if (perturb_density_x) {
            double phase_x = x_m - ld.x1s_full;
            x_m -= (LandauAlpha / kx) * std::sin(kx * phase_x);
            x_m = wrap_to_interval(x_m, ld.x1s_full, ld.x1f_full);
          }
          if (perturb_density_y) {
            double phase_y = y_m - ld.x2s_full;
            y_m -= (LandauAlphaY / ky) * std::sin(ky * phase_y);
            y_m = wrap_to_interval(y_m, ld.x2s_full, ld.x2f_full);
          }

          if (x_m < grid.x1s || x_m >= grid.x1f || y_m < grid.x2s ||
              y_m >= grid.x2f)
            continue;

          Particle p;
          p.x = (x_m - global_x_origin) / grid.dx - grid.N0x;
          p.y = (y_m - global_y_origin) / grid.dy - grid.N0y;
          p.z = 0.0;
          p.vx = ld.v1drift +
                 ld.v1thermal *
                     deterministic_gaussian_sample(sp_idx, load_idx, sample_idx, 0);
          p.vy = ld.v2drift +
                 ld.v2thermal *
                     deterministic_gaussian_sample(sp_idx, load_idx, sample_idx, 1);
          p.vz = ld.v3drift +
                 ld.v3thermal *
                     deterministic_gaussian_sample(sp_idx, load_idx, sample_idx, 2);
          store_loaded_particle(sp_idx, p);
        }
    }
}

static bool periodic_x_enabled() {
  return grid.boundaries.size() > 1 && grid.boundaries[0].name == "Periodic" &&
         grid.boundaries[1].name == "Periodic";
}

static bool periodic_y_enabled() {
  return grid.boundaries.size() > 3 && grid.boundaries[2].name == "Periodic" &&
         grid.boundaries[3].name == "Periodic";
}

static int effective_deposition_mode() {
  if (deposition_mode < kDepositionModeCIC ||
      deposition_mode > kDepositionModeCICPlusFilter)
    return kDepositionModeCIC;
  return deposition_mode;
}

static hexapic::deposition_action::ActionId active_deposition_action() {
  using hexapic::deposition_action::ActionId;
  ActionId action = static_cast<ActionId>(deposition_action);
  return hexapic::deposition_action::is_enabled(action) ? action
                                                        : ActionId::kDisabled;
}

static int wrap_idx(int idx, int size) {
  if (size <= 0)
    return 0;
  int out = idx % size;
  if (out < 0)
    out += size;
  return out;
}

static int periodic_span_x() {
  if (!periodic_x_enabled())
    return grid.Nxg;
  return std::max(grid.Nxg - 1, 1);
}

static int periodic_span_y() {
  if (!periodic_y_enabled())
    return grid.Nyg;
  return std::max(grid.Nyg - 1, 1);
}

static REAL clamp_cell_offset(REAL value) {
  REAL eps = static_cast<REAL>(1e-12);
  if (value < 0.0)
    return 0.0;
  if (value >= 1.0)
    return static_cast<REAL>(1.0) - eps;
  return value;
}

static void normalize_weights(REAL *weights, int count) {
  REAL sum = 0.0;
  for (int i = 0; i < count; i++) {
    if (weights[i] < 0.0 && std::abs(weights[i]) < static_cast<REAL>(1e-18))
      weights[i] = 0.0;
    sum += weights[i];
  }
  if (sum <= 0.0) {
    REAL uniform = static_cast<REAL>(1.0) / static_cast<REAL>(count);
    for (int i = 0; i < count; i++)
      weights[i] = uniform;
    return;
  }
  REAL inv = static_cast<REAL>(1.0) / sum;
  for (int i = 0; i < count; i++)
    weights[i] *= inv;
}

static bool map_global_node_index(int ii_global, int jj_global, int &out_idx) {
  int ii = ii_global;
  int jj = jj_global;
  if (periodic_x_enabled())
    ii = wrap_idx(ii, periodic_span_x());
  else if (ii < 0 || ii >= grid.Nxg)
    return false;

  if (periodic_y_enabled())
    jj = wrap_idx(jj, periodic_span_y());
  else if (jj < 0 || jj >= grid.Nyg)
    return false;

  out_idx = ii * grid.Nyg + jj;
  return true;
}

static void compute_cic_weights(REAL px, REAL py, REAL *weights) {
  weights[0] = (1.0 - px) * (1.0 - py);
  weights[1] = (1.0 - px) * py;
  weights[2] = px * (1.0 - py);
  weights[3] = px * py;
}

static void compute_gaussian_lookup_weights(REAL px, REAL py, int mode,
                                            REAL *weights,
                                            int &stencil_radius) {
  Phase1Lookup *lookup = nullptr;
  std::string table_path;
  int stencil_size = 0;
  bool *warned = nullptr;
  const char *mode_name = "gaussian";

  if (mode == kDepositionModeGaussian5x5) {
    lookup = &compact_lookup_5x5;
    table_path = compact_lookup_table_5x5;
    stencil_size = kStencilSize5;
    warned = &warned_lookup_5x5;
    mode_name = "gaussian_5x5";
  } else if (mode == kDepositionModeGaussian7x7) {
    lookup = &compact_lookup_7x7;
    table_path = compact_lookup_table_7x7;
    stencil_size = kStencilSize7;
    warned = &warned_lookup_7x7;
    mode_name = "gaussian_7x7";
  } else {
    std::cerr << "ERROR: unsupported Gaussian lookup deposition mode " << mode
              << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }

  if (!lookup->load_binary(table_path, stencil_size)) {
    if (!*warned && !rmpi)
      std::cerr << "ERROR: failed to load " << mode_name
                << " lookup table: " << table_path << "\n";
    *warned = true;
    MPI_Abort(comm, EXIT_FAILURE);
  }

  stencil_radius = stencil_size / 2;
  for (int i = 0; i < kMaxGaussianLookupWeights; i++)
    weights[i] = 0.0;
  if (!lookup->sample_trilinear(px - static_cast<REAL>(0.5),
                                py - static_cast<REAL>(0.5),
                                kGaussianLookupSigmaOverDx, weights)) {
    std::cerr << "ERROR: failed to sample " << mode_name
              << " lookup weights at px=" << px << ", py=" << py
              << ", sigma/dx=" << kGaussianLookupSigmaOverDx << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }
  normalize_weights(weights, stencil_size * stencil_size);
}

static bool map_stencil_values_to_nodes_global(int i0_global, int j0_global,
                                               const REAL *values,
                                               int stencil_radius,
                                               int *idx_out, REAL *v_out,
                                               int &count_out,
                                               bool renormalize) {
  int stencil_size = 2 * stencil_radius + 1;
  count_out = 0;
  REAL kept_sum = 0.0;
  for (int oy = -stencil_radius; oy <= stencil_radius; oy++) {
    int row = oy + stencil_radius;
    for (int ox = -stencil_radius; ox <= stencil_radius; ox++) {
      int col = ox + stencil_radius;
      REAL value = values[row * stencil_size + col];
      if (value == 0.0)
        continue;
      int idx = -1;
      if (!map_global_node_index(i0_global + ox, j0_global + oy, idx))
        continue;
      kept_sum += value;
      bool found = false;
      for (int k = 0; k < count_out; k++)
        if (idx_out[k] == idx) {
          v_out[k] += value;
          found = true;
          break;
        }
      if (!found) {
        idx_out[count_out] = idx;
        v_out[count_out] = value;
        count_out++;
      }
    }
  }
  if (kept_sum <= 0.0 || count_out == 0)
    return false;
  if (renormalize) {
    REAL inv = static_cast<REAL>(1.0) / kept_sum;
    for (int k = 0; k < count_out; k++)
      v_out[k] *= inv;
  }
  return true;
}

static bool map_stencil_weights_to_nodes_global(int i0_global, int j0_global,
                                                const REAL *weights,
                                                int stencil_radius,
                                                int *idx_out, REAL *w_out,
                                                int &count_out) {
  return map_stencil_values_to_nodes_global(i0_global, j0_global, weights,
                                            stencil_radius, idx_out, w_out,
                                            count_out, true);
}

static bool split_particle_position_global(REAL x_global, REAL y_global,
                                           int &i0_global, int &j0_global,
                                           REAL &px, REAL &py) {
  REAL eps = static_cast<REAL>(1e-12);

  if (periodic_x_enabled()) {
    REAL span = static_cast<REAL>(periodic_span_x());
    x_global = std::fmod(x_global, span);
    if (x_global < 0.0)
      x_global += span;
  } else {
    REAL xmax =
        std::max(static_cast<REAL>(grid.Nxg - 1) - eps, static_cast<REAL>(0.0));
    x_global = std::min(std::max(x_global, static_cast<REAL>(0.0)), xmax);
  }

  if (periodic_y_enabled()) {
    REAL span = static_cast<REAL>(periodic_span_y());
    y_global = std::fmod(y_global, span);
    if (y_global < 0.0)
      y_global += span;
  } else {
    REAL ymax =
        std::max(static_cast<REAL>(grid.Nyg - 1) - eps, static_cast<REAL>(0.0));
    y_global = std::min(std::max(y_global, static_cast<REAL>(0.0)), ymax);
  }

  i0_global = static_cast<int>(std::floor(x_global));
  j0_global = static_cast<int>(std::floor(y_global));
  if (periodic_x_enabled())
    i0_global = wrap_idx(i0_global, periodic_span_x());
  else
    i0_global = std::min(std::max(i0_global, 0), std::max(grid.Nxg - 2, 0));
  if (periodic_y_enabled())
    j0_global = wrap_idx(j0_global, periodic_span_y());
  else
    j0_global = std::min(std::max(j0_global, 0), std::max(grid.Nyg - 2, 0));

  px = clamp_cell_offset(x_global - static_cast<REAL>(i0_global));
  py = clamp_cell_offset(y_global - static_cast<REAL>(j0_global));
  return true;
}

static REAL wrap_periodic_coordinate(REAL coord, int n_cells, bool &wrapped) {
  REAL wrapped_coord = std::fmod(coord, static_cast<REAL>(n_cells));
  if (wrapped_coord < 0.0)
    wrapped_coord += static_cast<REAL>(n_cells);
  if (wrapped_coord >= static_cast<REAL>(n_cells))
    wrapped_coord = std::nextafter(static_cast<REAL>(n_cells), 0.0);
  if (fabs(wrapped_coord - coord) > 10 * std::numeric_limits<REAL>::epsilon())
    wrapped = true;
  return wrapped_coord;
}

static int boundary_direction_from_global(REAL xg, REAL yg) {
  int nx_global = grid.Nxg - 1;
  int ny_global = grid.Nyg - 1;
  bool xmin = xg < 0.0;
  bool xmax = xg >= nx_global;
  bool ymin = yg < 0.0;
  bool ymax = yg >= ny_global;

  if (xmin && ymin)
    return 4;
  if (xmin && ymax)
    return 5;
  if (xmax && ymin)
    return 6;
  if (xmax && ymax)
    return 7;
  if (xmin)
    return 0;
  if (xmax)
    return 1;
  if (ymin)
    return 2;
  return 3;
}

static int owner_rank_from_global_position(REAL xg, REAL yg) {
  for (int rank = 0; rank < nmpi; rank++) {
    const std::array<int, 4> &box = particle_rank_boxes[rank];
    if (box[2] <= 0 || box[3] <= 0)
      continue;
    if (xg >= box[0] && xg < box[0] + box[2] && yg >= box[1] &&
        yg < box[1] + box[3])
      return rank;
  }
  return -1;
}

static void enqueue_particle_for_rank(int sp, int rank, const Particle &p_global) {
  grid.neighbours.send_data_3d[sp][rank].insert(
      grid.neighbours.send_data_3d[sp][rank].end(),
      {p_global.x, p_global.y, p_global.z, p_global.vx, p_global.vy,
       p_global.vz});
}

static void cell_init(int i, int j) {
  int index = i * grid.Ncy + j;

  // set position in local grid
  cells[index].n0x = i;
  cells[index].n0y = j;
  // empty particle vector
  cells[index].part = {};
  for (int k = 0; k < species.size(); k++)
    cells[index].part.push_back({});

  // add neighbouring cells
  std::vector<int> neighb_cells = {index - grid.Ncy,
                                   index + grid.Ncy,
                                   index - 1,
                                   index + 1,
                                   index - grid.Ncy - 1,
                                   index - grid.Ncy + 1,
                                   index + grid.Ncy - 1,
                                   index + grid.Ncy + 1};
  cells[index].neighbours = neighb_cells;
}

static void cells_init_interior() {
  cells_interior.reserve((grid.Ncx - 2) * (grid.Ncy - 2));
  for (int i = 1; i < grid.Ncx - 1; i++)
    for (int j = 1; j < grid.Ncy - 1; j++) {
      cell_init(i, j);
      int index = i * grid.Ncy + j;
      cells_interior.push_back(
          &cells[index]); // create pointer to interior cell
    }
}

static void cells_init_edge() {
  int i, j, k;
  cells_edge.reserve(2 * grid.Ncx + 2 * grid.Ncy - 4);
  // bottom edge
  for (j = 0; j < grid.Ncy; j++) {
    cell_init(0, j);
    cells_edge.push_back(&cells[j]); // create pointer to edge cell
  }
  // top edge
  for (j = 0; j < grid.Ncy; j++) {
    cell_init(grid.Ncx - 1, j);
    cells_edge.push_back(
        &cells[(grid.Ncx - 1) * grid.Ncy + j]); // create pointer to edge cell
  }
  // left edge
  for (i = 1; i < grid.Ncx - 1; i++) {
    cell_init(i, 0);
    cells_edge.push_back(&cells[i * grid.Ncy]); // create pointer to edge cell
  }
  // right edge
  for (i = 1; i < grid.Ncx - 1; i++) {
    cell_init(i, grid.Ncy - 1);
    cells_edge.push_back(
        &cells[i * grid.Ncy + grid.Ncy - 1]); // create pointer to edge cell
  }
  // add connections to other ranks as: -(r + 1)
  Neighbours &g_neigh = grid.neighbours;
  for (i = 0; i < grid.Ncy; i++) {
    j = grid.Ncy * (grid.Ncx - 1) + i;
    cells[i].neighbours[0] = -(g_neigh.rxmin[i] + 1);
    cells[j].neighbours[1] = -(g_neigh.rxmax[i] + 1);
    if (i > 0) {
      cells[i].neighbours[4] = -(g_neigh.rxmin[i - 1] + 1);
      cells[j].neighbours[6] = -(g_neigh.rxmax[i - 1] + 1);
    } else {
      cells[i].neighbours[4] = -(g_neigh.corners[0] + 1);
      cells[j].neighbours[6] = -(g_neigh.corners[2] + 1);
    }
    if (i < grid.Ncy - 1) {
      cells[i].neighbours[5] = -(g_neigh.rxmin[i + 1] + 1);
      cells[j].neighbours[7] = -(g_neigh.rxmax[i + 1] + 1);
    } else {
      cells[i].neighbours[5] = -(g_neigh.corners[1] + 1);
      cells[j].neighbours[7] = -(g_neigh.corners[3] + 1);
    }
  }
  for (i = 0; i < grid.Ncx; i++) {
    j = i * grid.Ncy;
    k = (i + 1) * grid.Ncy - 1;
    cells[j].neighbours[2] = -(g_neigh.rymin[i] + 1);
    cells[k].neighbours[3] = -(g_neigh.rymax[i] + 1);
    if (i > 0) {
      cells[j].neighbours[4] = -(g_neigh.rymin[i - 1] + 1);
      cells[k].neighbours[5] = -(g_neigh.rymax[i - 1] + 1);
    }
    if (i < grid.Ncx - 1) {
      cells[j].neighbours[6] = -(g_neigh.rymin[i + 1] + 1);
      cells[k].neighbours[7] = -(g_neigh.rymax[i + 1] + 1);
    }
  }
}

void cells_init() {
  cells = {};
  cells.resize(grid.Ncx * grid.Ncy); // allocate once so pointers are stable
  cells_init_interior();
  cells_init_edge();
}

void grid_init(int argc, char **args) {

  int i, j;
  grid.sort_boundaries();

  dV = grid.dx * grid.dy * 1; ///< cell volume
  grid.p2n = np2c_global / dV;
  Sconst = -q_e * grid.p2n / EPS0; ///< particle to charge density
  Sconst *= grid.dx * grid.dx;     ///< from multiplication of matrix A
  hdt = dt / 2;                    ///< half time step
  // number of cells in each direction

  // allocate memory for meshes

  int n_corners = grid.Nx * grid.Ny;
  rho.reserve(n_corners); // computed at cell corners
  grid.V_store.reserve(n_corners);

  for (i = 0; i < n_corners; i++) {
    rho.push_back(0);
    grid.V_store.push_back(0);
  }

  int n_cells = grid.Ncx * grid.Ncy;
  for (auto &sp : species) {
    sp.n_store.reserve(n_cells); ///< computed at cell center
    sp.T_store.reserve(n_cells);
    sp.v_store.reserve(n_cells);
    for (i = 0; i < n_cells; i++) {
      sp.n_store.push_back(0);
      sp.T_store.push_back(0);
      sp.v_store.push_back(0);
    }
  }
  t_store.reserve(1); ///< time
  t_store.push_back(0);
}

void inject_particle(int sp, REAL x, REAL y) {
  Particle p;
  if ((!x) && (!y)) {
    p.x = (grid.Nx - 2) * frand() + 1;
    p.y = (grid.Ny - 2) * frand() + 1;
  } else {
    p.x = x;
    p.y = y;
  }
  p.z = 0;
  p.vx = 0; ///< (10+frand())/20;
  p.vy = 0; ///< (10+frand())/20;
  p.vz = 0; ///< (10+frand())/20;
  int cell_i = int(p.y) + int(p.x) * grid.Ncy;
  species[sp].n += 1;
  cells[cell_i].part[sp].push_back(p);
}

static void part2grid_cic_cell(Cell &cell, bool edge_cell) {
  int i = cell.n0x;
  int j = cell.n0y;
  int J = i * grid.Ny + j;

  for (int k = 0; k < species.size(); k++) {
    Species &sp = species[k];
    int q = sp.charge;
    if (q == 0)
      continue;
    double physical_charge = np2c_global * sp.charge * q_e;
    for (auto &p : cell.part[k]) {
      REAL px = clamp_cell_offset(p.x - i);
      REAL py = clamp_cell_offset(p.y - j);
      REAL weights[4];
      compute_cic_weights(px, py, weights);
      rho[J] += q * weights[0];
      rho[J + 1] += q * weights[1];
      rho[J + grid.Ny] += q * weights[2];
      rho[J + grid.Ny + 1] += q * weights[3];
      if (edge_cell) {
        interface_deposition_edge_charge_step[k] += physical_charge;
        interface_deposition_edge_abs_charge_step[k] += fabs(physical_charge);
        interface_deposition_edge_particles_step[k] += 1;
      } else {
        interface_deposition_interior_charge_step[k] += physical_charge;
        interface_deposition_interior_abs_charge_step[k] += fabs(physical_charge);
        interface_deposition_interior_particles_step[k] += 1;
      }
    }
  }
}

static void apply_cic_filter_global(std::vector<double> &values) {
  if (values.empty())
    return;

  int nx = periodic_span_x();
  int ny = periodic_span_y();
  std::vector<double> tmp(values.size(), 0.0);
  std::vector<double> filtered(values.size(), 0.0);

  auto filter_axis = [&](const std::vector<double> &src, std::vector<double> &dst,
                         bool axis_x) {
    for (int ii = 0; ii < nx; ii++)
      for (int jj = 0; jj < ny; jj++) {
        int center_idx = -1;
        if (!map_global_node_index(ii, jj, center_idx))
          continue;

        double accum = 0.0;
        double norm = 0.0;
        for (int offset = -1; offset <= 1; offset++) {
          double weight = (offset == 0) ? 0.5 : 0.25;
          int ni = ii;
          int nj = jj;
          if (axis_x)
            ni += offset;
          else
            nj += offset;

          if (axis_x && !periodic_x_enabled() && (ni < 0 || ni >= nx))
            continue;
          if (!axis_x && !periodic_y_enabled() && (nj < 0 || nj >= ny))
            continue;

          int src_idx = -1;
          if (!map_global_node_index(ni, nj, src_idx))
            continue;
          accum += weight * src[static_cast<size_t>(src_idx)];
          norm += weight;
        }
        if (norm > 0.0)
          dst[static_cast<size_t>(center_idx)] = accum / norm;
      }
  };

  filter_axis(values, tmp, true);
  filter_axis(tmp, filtered, false);
  values.swap(filtered);
}

static void apply_action_filter_global(
    std::vector<double> &values,
    hexapic::deposition_action::ActionId action) {
  std::vector<double> kernel;
  hexapic::deposition_action::fill_filter_kernel_1d(action, kernel);
  hexapic::deposition_action::apply_separable_filter(
      values, periodic_span_x(), periodic_span_y(), grid.Nyg,
      periodic_x_enabled(), periodic_y_enabled(), kernel);
}

static REAL node_potential_global(int ii_global, int jj_global);
static REAL node_ex_global(int ii_global, int jj_global);
static REAL node_ey_global(int ii_global, int jj_global);

static void ensure_compact_global_potential() {
  if (compact_phi_tstep == tstep)
    return;

  size_t total_nodes =
      static_cast<size_t>(grid.Nxg) * static_cast<size_t>(grid.Nyg);
  if (compact_phi_send.size() != total_nodes)
    compact_phi_send.assign(total_nodes, 0.0);
  if (compact_phi_sum.size() != total_nodes)
    compact_phi_sum.assign(total_nodes, 0.0);
  if (compact_ex_sum.size() != total_nodes)
    compact_ex_sum.assign(total_nodes, 0.0);
  if (compact_ey_sum.size() != total_nodes)
    compact_ey_sum.assign(total_nodes, 0.0);
  std::fill(compact_phi_send.begin(), compact_phi_send.end(), 0.0);

  for (int ix = 0; ix < hypre.nx; ix++)
    for (int iy = 0; iy < hypre.ny; iy++) {
      int global_idx = -1;
      if (!map_global_node_index(grid.N0x + ix, grid.N0y + iy, global_idx))
        continue;
      compact_phi_send[static_cast<size_t>(global_idx)] =
          V_local[ix * grid.Ny + iy];
    }

  MPI_Allreduce(compact_phi_send.data(), compact_phi_sum.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);

  for (int ii = 0; ii < grid.Nxg; ++ii) {
    for (int jj = 0; jj < grid.Nyg; ++jj) {
      int idx = ii * grid.Nyg + jj;

      REAL phi_xm = node_potential_global(ii - 1, jj);
      REAL phi_xp = node_potential_global(ii + 1, jj);
      REAL phi_ym = node_potential_global(ii, jj - 1);
      REAL phi_yp = node_potential_global(ii, jj + 1);

      if (!periodic_x_enabled()) {
        if (ii <= 0) {
          phi_xm = node_potential_global(ii, jj);
          phi_xp = node_potential_global(ii + 1, jj);
          compact_ex_sum[static_cast<size_t>(idx)] =
              -(phi_xp - phi_xm) / grid.dx;
        } else if (ii >= grid.Nxg - 1) {
          phi_xm = node_potential_global(ii - 1, jj);
          phi_xp = node_potential_global(ii, jj);
          compact_ex_sum[static_cast<size_t>(idx)] =
              -(phi_xp - phi_xm) / grid.dx;
        } else {
          compact_ex_sum[static_cast<size_t>(idx)] =
              -(phi_xp - phi_xm) / (static_cast<REAL>(2.0) * grid.dx);
        }
      } else {
        compact_ex_sum[static_cast<size_t>(idx)] =
            -(phi_xp - phi_xm) / (static_cast<REAL>(2.0) * grid.dx);
      }

      if (!periodic_y_enabled()) {
        if (jj <= 0) {
          phi_ym = node_potential_global(ii, jj);
          phi_yp = node_potential_global(ii, jj + 1);
          compact_ey_sum[static_cast<size_t>(idx)] =
              -(phi_yp - phi_ym) / grid.dy;
        } else if (jj >= grid.Nyg - 1) {
          phi_ym = node_potential_global(ii, jj - 1);
          phi_yp = node_potential_global(ii, jj);
          compact_ey_sum[static_cast<size_t>(idx)] =
              -(phi_yp - phi_ym) / grid.dy;
        } else {
          compact_ey_sum[static_cast<size_t>(idx)] =
              -(phi_yp - phi_ym) / (static_cast<REAL>(2.0) * grid.dy);
        }
      } else {
        compact_ey_sum[static_cast<size_t>(idx)] =
            -(phi_yp - phi_ym) / (static_cast<REAL>(2.0) * grid.dy);
      }
    }
  }
  compact_phi_tstep = tstep;
}

static REAL node_potential_global(int ii_global, int jj_global) {
  int global_idx = -1;
  if (!map_global_node_index(ii_global, jj_global, global_idx))
    return 0.0;
  return static_cast<REAL>(compact_phi_sum[static_cast<size_t>(global_idx)]);
}

static REAL node_ex_global(int ii_global, int jj_global) {
  int global_idx = -1;
  if (!map_global_node_index(ii_global, jj_global, global_idx))
    return 0.0;
  return static_cast<REAL>(compact_ex_sum[static_cast<size_t>(global_idx)]);
}

static REAL node_ey_global(int ii_global, int jj_global) {
  int global_idx = -1;
  if (!map_global_node_index(ii_global, jj_global, global_idx))
    return 0.0;
  return static_cast<REAL>(compact_ey_sum[static_cast<size_t>(global_idx)]);
}

static REAL interpolate_potential_compact_global(REAL x_global, REAL y_global,
                                                 int mode) {
  int i0_global = 0;
  int j0_global = 0;
  REAL px = 0.0;
  REAL py = 0.0;
  if (!split_particle_position_global(x_global, y_global, i0_global, j0_global,
                                      px, py))
    return 0.0;

  int stencil_radius = (mode == kDepositionModeGaussian5x5) ? kStencilRadius5
                                                           : kStencilRadius7;
  REAL weights[kMaxGaussianLookupWeights];
  compute_gaussian_lookup_weights(px, py, mode, weights, stencil_radius);

  int idx[kMaxGaussianLookupWeights];
  REAL vals[kMaxGaussianLookupWeights];
  int count = 0;
  if (!map_stencil_weights_to_nodes_global(i0_global, j0_global, weights,
                                           stencil_radius, idx, vals, count))
    return 0.0;

  REAL phi = 0.0;
  for (int k = 0; k < count; k++) {
    int ii = idx[k] / grid.Nyg;
    int jj = idx[k] % grid.Nyg;
    phi += vals[k] * node_potential_global(ii, jj);
  }
  return phi;
}

static REAL interpolate_potential_action_global(
    REAL x_global, REAL y_global,
    hexapic::deposition_action::ActionId action) {
  int i0_global = 0;
  int j0_global = 0;
  REAL px = 0.0;
  REAL py = 0.0;
  if (!split_particle_position_global(x_global, y_global, i0_global, j0_global,
                                      px, py))
    return 0.0;

  REAL weights[kMaxGaussianLookupWeights];
  std::fill(weights, weights + kMaxGaussianLookupWeights, 0.0);
  int stencil_radius = 0;
  try {
    hexapic::deposition_action::fill_direct_stencil_weights(
        action, static_cast<double>(px), static_cast<double>(py), weights,
        kMaxGaussianLookupWeights, stencil_radius,
        static_cast<double>(deposition_sigma_cells));
  } catch (const std::exception &ex) {
    if (!rmpi)
      std::cerr << "ERROR: action gather failed: " << ex.what() << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }

  int idx[kMaxGaussianLookupWeights];
  REAL vals[kMaxGaussianLookupWeights];
  int count = 0;
  if (!map_stencil_weights_to_nodes_global(i0_global, j0_global, weights,
                                           stencil_radius, idx, vals, count))
    return 0.0;

  REAL phi = 0.0;
  for (int k = 0; k < count; k++) {
    int ii = idx[k] / grid.Nyg;
    int jj = idx[k] % grid.Nyg;
    phi += vals[k] * node_potential_global(ii, jj);
  }
  return phi;
}

static void gather_field_compact(int i0, int j0, REAL px, REAL py, int mode,
                                 REAL *Ef) {
  ensure_compact_global_potential();
  int i0_global = grid.N0x + i0;
  int j0_global = grid.N0y + j0;
  int stencil_radius = (mode == kDepositionModeGaussian5x5) ? kStencilRadius5
                                                           : kStencilRadius7;
  REAL weights[kMaxGaussianLookupWeights];
  compute_gaussian_lookup_weights(px, py, mode, weights, stencil_radius);

  int idx[kMaxGaussianLookupWeights];
  REAL vals[kMaxGaussianLookupWeights];
  int count = 0;
  if (!map_stencil_weights_to_nodes_global(i0_global, j0_global, weights,
                                           stencil_radius, idx, vals, count)) {
    Ef[0] = 0.0;
    Ef[1] = 0.0;
    return;
  }

  Ef[0] = 0.0;
  Ef[1] = 0.0;
  for (int k = 0; k < count; ++k) {
    int ii = idx[k] / grid.Nyg;
    int jj = idx[k] % grid.Nyg;
    Ef[0] += vals[k] * node_ex_global(ii, jj);
    Ef[1] += vals[k] * node_ey_global(ii, jj);
  }
}

static void gather_field_action(int i0, int j0, REAL px, REAL py,
                                hexapic::deposition_action::ActionId action,
                                REAL *Ef) {
  ensure_compact_global_potential();
  int i0_global = grid.N0x + i0;
  int j0_global = grid.N0y + j0;
  REAL weights[kMaxGaussianLookupWeights];
  REAL dweights_dpx[kMaxGaussianLookupWeights];
  REAL dweights_dpy[kMaxGaussianLookupWeights];
  std::fill(weights, weights + kMaxGaussianLookupWeights, 0.0);
  std::fill(dweights_dpx, dweights_dpx + kMaxGaussianLookupWeights, 0.0);
  std::fill(dweights_dpy, dweights_dpy + kMaxGaussianLookupWeights, 0.0);
  int stencil_radius = 0;
  try {
    hexapic::deposition_action::fill_direct_stencil_weights_and_derivatives(
        action, static_cast<double>(px), static_cast<double>(py), weights,
        dweights_dpx, dweights_dpy, kMaxGaussianLookupWeights, stencil_radius,
        static_cast<double>(deposition_sigma_cells));
  } catch (const std::exception &ex) {
    if (!rmpi)
      std::cerr << "ERROR: action gather failed: " << ex.what() << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }

  int idx_x[kMaxGaussianLookupWeights];
  int idx_y[kMaxGaussianLookupWeights];
  REAL dpx_vals[kMaxGaussianLookupWeights];
  REAL dpy_vals[kMaxGaussianLookupWeights];
  int count_x = 0;
  int count_y = 0;
  if (!map_stencil_values_to_nodes_global(i0_global, j0_global, dweights_dpx,
                                          stencil_radius, idx_x, dpx_vals,
                                          count_x, false) ||
      !map_stencil_values_to_nodes_global(i0_global, j0_global, dweights_dpy,
                                          stencil_radius, idx_y, dpy_vals,
                                          count_y, false)) {
    Ef[0] = 0.0;
    Ef[1] = 0.0;
    return;
  }

  Ef[0] = 0.0;
  Ef[1] = 0.0;
  for (int k = 0; k < count_x; ++k) {
    int ii = idx_x[k] / grid.Nyg;
    int jj = idx_x[k] % grid.Nyg;
    Ef[0] -= (dpx_vals[k] / grid.dx) * node_potential_global(ii, jj);
  }
  for (int k = 0; k < count_y; ++k) {
    int ii = idx_y[k] / grid.Nyg;
    int jj = idx_y[k] % grid.Nyg;
    Ef[1] -= (dpy_vals[k] / grid.dy) * node_potential_global(ii, jj);
  }
}

static void part2grid_global_nonlocal(int mode) {
  size_t total_nodes =
      static_cast<size_t>(grid.Nxg) * static_cast<size_t>(grid.Nyg);
  if (compact_rho_send.size() != total_nodes)
    compact_rho_send.assign(total_nodes, 0.0);
  if (compact_rho_sum.size() != total_nodes)
    compact_rho_sum.assign(total_nodes, 0.0);
  std::fill(compact_rho_send.begin(), compact_rho_send.end(), 0.0);

  for (size_t cell_index = 0; cell_index < cells.size(); cell_index++) {
    Cell &cell = cells[cell_index];
    bool edge_cell = is_edge(cell.n0x, cell.n0y, grid.Ncx, grid.Ncy);
    int i_global = grid.N0x + cell.n0x;
    int j_global = grid.N0y + cell.n0y;

    for (int sp_i = 0; sp_i < species.size(); sp_i++) {
      Species &sp = species[sp_i];
      int q = sp.charge;
      if (q == 0)
        continue;
      double physical_charge = np2c_global * sp.charge * q_e;

      for (auto &p : cell.part[sp_i]) {
        REAL px = clamp_cell_offset(p.x - cell.n0x);
        REAL py = clamp_cell_offset(p.y - cell.n0y);

        if (mode == kDepositionModeCICPlusFilter) {
          REAL weights[4];
          compute_cic_weights(px, py, weights);
          int target_idx = -1;
          if (map_global_node_index(i_global, j_global, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[0];
          if (map_global_node_index(i_global, j_global + 1, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[1];
          if (map_global_node_index(i_global + 1, j_global, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[2];
          if (map_global_node_index(i_global + 1, j_global + 1, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[3];
        } else {
          REAL weights[kMaxGaussianLookupWeights];
          int stencil_radius = kStencilRadius7;
          compute_gaussian_lookup_weights(px, py, mode, weights,
                                          stencil_radius);

          int idx[kMaxGaussianLookupWeights];
          REAL vals[kMaxGaussianLookupWeights];
          int count = 0;
          if (!map_stencil_weights_to_nodes_global(i_global, j_global, weights,
                                                   stencil_radius, idx, vals,
                                                   count))
            continue;
          for (int k = 0; k < count; k++)
            compact_rho_send[static_cast<size_t>(idx[k])] += q * vals[k];
        }

        if (edge_cell) {
          interface_deposition_edge_charge_step[sp_i] += physical_charge;
          interface_deposition_edge_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_edge_particles_step[sp_i] += 1;
        } else {
          interface_deposition_interior_charge_step[sp_i] += physical_charge;
          interface_deposition_interior_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_interior_particles_step[sp_i] += 1;
        }
      }
    }
  }

  MPI_Allreduce(compact_rho_send.data(), compact_rho_sum.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);
  if (mode == kDepositionModeCICPlusFilter)
    apply_cic_filter_global(compact_rho_sum);

  for (int i = 0; i < grid.Nx; i++)
    for (int j = 0; j < grid.Ny; j++) {
      int local_idx = i * grid.Ny + j;
      int global_idx = -1;
      if (!map_global_node_index(grid.N0x + i, grid.N0y + j, global_idx)) {
        rho[local_idx] = 0.0;
        continue;
      }
      rho[local_idx] = Sconst * compact_rho_sum[static_cast<size_t>(global_idx)];
    }

  compact_skip_rho_halo_exchange = true;
}

static void part2grid_action_library_nonlocal(
    hexapic::deposition_action::ActionId action) {
  size_t total_nodes =
      static_cast<size_t>(grid.Nxg) * static_cast<size_t>(grid.Nyg);
  if (compact_rho_send.size() != total_nodes)
    compact_rho_send.assign(total_nodes, 0.0);
  if (compact_rho_sum.size() != total_nodes)
    compact_rho_sum.assign(total_nodes, 0.0);
  std::fill(compact_rho_send.begin(), compact_rho_send.end(), 0.0);

  bool use_post_filter = hexapic::deposition_action::uses_post_filter(action);
  bool use_direct_stencil =
      hexapic::deposition_action::uses_direct_stencil(action);

  for (size_t cell_index = 0; cell_index < cells.size(); cell_index++) {
    Cell &cell = cells[cell_index];
    bool edge_cell = is_edge(cell.n0x, cell.n0y, grid.Ncx, grid.Ncy);
    int i_global = grid.N0x + cell.n0x;
    int j_global = grid.N0y + cell.n0y;

    for (int sp_i = 0; sp_i < species.size(); sp_i++) {
      Species &sp = species[sp_i];
      int q = sp.charge;
      if (q == 0)
        continue;
      double physical_charge = np2c_global * sp.charge * q_e;

      for (auto &p : cell.part[sp_i]) {
        REAL px = clamp_cell_offset(p.x - cell.n0x);
        REAL py = clamp_cell_offset(p.y - cell.n0y);

        if (use_post_filter) {
          REAL weights[4];
          compute_cic_weights(px, py, weights);
          int target_idx = -1;
          if (map_global_node_index(i_global, j_global, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[0];
          if (map_global_node_index(i_global, j_global + 1, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[1];
          if (map_global_node_index(i_global + 1, j_global, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[2];
          if (map_global_node_index(i_global + 1, j_global + 1, target_idx))
            compact_rho_send[static_cast<size_t>(target_idx)] += q * weights[3];
        } else if (use_direct_stencil) {
          REAL weights[kMaxGaussianLookupWeights];
          std::fill(weights, weights + kMaxGaussianLookupWeights, 0.0);
          int stencil_radius = 0;
          try {
            hexapic::deposition_action::fill_direct_stencil_weights(
                action, static_cast<double>(px), static_cast<double>(py),
                weights, kMaxGaussianLookupWeights, stencil_radius,
                static_cast<double>(deposition_sigma_cells));
          } catch (const std::exception &ex) {
            if (!rmpi)
              std::cerr << "ERROR: safe deposition action failed: " << ex.what()
                        << "\n";
            MPI_Abort(comm, EXIT_FAILURE);
          }

          int idx[kMaxGaussianLookupWeights];
          REAL vals[kMaxGaussianLookupWeights];
          int count = 0;
          if (!map_stencil_weights_to_nodes_global(i_global, j_global, weights,
                                                   stencil_radius, idx, vals,
                                                   count))
            continue;
          for (int k = 0; k < count; k++)
            compact_rho_send[static_cast<size_t>(idx[k])] += q * vals[k];
        }

        if (edge_cell) {
          interface_deposition_edge_charge_step[sp_i] += physical_charge;
          interface_deposition_edge_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_edge_particles_step[sp_i] += 1;
        } else {
          interface_deposition_interior_charge_step[sp_i] += physical_charge;
          interface_deposition_interior_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_interior_particles_step[sp_i] += 1;
        }
      }
    }
  }

  MPI_Allreduce(compact_rho_send.data(), compact_rho_sum.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);
  if (use_post_filter)
    apply_action_filter_global(compact_rho_sum, action);

  for (int i = 0; i < grid.Nx; i++)
    for (int j = 0; j < grid.Ny; j++) {
      int local_idx = i * grid.Ny + j;
      int global_idx = -1;
      if (!map_global_node_index(grid.N0x + i, grid.N0y + j, global_idx)) {
        rho[local_idx] = 0.0;
        continue;
      }
      rho[local_idx] = Sconst * compact_rho_sum[static_cast<size_t>(global_idx)];
    }

  compact_skip_rho_halo_exchange = true;
}

static void accumulate_cic_to_global_send(int i_global, int j_global, REAL px,
                                          REAL py, REAL charge_scale,
                                          std::vector<double> &rho_send) {
  REAL weights[4];
  compute_cic_weights(px, py, weights);
  int target_idx = -1;
  if (map_global_node_index(i_global, j_global, target_idx))
    rho_send[static_cast<size_t>(target_idx)] += charge_scale * weights[0];
  if (map_global_node_index(i_global, j_global + 1, target_idx))
    rho_send[static_cast<size_t>(target_idx)] += charge_scale * weights[1];
  if (map_global_node_index(i_global + 1, j_global, target_idx))
    rho_send[static_cast<size_t>(target_idx)] += charge_scale * weights[2];
  if (map_global_node_index(i_global + 1, j_global + 1, target_idx))
    rho_send[static_cast<size_t>(target_idx)] += charge_scale * weights[3];
}

static void accumulate_direct_action_to_global_send(
    int i_global, int j_global, REAL px, REAL py,
    hexapic::deposition_action::ActionId action,
    REAL charge_scale, std::vector<double> &rho_send) {
  REAL weights[kMaxGaussianLookupWeights];
  std::fill(weights, weights + kMaxGaussianLookupWeights, 0.0);
  int stencil_radius = 0;
  try {
    hexapic::deposition_action::fill_direct_stencil_weights(
        action, static_cast<double>(px), static_cast<double>(py), weights,
        kMaxGaussianLookupWeights, stencil_radius,
        static_cast<double>(deposition_sigma_cells));
  } catch (const std::exception &ex) {
    if (!rmpi)
      std::cerr << "ERROR: controller direct stencil failed: " << ex.what()
                << "\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }

  int idx[kMaxGaussianLookupWeights];
  REAL vals[kMaxGaussianLookupWeights];
  int count = 0;
  if (!map_stencil_weights_to_nodes_global(i_global, j_global, weights,
                                           stencil_radius, idx, vals, count))
    return;
  for (int k = 0; k < count; k++)
    rho_send[static_cast<size_t>(idx[k])] += charge_scale * vals[k];
}

static void part2grid_controller_nonlocal() {
  using hexapic::deposition_action::ActionId;
  size_t total_nodes =
      static_cast<size_t>(grid.Nxg) * static_cast<size_t>(grid.Nyg);

  std::vector<double> rho_send_total(total_nodes, 0.0);
  std::vector<double> rho_sum_total(total_nodes, 0.0);
  std::vector<double> rho_send_filter_weak(total_nodes, 0.0);
  std::vector<double> rho_send_filter_medium(total_nodes, 0.0);
  std::vector<double> rho_sum_filter_weak(total_nodes, 0.0);
  std::vector<double> rho_sum_filter_medium(total_nodes, 0.0);

  for (size_t cell_index = 0; cell_index < cells.size(); cell_index++) {
    Cell &cell = cells[cell_index];
    bool edge_cell = is_edge(cell.n0x, cell.n0y, grid.Ncx, grid.Ncy);
    int i_global = grid.N0x + cell.n0x;
    int j_global = grid.N0y + cell.n0y;
    ActionId action =
        static_cast<ActionId>(controller_action_for_global_cell(i_global, j_global));
    if (!hexapic::deposition_action::is_enabled(action))
      action = ActionId::kCIC;

    for (int sp_i = 0; sp_i < species.size(); sp_i++) {
      Species &sp = species[sp_i];
      int q = sp.charge;
      if (q == 0)
        continue;
      double physical_charge = np2c_global * sp.charge * q_e;

      for (auto &p : cell.part[sp_i]) {
        REAL px = clamp_cell_offset(p.x - cell.n0x);
        REAL py = clamp_cell_offset(p.y - cell.n0y);

        if (hexapic::deposition_action::uses_post_filter(action)) {
          std::vector<double> &target =
              (action == ActionId::kCICPlusFilterMedium) ? rho_send_filter_medium
                                                         : rho_send_filter_weak;
          accumulate_cic_to_global_send(i_global, j_global, px, py, q, target);
        } else if (hexapic::deposition_action::uses_direct_stencil(action)) {
          accumulate_direct_action_to_global_send(i_global, j_global, px, py,
                                                  action, q, rho_send_total);
        } else {
          accumulate_cic_to_global_send(i_global, j_global, px, py, q,
                                        rho_send_total);
        }

        if (edge_cell) {
          interface_deposition_edge_charge_step[sp_i] += physical_charge;
          interface_deposition_edge_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_edge_particles_step[sp_i] += 1;
        } else {
          interface_deposition_interior_charge_step[sp_i] += physical_charge;
          interface_deposition_interior_abs_charge_step[sp_i] +=
              fabs(physical_charge);
          interface_deposition_interior_particles_step[sp_i] += 1;
        }
      }
    }
  }

  MPI_Allreduce(rho_send_total.data(), rho_sum_total.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(rho_send_filter_weak.data(), rho_sum_filter_weak.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(rho_send_filter_medium.data(), rho_sum_filter_medium.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);

  if (std::any_of(rho_sum_filter_weak.begin(), rho_sum_filter_weak.end(),
                  [](double value) { return value != 0.0; })) {
    apply_action_filter_global(rho_sum_filter_weak, ActionId::kCICPlusFilterWeak);
    for (size_t idx = 0; idx < total_nodes; ++idx)
      rho_sum_total[idx] += rho_sum_filter_weak[idx];
  }
  if (std::any_of(rho_sum_filter_medium.begin(), rho_sum_filter_medium.end(),
                  [](double value) { return value != 0.0; })) {
    apply_action_filter_global(rho_sum_filter_medium,
                               ActionId::kCICPlusFilterMedium);
    for (size_t idx = 0; idx < total_nodes; ++idx)
      rho_sum_total[idx] += rho_sum_filter_medium[idx];
  }

  for (int i = 0; i < grid.Nx; i++)
    for (int j = 0; j < grid.Ny; j++) {
      int local_idx = i * grid.Ny + j;
      int global_idx = -1;
      if (!map_global_node_index(grid.N0x + i, grid.N0y + j, global_idx)) {
        rho[local_idx] = 0.0;
        continue;
      }
      rho[local_idx] = Sconst * rho_sum_total[static_cast<size_t>(global_idx)];
    }

  compact_skip_rho_halo_exchange = true;
}

void part2grid_edge() {
  size_t i, j, J;
  int mode = effective_deposition_mode();
  auto action = active_deposition_action();

  std::fill(rho.begin(), rho.end(), 0.0);

  if (controller_runtime_enabled || controller_sigma_runtime_enabled) {
    compact_skip_rho_halo_exchange = true;
    return;
  }

  if (action != hexapic::deposition_action::ActionId::kDisabled &&
      action != hexapic::deposition_action::ActionId::kCIC) {
    compact_skip_rho_halo_exchange = true;
    return;
  }

  if (action == hexapic::deposition_action::ActionId::kDisabled &&
      mode != kDepositionModeCIC) {
    compact_skip_rho_halo_exchange = true;
    return;
  }
  compact_skip_rho_halo_exchange = false;

  for (Cell *p : cells_edge)
    part2grid_cic_cell(*p, true);

  for (i = 0; i < grid.Nx; i++) {
    J = i * grid.Ny;
    rho[J] *= Sconst;
    J += grid.Ny - 1;
    rho[J] *= Sconst;
  }

  for (j = 1; j < grid.Ny - 1; j++) {
    J = j;
    rho[J] *= Sconst;
    J += (grid.Nx - 1) * grid.Ny;
    rho[J] *= Sconst;
  }

  hypre_post_rho_halo();
}

void part2grid_interior() {
  int mode = effective_deposition_mode();
  auto action = active_deposition_action();

  if (controller_runtime_enabled) {
    part2grid_controller_nonlocal();
    return;
  }

  if (controller_sigma_runtime_enabled) {
    hexapic::kde::part2grid_kde_nonlocal();
    return;
  }

  if (hexapic::deposition_action::uses_kde(action)) {
    hexapic::kde::part2grid_kde_nonlocal();
    return;
  }

  if (action != hexapic::deposition_action::ActionId::kDisabled &&
      action != hexapic::deposition_action::ActionId::kCIC) {
    part2grid_action_library_nonlocal(action);
    return;
  }

  if (action == hexapic::deposition_action::ActionId::kDisabled &&
      mode != kDepositionModeCIC) {
    part2grid_global_nonlocal(mode);
    return;
  }
  compact_skip_rho_halo_exchange = false;

  for (Cell *p : cells_interior)
    part2grid_cic_cell(*p, false);

  for (int i = 1; i < grid.Nx - 1; i++)
    for (int j = 1; j < grid.Ny - 1; j++) {
      int J = i * grid.Ny + j;
      rho[J] *= Sconst;
    }
}

void grid2part(REAL Px, REAL Py, double *V, REAL *Ef) {
  /* Map electric field to particle position

J,(i,j)_______px_________J+grid.Ny,(i+1,j)
          |       |          |
          |       |          |
    py|_______|__________|
          |       |          |
          |       |          |
          |       |          |
          |_______|__________|
J+1,(i,j+1)				J+1+grid.Ny,(i+1,j+1) */

  int i, j, J;
  REAL px, py;         // position inside the cell
  i = Px;              // cell index in X direction
  j = Py;              // cell index in Y direction
  px = Px - i;         // X position inside the cell
  py = Py - j;         // Y position inside the cell
  J = i * grid.Ny + j; // global index of cell in linearized mesh

  // The solution of the matrix is the exact potential
  Ef[0] = -((V[J + grid.Ny] - V[J]) * (1 - py) +
            (V[J + grid.Ny + 1] - V[J + 1]) * py) /
          grid.dx;
  Ef[1] = -((V[J + 1] - V[J]) * (1 - px) +
            (V[J + grid.Ny + 1] - V[J + grid.Ny]) * px) /
          grid.dy;
}

inline static void grid2cell(int J, double *V, REAL *V_cell) {
  /* Map electric field to cell nodes
  b-l, t-l, b-r, t-r
  */
  V_cell[0] = V[J];
  V_cell[1] = V[J + 1];
  V_cell[2] = V[J + grid.Ny];
  V_cell[3] = V[J + grid.Ny + 1];
}

inline static void cell2part(const int i, const int j, const REAL px,
                             const REAL py, const REAL *V_cell, REAL *Ef) {
  Ef[0] = -((V_cell[2] - V_cell[0]) * (1 - py) + (V_cell[3] - V_cell[1]) * py) /
          grid.dx;
  Ef[1] = -((V_cell[1] - V_cell[0]) * (1 - px) + (V_cell[3] - V_cell[2]) * px) /
          grid.dy;
}

inline static void particle_mover_boris_cell(Cell &cell) {
  int i, j, k, J;
  REAL V_cell[4], Ef[2], px, py, vx, vy, vz, chom, *sb, *tb;
  int sp, sc;
  using hexapic::deposition_action::ActionId;
  bool halo_stencil = particle_cell_halo_mask[static_cast<size_t>(
                           cell.n0x * grid.Ncy + cell.n0y)] != 0;
  ActionId gather_action = ActionId::kDisabled;
  int gather_mode = kDepositionModeCIC;
  bool use_action_gather = false;
  bool use_compact_mode_gather = false;
  bool use_kde_gather = false;

  i = cell.n0x;
  j = cell.n0y;
  J = j + grid.Ny * i;
  grid2cell(J, V_local, &V_cell[0]);

  if (controller_sigma_runtime_enabled) {
    use_kde_gather = true;
  } else if (controller_runtime_enabled) {
    gather_action = static_cast<ActionId>(
        controller_action_for_global_cell(grid.N0x + i, grid.N0y + j));
    if (!hexapic::deposition_action::is_enabled(gather_action))
      gather_action = ActionId::kCIC;
    use_action_gather =
        hexapic::deposition_action::uses_direct_stencil(gather_action);
  } else {
    gather_action = active_deposition_action();
    if (hexapic::deposition_action::uses_kde(gather_action)) {
      use_kde_gather = true;
    } else if (gather_action != ActionId::kDisabled && gather_action != ActionId::kCIC) {
      use_action_gather =
          hexapic::deposition_action::uses_direct_stencil(gather_action);
    } else {
      gather_mode = effective_deposition_mode();
      use_compact_mode_gather =
          (gather_mode == kDepositionModeGaussian5x5 ||
           gather_mode == kDepositionModeGaussian7x7);
    }
  }

  for (k = 0; k < species.size(); k++) {
    Species &sp = species[k];
    if (tstep % sp.subcycle != 0)
      continue;
    if (halo_stencil)
      interface_halo_stencil_particles_step[k] +=
          static_cast<unsigned long long>(cell.part[k].size());
    if (sp.charge != 0.0)
      for (auto &p : cell.part[k]) { // charged particles
        /* Calculate electric field at particle position */
        px = p.x - i;
        py = p.y - j;
        if (use_kde_gather) {
          hexapic::kde::gather_field_kde(i, j, px, py, &Ef[0]);
        } else if (use_action_gather) {
          gather_field_action(i, j, px, py, gather_action, &Ef[0]);
        } else if (use_compact_mode_gather) {
          gather_field_compact(i, j, px, py, gather_mode, &Ef[0]);
        } else {
          cell2part(i, j, px, py, V_cell, &Ef[0]);
        }

        if (momentum_audit_enabled()) {
          const double physical_charge =
              np2c_global * static_cast<double>(sp.charge) * q_e;
          const double fx = physical_charge * static_cast<double>(Ef[0]);
          const double fy = physical_charge * static_cast<double>(Ef[1]);
          momentum_audit_force_local[0] += fx;
          momentum_audit_force_local[1] += fy;
          momentum_audit_force_abs_local[0] += std::abs(fx);
          momentum_audit_force_abs_local[1] += std::abs(fy);
        }

        /* First Half-Step Velocity Update with Electric Field */
        p.vx += sp.chom * Ef[0] * hdt;
        p.vy += sp.chom * Ef[1] * hdt;

        /* Intermediate velocity update due to rotation in magnetic field */
        vx = p.vx + (p.vy * sp.tb[2] - p.vz * sp.tb[1]);
        vy = p.vy + (p.vz * sp.tb[0] - p.vx * sp.tb[2]);
        vz = p.vz + (p.vx * sp.tb[1] - p.vy * sp.tb[0]);

        /* Final velocity update due to rotation in magnetic field */
        p.vx += vy * sp.sb[2] - vz * sp.sb[1];
        p.vy += vz * sp.sb[0] - vx * sp.sb[2];
        p.vz += vx * sp.sb[1] - vy * sp.sb[0];

        /* Second Half-Step Velocity Update with Electric Field */
        p.vx += sp.chom * Ef[0] * hdt;
        p.vy += sp.chom * Ef[1] * hdt;

        /* Full-Step Position Update */
        p.x += sp.subcycle * p.vx * dt / grid.dx; ///< normalized to cell size
        p.y += sp.subcycle * p.vy * dt / grid.dy; ///< normalized to cell size
        p.z += sp.subcycle * p.vz * dt / grid.dy; ///< normalized to cell size

        // /* Save trajectory if test particle exists */
        // if(sp.part.size()==1) {
        // 	trajx.push_back(p.x);
        // 	trajy.push_back(p.y);
        // 	trajz.push_back(p.z);
        // 	}
      }
    else
      for (auto &p : cell.part[k]) { // neutrals
        /* Full-Step Position Update */
        p.x += sp.subcycle * p.vx * dt / grid.dx;
        p.y += sp.subcycle * p.vy * dt / grid.dy;
        p.z += sp.subcycle * p.vz * dt / grid.dy;
      }
  }
}

void particle_mover_boris(const std::vector<Cell *> &cells_x) {
  for (Cell *p : cells_x)
    particle_mover_boris_cell(*p);
}

void momentum_audit_reset() {
  momentum_audit_force_local[0] = 0.0;
  momentum_audit_force_local[1] = 0.0;
  momentum_audit_force_abs_local[0] = 0.0;
  momentum_audit_force_abs_local[1] = 0.0;
  if (momentum_audit_enabled())
    momentum_audit_local_particle_momentum(momentum_audit_before_local);
}

void momentum_audit_report() {
  if (!momentum_audit_enabled())
    return;

  double force_global[2] = {0.0, 0.0};
  double force_abs_global[2] = {0.0, 0.0};
  double before_global[2] = {0.0, 0.0};
  double after_local[2] = {0.0, 0.0};
  double after_global[2] = {0.0, 0.0};
  momentum_audit_local_particle_momentum(after_local);
  MPI_Reduce(momentum_audit_force_local, force_global, 2, MPI_DOUBLE, MPI_SUM,
             0, comm);
  MPI_Reduce(momentum_audit_force_abs_local, force_abs_global, 2, MPI_DOUBLE,
             MPI_SUM, 0, comm);
  MPI_Reduce(momentum_audit_before_local, before_global, 2, MPI_DOUBLE,
             MPI_SUM, 0, comm);
  MPI_Reduce(after_local, after_global, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
  if (!rmpi) {
    if (!momentum_audit_reference_set) {
      momentum_audit_initial_global[0] = before_global[0];
      momentum_audit_initial_global[1] = before_global[1];
      momentum_audit_impulse_global[0] = 0.0;
      momentum_audit_impulse_global[1] = 0.0;
      momentum_audit_reference_set = true;
    }
    momentum_audit_impulse_global[0] += force_global[0] * dt;
    momentum_audit_impulse_global[1] += force_global[1] * dt;
    const double step_dp_x = after_global[0] - before_global[0];
    const double step_dp_y = after_global[1] - before_global[1];
    const double drift_x =
        after_global[0] - momentum_audit_initial_global[0];
    const double drift_y =
        after_global[1] - momentum_audit_initial_global[1];
    const double rel_x =
        std::abs(force_global[0]) / (force_abs_global[0] + 1.0e-300);
    const double rel_y =
        std::abs(force_global[1]) / (force_abs_global[1] + 1.0e-300);
    std::cout << std::setprecision(17) << "MOMENTUM_AUDIT step=" << tstep
              << " force_x=" << force_global[0]
              << " force_y=" << force_global[1]
              << " abs_force_x=" << force_abs_global[0]
              << " abs_force_y=" << force_abs_global[1]
              << " relative_x=" << rel_x << " relative_y=" << rel_y
              << " step_dp_x=" << step_dp_x << " step_dp_y=" << step_dp_y
              << " momentum_drift_x=" << drift_x
              << " momentum_drift_y=" << drift_y
              << " integrated_force_dt_x=" << momentum_audit_impulse_global[0]
              << " integrated_force_dt_y=" << momentum_audit_impulse_global[1]
              << " balance_error_x="
              << drift_x - momentum_audit_impulse_global[0]
              << " balance_error_y="
              << drift_y - momentum_audit_impulse_global[1]
              << "\n";
  }
}

void particle_boundaries() {

  // erase particles out of boundary for open boundary
  // put them to the other side if periodic boundary
  // reflect if neumann boundary condition

  bool erase_particle;
  int n_sp;
  std::vector<std::vector<std::vector<REAL>>> &send_data_3d =
      grid.neighbours.send_data_3d;
  bool periodic_x = periodic_x_enabled();
  bool periodic_y = periodic_y_enabled();
  int nx_global = grid.Nxg - 1;
  int ny_global = grid.Nyg - 1;

  transport_particles_sent_step = 0;
  transport_particles_received_step = 0;
  transport_particles_wrapped_step = 0;
  transport_particles_multi_rank_step = 0;

  for (int i = 0; i < species.size(); i++) {
    Species &sp = species[i];
    if (tstep % sp.subcycle != 0)
      continue;

    for (int cell_i = 0; cell_i < cells.size(); cell_i++) {
      Cell &cell = cells[cell_i];
      std::vector<Particle> &part = cell.part[i];
      for (int p_i = 0; p_i < part.size();) {
        Particle p = part[p_i];
        erase_particle = false;

        REAL xg = p.x + grid.N0x;
        REAL yg = p.y + grid.N0y;
        bool wrapped = false;

        if (periodic_x)
          xg = wrap_periodic_coordinate(xg, nx_global, wrapped);
        if (periodic_y)
          yg = wrap_periodic_coordinate(yg, ny_global, wrapped);

        if (!periodic_x && (xg < 0.0 || xg >= nx_global))
          erase_particle = true;
        if (!periodic_y && (yg < 0.0 || yg >= ny_global))
          erase_particle = true;

        if (erase_particle) {
          int lrbt = boundary_direction_from_global(xg, yg);
          particle_wall_interaction(cell_i, i, p_i, lrbt);
          remove_particle(cell_i, i, p_i);
          continue;
        }

        if (wrapped)
          transport_particles_wrapped_step += 1;

        int owner_rank = owner_rank_from_global_position(xg, yg);
        if (owner_rank < 0) {
          std::cerr << "ERROR: Unable to find owner rank for particle of species "
                    << sp.name << " at global position (" << xg << ", " << yg
                    << ") on rank " << rmpi << "\n";
          MPI_Abort(comm, EXIT_FAILURE);
        }

        if (owner_rank == rmpi) {
          p.x = xg - grid.N0x;
          p.y = yg - grid.N0y;
          int dst_x = static_cast<int>(p.x);
          int dst_y = static_cast<int>(p.y);
          if (dst_x < 0 || dst_x >= grid.Ncx || dst_y < 0 || dst_y >= grid.Ncy) {
            std::cerr << "ERROR: Local particle routing failed for species "
                      << sp.name << " on rank " << rmpi << ": local position ("
                      << p.x << ", " << p.y << "), domain=(" << grid.Ncx << ", "
                      << grid.Ncy << ")\n";
            MPI_Abort(comm, EXIT_FAILURE);
          }
          int dst_cell = dst_x * grid.Ncy + dst_y;
          if (dst_cell == cell_i) {
            part[p_i] = p;
            p_i++;
          } else {
            cells[dst_cell].part[i].push_back(p);
            std::swap(part[p_i], part.back());
            part.pop_back();
          }
          continue;
        }

        Particle p_global = p;
        p_global.x = xg;
        p_global.y = yg;
        enqueue_particle_for_rank(i, owner_rank, p_global);
        transport_particles_sent_step += 1;
        if (!grid.neighbours.all.count(owner_rank))
          transport_particles_multi_rank_step += 1;
        remove_particle(cell_i, i, p_i);
      }
    }
  }

  std::vector<std::vector<REAL>> &send_data = grid.neighbours.send_data;
  for (int rank : particle_exchange_ranks) {
    send_data[rank].push_back(static_cast<double>(species.size()));
    for (int i = 0; i < species.size(); i++)
      send_data[rank].push_back(0.0);
  }

  for (int rank : particle_exchange_ranks)
    for (int i = 0; i < species.size(); i++) {
      n_sp = (send_data_3d[i][rank].size()) / 6;
      send_data[rank][i + 1] = n_sp;
      if (n_sp > 0) {
        send_data[rank].insert(send_data[rank].end(), send_data_3d[i][rank].begin(),
                               send_data_3d[i][rank].end());
      }
      send_data_3d[i][rank] = {};
    }

  send_receive_particles();
  add_received_particles();
}

void add_received_particles() {
  for (int rank : particle_exchange_ranks) {

    std::vector<REAL> &recv_data = grid.neighbours.recv_data[rank];
    size_t i, j, sp, n_sp;

    if (recv_data.empty())
      continue;

    n_sp = static_cast<size_t>(recv_data[0]);
    i = 1 + n_sp;

    for (sp = 0; sp < n_sp; sp++) {
      size_t n_recv = static_cast<size_t>(recv_data[sp + 1]);
      species[sp].n += static_cast<unsigned int>(n_recv);
      for (j = 0; j < n_recv; j++) {
        Particle p;
        REAL xg = recv_data[i];
        REAL yg = recv_data[i + 1];
        p.x = xg - grid.N0x;
        p.y = yg - grid.N0y;
        p.z = recv_data[i + 2];
        p.vx = recv_data[i + 3];
        p.vy = recv_data[i + 4];
        p.vz = recv_data[i + 5];
        i += 6;

        int owner_rank = owner_rank_from_global_position(xg, yg);
        if (owner_rank != rmpi || p.x < 0.0 || p.x >= grid.Ncx || p.y < 0.0 ||
            p.y >= grid.Ncy) {
          std::cerr << "ERROR: Received particle of species " << species[sp].name
                    << " routed to wrong rank. sender=" << rank
                    << ", owner=" << owner_rank << ", receiver=" << rmpi
                    << ", global=(" << xg << ", " << yg << "), local=(" << p.x
                    << ", " << p.y << ")\n";
          MPI_Abort(comm, EXIT_FAILURE);
        }

        size_t cell_i = static_cast<size_t>(p.y) +
                        static_cast<size_t>(p.x) * grid.Ncy;
        cells[cell_i].part[sp].push_back(p);
        transport_particles_received_step += 1;
      }
    }
  }
}

void send_receive_particles() {
  std::vector<int> send_sizes(nmpi, 0), recv_sizes(nmpi, 0);
  std::vector<MPI_Request> size_requests;
  std::vector<MPI_Request> data_requests;

  for (int rank : particle_exchange_ranks) {
    send_sizes[rank] = static_cast<int>(grid.neighbours.send_data[rank].size());

    size_requests.push_back(MPI_Request());
    MPI_Isend(&send_sizes[rank], 1, MPI_INT, rank, 0, MPI_COMM_WORLD,
              &size_requests.back());

    size_requests.push_back(MPI_Request());
    MPI_Irecv(&recv_sizes[rank], 1, MPI_INT, rank, 0, MPI_COMM_WORLD,
              &size_requests.back());
  }

  if (!size_requests.empty())
    MPI_Waitall(static_cast<int>(size_requests.size()), size_requests.data(),
                MPI_STATUSES_IGNORE);

  for (int rank : particle_exchange_ranks)
    grid.neighbours.recv_data[rank].assign(recv_sizes[rank], 0.0);

  for (int rank : particle_exchange_ranks) {
    std::vector<REAL> &data = grid.neighbours.send_data[rank];
    data_requests.push_back(MPI_Request());
    MPI_Isend(data.data(), send_sizes[rank], MPI_DOUBLE, rank, 1, MPI_COMM_WORLD,
              &data_requests.back());

    data_requests.push_back(MPI_Request());
    MPI_Irecv(grid.neighbours.recv_data[rank].data(), recv_sizes[rank],
              MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &data_requests.back());
  }
  if (!data_requests.empty())
    MPI_Waitall(static_cast<int>(data_requests.size()), data_requests.data(),
                MPI_STATUSES_IGNORE);

  for (int rank : particle_exchange_ranks)
    std::vector<REAL>().swap(grid.neighbours.send_data[rank]);
}

void initial_particle_load() {
  int i, j, Npart, cell_i;
  for (i = 0; i < species.size(); i++) {
    Species &sp = species[i];
    for (size_t load_i = 0; load_i < sp.load.size(); load_i++) {
      auto &ld = sp.load[load_i];
      if (LandauInitModeFlag == 2 && ld.method == 0) {
        initial_particle_load_landau_quiet(i, static_cast<int>(load_i), sp, ld);
        continue;
      }
      if (ld.method == 0) {
        REAL global_expected =
            ld.density_full * (ld.x1f_full - ld.x1s_full) *
            (ld.x2f_full - ld.x2s_full) / np2c_global;
        int global_count =
            static_cast<int>(std::floor(global_expected + 1e-12));

        // Preserve sparse global loads exactly across decomposed domains.
        if (global_count > 0 && global_count <= 1024) {
          for (int sample_i = 0; sample_i < global_count; sample_i++) {
            REAL x_m, y_m;
            if (global_count == 1) {
              REAL eps_x = 1e-6 * grid.dx;
              REAL eps_y = 1e-6 * grid.dy;
              x_m = std::min(0.5 * (ld.x1s_full + ld.x1f_full) + eps_x,
                             ld.x1f_full - eps_x);
              y_m = std::min(0.5 * (ld.x2s_full + ld.x2f_full) + eps_y,
                             ld.x2f_full - eps_y);
            } else {
              x_m = ld.x1s_full +
                    deterministic_unit_sample(i, static_cast<int>(load_i),
                                              sample_i, 0) *
                        (ld.x1f_full - ld.x1s_full);
              y_m = ld.x2s_full +
                    deterministic_unit_sample(i, static_cast<int>(load_i),
                                              sample_i, 1) *
                        (ld.x2f_full - ld.x2s_full);
            }

            if (x_m < grid.x1s || x_m >= grid.x1f || y_m < grid.x2s ||
                y_m >= grid.x2f)
              continue;

            Particle p;
            p.x = x_m / grid.dx - grid.N0x;
            p.y = y_m / grid.dy - grid.N0y;
            p.z = 0.0;
            p.vx = ld.v1drift + ld.v1thermal * normvel_MaxBol();
            p.vy = ld.v2drift + ld.v2thermal * normvel_MaxBol();
            p.vz = ld.v3drift + ld.v3thermal * normvel_MaxBol();
            cell_i = int(p.y) + int(p.x) * grid.Ncy;
            cells[cell_i].part[i].push_back(p);
            sp.n += 1;
          }
          continue;
        }

        REAL local_expected = 0.0;
        if (!(grid.x1s > ld.x1f || grid.x1f < ld.x1s || grid.x2s > ld.x2f ||
              grid.x2f < ld.x2s))
          local_expected =
              ld.density * (ld.x1f - ld.x1s) * (ld.x2f - ld.x2s) / np2c_global;

        // Preserve the total global particle count after domain decomposition.
        Npart = decompose_expected_particle_count(local_expected);
        for (j = 0; j < Npart; j++) {
          Particle p;
          p.x = (frand() * (ld.x1f - ld.x1s) + ld.x1s) / grid.dx - grid.N0x;
          p.y = (frand() * (ld.x2f - ld.x2s) + ld.x2s) / grid.dy - grid.N0y;
          p.z = 0.0;
          p.vx = ld.v1drift + ld.v1thermal * normvel_MaxBol();
          p.vy = ld.v2drift + ld.v2thermal * normvel_MaxBol();
          p.vz = ld.v3drift + ld.v3thermal * normvel_MaxBol();
          cell_i = int(p.y) + int(p.x) * grid.Ncy;
          cells[cell_i].part[i].push_back(p);
          sp.n += 1;
        }
      } else if (!rmpi) {
        std::cout << "LOAD: Initial load method " << ld.method
                  << " not implemented. Use method=0, i.e. uniform."
                  << "\n";
        exit(1);
      }
    }
  }
}

void remove_particle(int cell_i, int sp, int p) {
  cells[cell_i].part[sp][p] = cells[cell_i].part[sp].back();
  cells[cell_i].part[sp].pop_back();
  species[sp].n -= 1;
}

void inject_particles() {
  int i, j, Npart, intx, inty, cell_i;
  REAL src;
  for (i = 0; i < species.size(); i++) {
    Species &sp = species[i];
    for (auto &inj : sp.inject) {
      if (inj.I > 0) {
        intx = inj.j2 - inj.j1;
        inty = inj.k2 - inj.k1;
        src = inj.I * dt / abs(sp.charge) / q_e / np2c_global;
        // just injection along one of dimensions
        if (intx)
          src *= intx;
        if (inty)
          src *= inty;
        inj.f += src;
        Npart = inj.f;
        for (j = 0; j < Npart; j++) {
          Particle p;
          if (intx) {
            p.x = frand() * intx + inj.j1 - grid.N0x;
            p.vx = inj.v1thermal * normvel_MaxBol();
          } else {
            p.x = frand() + inj.j1 - grid.N0x;
            p.vx = inj.normal * inj.v1thermal * normvel_vFv();
          }

          if (inty) {
            p.y = frand() * inty + inj.k1 - grid.N0y;
            p.vy = inj.v2thermal * normvel_MaxBol();
          } else {
            p.y = frand() + inj.k1 - grid.N0y;
            p.vy = inj.normal * inj.v2thermal * normvel_vFv();
          }
          p.z = 0.0;
          p.vz = inj.v3thermal * normvel_MaxBol();

          cell_i = p.x;
          cell_i *= grid.Ncy;
          cell_i += p.y;

          cells[cell_i].part[i].push_back(p);
          sp.n += 1;
        }
        inj.f -= Npart;
      }
    }
  }
}

REAL normvel_MaxBol() {

  /* Box-Muller method
   W. H. Press, B. P. Flannery, S. A. Teukolsky, and W. Y. Vetterling.
Numerical Recipes in C: The Art of Scientiﬁc Programming. Cambridge
University Press, Cambridge, UK, 1988.	*/

  static std::vector<REAL> vel;
  REAL n1 = 1, n2 = 1, R2, val;

  // remove previously used value
  if (vel.size())
    vel.pop_back();

  // return available value
  if (vel.size())
    return (vel.back());

  // else generate new values
  while (n1 * n1 + n2 * n2 > 1) {
    n1 = frand() * 2 - 1;
    n2 = frand() * 2 - 1;
  }

  R2 = n1 * n1 + n2 * n2;
  val = sqrt(-2 * log(R2) / R2);

  vel.push_back(n1 * val);
  vel.push_back(n2 * val);

  return (vel.back());
}

REAL normvel_vFv() {

  /*
   W. H. Press, B. P. Flannery, S. A. Teukolsky, and W. Y. Vetterling.
Numerical Recipes in C: The Art of Scientiﬁc Programming. Cambridge
University Press, Cambridge, UK, 1988.	*/

  return sqrt(-log(frand()));
}
