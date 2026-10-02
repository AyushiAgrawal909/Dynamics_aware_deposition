/**
 * SPDX-License-Identifier: EUPL-1.2
 * SPDX-FileCopyrightText: 2025 Stefan Costea, LeCAD-PEG
 *
 * HYPRE implementation of the field solver
 * For now just basic version with fixed solver and preconditioner
 * It uses Struct as for our grid this gives us the most optimized solutions
 * Beacause Struct does not have ghost values direct access with API,
 * 	manual MPI send/recive of this values is done

*/

#include "hexapic.hpp"

#include <cmath>
#include <cstdlib>

constexpr int HYPRE_TAG_V = 101;
constexpr int HYPRE_TAG_RHO = 102;
constexpr int HYPRE_TAG_DEBUG_SEND = 103;
constexpr int HYPRE_TAG_DEBUG_RECV = 104;

static bool hypre_audit_enabled() {
  const char *value = std::getenv("HEXAPIC_MOMENTUM_AUDIT");
  return value != nullptr && std::string(value) != "0";
}

static bool hypre_spd_form_enabled() {
  const char *value = std::getenv("HEXAPIC_HYPRE_SPD_FORM");
  return value != nullptr && std::string(value) != "0";
}

static bool hypre_cold_start_enabled() {
  const char *value = std::getenv("HEXAPIC_HYPRE_COLD_START");
  return value != nullptr && std::string(value) != "0";
}

static bool hypre_pin_gauge_enabled() {
  const char *value = std::getenv("HEXAPIC_HYPRE_PIN_GAUGE");
  return value != nullptr && std::string(value) != "0";
}

static double hypre_diagonal_shift() {
  const char *value = std::getenv("HEXAPIC_HYPRE_DIAGONAL_SHIFT");
  if (value == nullptr)
    return 0.0;
  char *end = nullptr;
  const double shift = std::strtod(value, &end);
  if (end == value || *end != '\0' || !std::isfinite(shift) || shift < 0.0)
    return 0.0;
  return shift;
}

static int hypre_relax_type() {
  const char *value = std::getenv("HEXAPIC_HYPRE_RELAX_TYPE");
  if (value == nullptr)
    return 2;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > 3)
    return 2;
  return static_cast<int>(parsed);
}

static bool hypre_is_fully_periodic();

static double hypre_runtime_tolerance() {
  constexpr double default_tolerance = 1.0e-9;
  const char *value = std::getenv("HEXAPIC_HYPRE_TOL");
  if (value == nullptr)
    return default_tolerance;
  char *end = nullptr;
  const double tolerance = std::strtod(value, &end);
  if (end == value || *end != '\0' || !std::isfinite(tolerance) ||
      tolerance <= 0.0) {
    if (!rmpi)
      std::cerr << "WARNING: invalid HEXAPIC_HYPRE_TOL='" << value
                << "'; using " << default_tolerance << "\n";
    return default_tolerance;
  }
  return tolerance;
}

static double hypre_manual_relative_residual_periodic() {
  if (!hypre_is_fully_periodic() || nmpi != 1)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<double> solution(static_cast<size_t>(hypre.size), 0.0);
  std::vector<double> source(static_cast<size_t>(hypre.size), 0.0);
  HYPRE_StructVectorGetBoxValues(x_h, hypre.ilower, hypre.iupper,
                                 solution.data());
  HYPRE_StructVectorGetBoxValues(b_h, hypre.ilower, hypre.iupper,
                                 source.data());

  double residual_sq = 0.0;
  double source_sq = 0.0;
  const int nx = hypre.nx;
  const int ny = hypre.ny;
  const double sign = hypre_spd_form_enabled() ? -1.0 : 1.0;
  const double center = sign * (-2.0 * dx2_inv - 2.0 * dy2_inv) +
                        hypre_diagonal_shift();
  for (int iy = 0; iy < ny; ++iy) {
    const int ym = (iy - 1 + ny) % ny;
    const int yp = (iy + 1) % ny;
    for (int ix = 0; ix < nx; ++ix) {
      const int xm = (ix - 1 + nx) % nx;
      const int xp = (ix + 1) % nx;
      const size_t idx = static_cast<size_t>(iy * nx + ix);
      const double applied =
          center * solution[idx] +
          sign * dx2_inv *
              (solution[static_cast<size_t>(iy * nx + xm)] +
               solution[static_cast<size_t>(iy * nx + xp)]) +
          sign * dy2_inv *
              (solution[static_cast<size_t>(ym * nx + ix)] +
               solution[static_cast<size_t>(yp * nx + ix)]);
      const double applied_with_gauge =
          hypre_pin_gauge_enabled() && ix == 0 && iy == 0
              ? solution[idx]
              : applied;
      const double residual = source[idx] - applied_with_gauge;
      residual_sq += residual * residual;
      source_sq += source[idx] * source[idx];
    }
  }
  return std::sqrt(residual_sq / (source_sq + 1.0e-300));
}

static bool hypre_is_fully_periodic() {
  return hypre.periodic[0] && hypre.periodic[1];
}

static long long hypre_global_unknowns() {
  long long nx = hypre.periodic[0] ? (grid.Nxg - 1) : grid.Nxg;
  long long ny = hypre.periodic[1] ? (grid.Nyg - 1) : grid.Nyg;
  return nx * ny;
}

static void hypre_subtract_global_mean(double *data, int local_size) {
  if (!hypre_is_fully_periodic())
    return;

  double local_sum = 0.0;
  for (int i = 0; i < local_size; i++)
    local_sum += data[i];

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  double mean = global_sum / static_cast<double>(hypre_global_unknowns());
  for (int i = 0; i < local_size; i++)
    data[i] -= mean;
}

static void hypre_validate_halo_sizes() {
  bool mismatch = false;

  for (int r : grid.neighbours.all) {
    int peer_send = 0;
    int peer_recv = 0;

    MPI_Sendrecv(&hypre.size_s[r], 1, MPI_INT, r, HYPRE_TAG_DEBUG_SEND,
                 &peer_send, 1, MPI_INT, r, HYPRE_TAG_DEBUG_SEND, comm,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(&hypre.size_r[r], 1, MPI_INT, r, HYPRE_TAG_DEBUG_RECV,
                 &peer_recv, 1, MPI_INT, r, HYPRE_TAG_DEBUG_RECV, comm,
                 MPI_STATUS_IGNORE);

    if (hypre.size_s[r] != peer_recv || hypre.size_r[r] != peer_send) {
      mismatch = true;
      std::cerr << "HYPRE halo mismatch rank " << rmpi << " <-> " << r
                << " send=" << hypre.size_s[r]
                << " peer_recv=" << peer_recv
                << " recv=" << hypre.size_r[r]
                << " peer_send=" << peer_send << "\n";
    }
  }

  if (mismatch)
    MPI_Abort(comm, MPI_ERR_TRUNCATE);
}

void hypre_init_geo() {
  if (grid.boundaries[0].name == "Periodic")
    hypre.periodic[0] = grid.Nxg - 1;
  else
    hypre.periodic[0] = 0;
  if (grid.boundaries[2].name == "Periodic")
    hypre.periodic[1] = grid.Nyg - 1;
  else
    hypre.periodic[1] = 0;

  // if it continues with next rank-> edge nodes are solved in next domain
  hypre.ilower[0] = grid.N0x;
  hypre.ilower[1] = grid.N0y;

  if ((grid.neighbours.rxmax[0] == rmpi) && (!hypre.periodic[0])) // boundary
    hypre.iupper[0] = hypre.ilower[0] + grid.Ncx;
  else
    hypre.iupper[0] = hypre.ilower[0] + grid.Ncx - 1;

  if ((grid.neighbours.rymax[0] == rmpi) && (!hypre.periodic[1])) // boundary
    hypre.iupper[1] = hypre.ilower[1] + grid.Ncy;
  else
    hypre.iupper[1] = hypre.ilower[1] + grid.Ncy - 1;

  hypre.size_extend = (grid.Ncx + 1) * (grid.Ncy + 1);
  hypre.nx = hypre.iupper[0] - hypre.ilower[0] + 1;
  hypre.ny = hypre.iupper[1] - hypre.ilower[1] + 1;
  hypre.size = hypre.nx * hypre.ny;

  hypre.size_s.resize(nmpi);
  hypre.size_r.resize(nmpi);
  hypre.V_s_idx.resize(nmpi);
  hypre.V_r_idx.resize(nmpi);
  hypre.send_buff.resize(nmpi);
  hypre.recv_buff.resize(nmpi);

  hypre.bcs = {0, 0, 0, 0};
  std::vector<Boundary> &bcs = grid.boundaries;

  if (bcs[0].j1 == hypre.ilower[0]) {
    if (bcs[0].name == "Equipotential")
      hypre.bcs[0] = 1;
    else if (bcs[0].name == "Dielectric")
      hypre.bcs[0] = 2;
  }

  if (bcs[1].j1 == hypre.iupper[0]) {
    if (bcs[1].name == "Equipotential")
      hypre.bcs[1] = 1;
    else if (bcs[1].name == "Dielectric")
      hypre.bcs[1] = 2;
  }

  if (bcs[2].k1 == hypre.ilower[1]) {
    if (bcs[2].name == "Equipotential")
      hypre.bcs[2] = 1;
    else if (bcs[2].name == "Dielectric")
      hypre.bcs[2] = 2;
  }

  if (bcs[3].k1 == hypre.iupper[1]) {
    if (bcs[3].name == "Equipotential")
      hypre.bcs[3] = 1;
    else if (bcs[3].name == "Dielectric")
      hypre.bcs[3] = 2;
  }
}

void hypre_init_ghost() {
  Neighbours &neigh = grid.neighbours;
  int r, r_prev, r_c;
  const bool spans_full_periodic_y =
      hypre.periodic[1] && (grid.Ncy == grid.Nyg - 1);

  // sending and receiving V

  // corner
  r_c = neigh.corners[0];
  if (!spans_full_periodic_y && r_c != rmpi)
    hypre.V_s_idx[r_c].push_back(0);

  // left boundary
  r_prev = r_c;
  for (int j = 0; j < neigh.rxmin.size(); j++) {
    r = neigh.rxmin[j];
    if (r == rmpi)
      continue;

    if (r != r_prev)
      hypre.V_s_idx[r].push_back(j);
    r_prev = r;

    if (j != neigh.rxmin.size() - 1)
      hypre.V_s_idx[r].push_back(j + 1);
    else if (grid.Ny == hypre.ny)
      hypre.V_s_idx[r].push_back(j + 1);
  }

  // bottom boundary
  r_prev = r_c;
  for (int i = 0; i < neigh.rymin.size(); i++) {
    r = neigh.rymin[i];
    if (r == rmpi)
      continue;

    if (r != r_prev)
      hypre.V_s_idx[r].push_back(i * grid.Ny);
    r_prev = r;

    if (i != neigh.rymin.size() - 1)
      hypre.V_s_idx[r].push_back((i + 1) * grid.Ny);
    else if (grid.Nx == hypre.nx)
      hypre.V_s_idx[r].push_back((i + 1) * grid.Ny);
  }

  // right boundary
  for (int j = 0; j < neigh.rxmax.size(); j++) {
    r = neigh.rxmax[j];
    if (r == rmpi)
      continue;

    hypre.V_r_idx[r].push_back(grid.Ny * (grid.Nx - 1) + j);
  }

  // top boundary
  for (int i = 0; i < neigh.rymax.size(); i++) {
    r = neigh.rymax[i];
    if (r == rmpi)
      continue;

    hypre.V_r_idx[r].push_back(grid.Ny * (i + 1) - 1);
  }

  // top corner
  r_c = neigh.corners[3];
  if (!spans_full_periodic_y && r_c != rmpi) {
    hypre.V_r_idx[r_c].push_back(grid.Ny * grid.Nx - 1);
  } else if (!spans_full_periodic_y) {
    r_c = neigh.rxmax[grid.Ncy - 1];
    if (r_c != rmpi) {
      hypre.V_r_idx[r_c].push_back(grid.Ny * grid.Nx - 1);
    } else if (grid.Nx == hypre.nx) {
      r_c = neigh.rymax[grid.Ncx - 1];
      if (r_c != rmpi) {
        hypre.V_r_idx[r_c].push_back(grid.Ny * grid.Nx - 1);
      }
    }
  }
  // if double boundary then no receiving...

  // rho sending and receiving is just in the opposite direction
  // also using opposite send recive buffers!

  // size, buffers
  for (int i = 0; i < nmpi; i++) {
    hypre.size_r[i] = hypre.V_r_idx[i].size();
    hypre.size_s[i] = hypre.V_s_idx[i].size();
    hypre.send_buff[i].resize(hypre.size_s[i]);
    hypre.recv_buff[i].resize(hypre.size_r[i]);
  }

  hypre_validate_halo_sizes();

  // --- Build persistent MPI requests for halo exchanges ---
  hypre_persist_v_reqs.clear();
  hypre_persist_rho_reqs.clear();

  // For V exchange: Recv into recv_buff (size_r), Send from send_buff (size_s)
  for (int r : grid.neighbours.all) {
    if (hypre.size_r[r] > 0) {
      MPI_Request req;
      MPI_Recv_init(hypre.recv_buff[r].data(), hypre.size_r[r], MPI_DOUBLE, r,
                    HYPRE_TAG_V, comm, &req);
      hypre_persist_v_reqs.push_back(req);
    }
  }
  for (int r : grid.neighbours.all) {
    if (hypre.size_s[r] > 0) {
      MPI_Request req;
      MPI_Send_init(hypre.send_buff[r].data(), hypre.size_s[r], MPI_DOUBLE, r,
                    HYPRE_TAG_V, comm, &req);
      hypre_persist_v_reqs.push_back(req);
    }
  }

  // For rho exchange (opposite buffers): Recv into send_buff (size_s), Send
  // from recv_buff (size_r)
  for (int r : grid.neighbours.all) {
    if (hypre.size_s[r] > 0) {
      MPI_Request req;
      MPI_Recv_init(hypre.send_buff[r].data(), hypre.size_s[r], MPI_DOUBLE, r,
                    HYPRE_TAG_RHO, comm, &req);
      hypre_persist_rho_reqs.push_back(req);
    }
  }
  for (int r : grid.neighbours.all) {
    if (hypre.size_r[r] > 0) {
      MPI_Request req;
      MPI_Send_init(hypre.recv_buff[r].data(), hypre.size_r[r], MPI_DOUBLE, r,
                    HYPRE_TAG_RHO, comm, &req);
      hypre_persist_rho_reqs.push_back(req);
    }
  }
}

void hypre_init_solver() {
  if (hypre_spd_form_enabled() && !hypre_is_fully_periodic()) {
    if (!rmpi)
      std::cerr << "ERROR: HEXAPIC_HYPRE_SPD_FORM audit currently supports "
                   "fully periodic grids only\n";
    MPI_Abort(comm, EXIT_FAILURE);
  }
  // grid
  HYPRE_StructGridCreate(comm, 2, &grid_h);
  HYPRE_StructGridSetExtents(grid_h, hypre.ilower, hypre.iupper);
  HYPRE_StructGridSetPeriodic(grid_h, hypre.periodic);
  HYPRE_StructGridAssemble(grid_h);

  // 5-point stencil
  HYPRE_StructStencilCreate(2, 5, &stencil_h);
  int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (int i = 0; i < 5; i++)
    HYPRE_StructStencilSetElement(stencil_h, i, offsets[i]);

  // matrix
  int nentries = 5;
  double values[5];
  int stencil_indices[5] = {0, 1, 2, 3, 4};
  HYPRE_StructMatrixCreate(comm, grid_h, stencil_h, &A_h);
  HYPRE_StructMatrixInitialize(A_h);

  dx2_inv = 1.0;
  dy2_inv = (grid.dx * grid.dx) / (grid.dy * grid.dy);

  std::vector<Boundary> &bcs = grid.boundaries;
  std::vector<int> bcs_ij;

  for (int j = hypre.ilower[1]; j <= hypre.iupper[1]; j++) {
    for (int i = hypre.ilower[0]; i <= hypre.iupper[0]; i++) {
      int index[2] = {i, j};
      values[0] = values[1] = values[2] = values[3] = values[4] = 0.0;

      // for periodic normal stencil
      // for dirichlet fixed point
      // for open the difference to outer point is 0

      bcs_ij = {0, 0, 0, 0};
      bool d = false;

      if ((i == bcs[0].j1 && i == bcs[0].j2)) { // xmin boundary
        if (bcs[0].name == "Equipotential") {
          bcs_ij[0] = 1;
          d = true;
        } else if (bcs[0].name == "Dielectric")
          bcs_ij[0] = 2;
      }

      if (i == bcs[1].j1 && i == bcs[1].j2) { // xmax boundary
        if (bcs[1].name == "Equipotential") {
          bcs_ij[1] = 1;
          d = true;
        } else if (bcs[1].name == "Dielectric")
          bcs_ij[1] = 2;
      }

      if (j == bcs[2].k1 && j == bcs[2].k2) { // ymin boundary
        if (bcs[2].name == "Equipotential") {
          bcs_ij[2] = 1;
          d = true;
        } else if (bcs[2].name == "Dielectric")
          bcs_ij[2] = 2;
      }

      if (j == bcs[3].k1 && j == bcs[3].k2) { // ymax boundary
        if (bcs[3].name == "Equipotential") {
          bcs_ij[3] = 1;
          d = true;
        } else if (bcs[3].name == "Dielectric")
          bcs_ij[3] = 2;
      }

      // if any dirichlet -> fixed boundary!
      if (d)
        values[0] = dx2_inv;
      else {
        if (bcs_ij[0] != 2) {
          values[0] -= dx2_inv;
          values[1] += dx2_inv;
        }
        if (bcs_ij[1] != 2) {
          values[0] -= dx2_inv;
          values[2] += dx2_inv;
        }
        if (bcs_ij[2] != 2) {
          values[0] -= dy2_inv;
          values[3] += dy2_inv;
        }
        if (bcs_ij[3] != 2) {
          values[0] -= dy2_inv;
          values[4] += dy2_inv;
        }
      }

      if (hypre_spd_form_enabled())
        for (double &value : values)
          value = -value;
      values[0] += hypre_diagonal_shift();

      if (hypre_pin_gauge_enabled()) {
        const int nx_periodic = grid.Nxg - 1;
        const int ny_periodic = grid.Nyg - 1;
        if (i == 0 && j == 0) {
          values[0] = 1.0;
          values[1] = values[2] = values[3] = values[4] = 0.0;
        } else {
          const int xm = (i - 1 + nx_periodic) % nx_periodic;
          const int xp = (i + 1) % nx_periodic;
          const int ym = (j - 1 + ny_periodic) % ny_periodic;
          const int yp = (j + 1) % ny_periodic;
          if (xm == 0 && j == 0)
            values[1] = 0.0;
          if (xp == 0 && j == 0)
            values[2] = 0.0;
          if (i == 0 && ym == 0)
            values[3] = 0.0;
          if (i == 0 && yp == 0)
            values[4] = 0.0;
        }
      }

      HYPRE_StructMatrixSetValues(A_h, index, nentries, stencil_indices,
                                  values);
    }
  }
  HYPRE_StructMatrixAssemble(A_h);

  // --- PFMG solver (geometric) ---
  // HYPRE_StructSolver pfmg;
  HYPRE_StructPFMGCreate(comm, &pfmg);
  const double pfmg_tolerance = hypre_runtime_tolerance();
  HYPRE_StructPFMGSetTol(pfmg, pfmg_tolerance);
  HYPRE_StructPFMGSetMaxIter(pfmg, 500);
  HYPRE_StructPFMGSetRelChange(pfmg, 0);
  HYPRE_StructPFMGSetNumPreRelax(pfmg, 1);
  HYPRE_StructPFMGSetNumPostRelax(pfmg, 1);
  HYPRE_StructPFMGSetRelaxType(pfmg, hypre_relax_type());
  HYPRE_StructPFMGSetLogging(pfmg, hypre_audit_enabled() ? 1 : 0);
  if (!rmpi && hypre_audit_enabled())
    std::cout << std::setprecision(17)
              << "HYPRE_AUDIT configured_tolerance=" << pfmg_tolerance
              << " spd_form=" << (hypre_spd_form_enabled() ? 1 : 0)
              << " cold_start=" << (hypre_cold_start_enabled() ? 1 : 0)
              << " pin_gauge=" << (hypre_pin_gauge_enabled() ? 1 : 0)
              << " diagonal_shift=" << hypre_diagonal_shift()
              << " relax_type=" << hypre_relax_type()
              << "\n";

  V_h = (double *)calloc(hypre.size, sizeof(double));
  rho_h = (double *)calloc(hypre.size, sizeof(double));
  V_local = (double *)calloc(hypre.size_extend, sizeof(double));

  HYPRE_StructVectorCreate(comm, grid_h, &b_h);
  HYPRE_StructVectorCreate(comm, grid_h, &x_h);
  HYPRE_StructVectorInitialize(b_h);
  HYPRE_StructVectorInitialize(x_h);

  HYPRE_StructVectorSetBoxValues(b_h, hypre.ilower, hypre.iupper, rho_h);
  HYPRE_StructVectorSetBoxValues(x_h, hypre.ilower, hypre.iupper, V_h);

  HYPRE_StructVectorAssemble(b_h);
  HYPRE_StructVectorAssemble(x_h);

  // One-time PFMG setup (outside timestep loop)
  HYPRE_StructPFMGSetup(pfmg, A_h, b_h, x_h);
}

void hypre_init() {
  hypre_init_geo();
  hypre_init_ghost();
  hypre_init_solver();
}

/* Split send and receive in order to overlap with computation */

// ---- V halo (pack + start) ----
void hypre_post_V_halo() {
  // pack
  for (int r : grid.neighbours.all)
    for (int j = 0; j < hypre.size_s[r]; ++j)
      hypre.send_buff[r][j] = V_local[hypre.V_s_idx[r][j]];

  if (!hypre_persist_v_reqs.empty())
    MPI_Startall((int)hypre_persist_v_reqs.size(), hypre_persist_v_reqs.data());
}

void hypre_finish_V_halo() {
  if (!hypre_persist_v_reqs.empty())
    MPI_Waitall((int)hypre_persist_v_reqs.size(), hypre_persist_v_reqs.data(),
                MPI_STATUSES_IGNORE);

  // unpack
  for (int r : grid.neighbours.all)
    for (int j = 0; j < hypre.size_r[r]; ++j)
      V_local[hypre.V_r_idx[r][j]] = hypre.recv_buff[r][j];
}

// ---- rho halo (pack + start) ----
void hypre_post_rho_halo() {
  // note: your rho packs into recv_buff in the opposite direction
  for (int r : grid.neighbours.all)
    for (int j = 0; j < hypre.size_r[r]; ++j)
      hypre.recv_buff[r][j] = rho[hypre.V_r_idx[r][j]];

  if (!hypre_persist_rho_reqs.empty())
    MPI_Startall((int)hypre_persist_rho_reqs.size(),
                 hypre_persist_rho_reqs.data());
}

void hypre_finish_rho_halo() {
  if (!hypre_persist_rho_reqs.empty())
    MPI_Waitall((int)hypre_persist_rho_reqs.size(),
                hypre_persist_rho_reqs.data(), MPI_STATUSES_IGNORE);

  // accumulate
  for (int r : grid.neighbours.all)
    for (int j = 0; j < hypre.size_s[r]; ++j)
      rho[hypre.V_s_idx[r][j]] += hypre.send_buff[r][j];
}

// void hypre_exchange_ghost_V() {
//     // Pack send buffers from V_local
//     for (int r : grid.neighbours.all) {
//         for (int j = 0; j < hypre.size_s[r]; ++j) {
//             hypre.send_buff[r][j] = V_local[hypre.V_s_idx[r][j]];
//         }
//     }

//     // Start all persistent recvs + sends (created in hypre_init_ghost)
//     if (!hypre_persist_v_reqs.empty()) {
//         MPI_Startall(static_cast<int>(hypre_persist_v_reqs.size()),
//         hypre_persist_v_reqs.data());
//         MPI_Waitall(static_cast<int>(hypre_persist_v_reqs.size()),
//         hypre_persist_v_reqs.data(), MPI_STATUSES_IGNORE);
//     }

//     // Unpack into V_local
//     for (int r : grid.neighbours.all) {
//         for (int j = 0; j < hypre.size_r[r]; ++j) {
//             V_local[hypre.V_r_idx[r][j]] = hypre.recv_buff[r][j];
//         }
//     }
// }

void hypre_update_Vlocal() {
  int i;
  for (int x_i = 0; x_i < hypre.nx; x_i++) {
    i = x_i * (grid.Ncy + 1);
    for (int y_i = 0; y_i < hypre.ny; y_i++) {
      V_local[i] = V_h[y_i * hypre.nx + x_i];
      i++;
    }
  }
}

void hypre_get_V() {
  HYPRE_StructVectorGetBoxValues(x_h, hypre.ilower, hypre.iupper, V_h);
  hypre_subtract_global_mean(V_h, hypre.size);

  // first overwrite what you already have then send, receive and write
  hypre_update_Vlocal();

  // hypre_exchange_ghost_V();
  hypre_post_V_halo();
}

void hypre_field_solver() {
  if (hypre_cold_start_enabled()) {
    std::vector<double> zeros(static_cast<size_t>(hypre.size), 0.0);
    HYPRE_StructVectorSetBoxValues(x_h, hypre.ilower, hypre.iupper,
                                   zeros.data());
    HYPRE_StructVectorAssemble(x_h);
    HYPRE_StructPFMGSetZeroGuess(pfmg);
  } else {
    HYPRE_StructPFMGSetNonZeroGuess(pfmg);
  }
  HYPRE_StructPFMGSolve(pfmg, A_h, b_h, x_h);
  if (hypre_audit_enabled()) {
    int iterations = 0;
    double residual = 0.0;
    HYPRE_StructPFMGGetNumIterations(pfmg, &iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(pfmg, &residual);
    const double manual_residual = hypre_manual_relative_residual_periodic();
    if (!rmpi)
      std::cout << std::setprecision(17) << "HYPRE_AUDIT step=" << tstep
                << " iterations=" << iterations
                << " final_relative_residual=" << residual
                << " manual_relative_residual=" << manual_residual << "\n";
  }
}

// void hypre_exchange_ghost_rho() {
//     // Pack from rho into recv_buff (note: opposite buffers by design)
//     for (int r : grid.neighbours.all) {
//         for (int j = 0; j < hypre.size_r[r]; ++j) {
//             hypre.recv_buff[r][j] = rho[hypre.V_r_idx[r][j]];
//         }
//     }

//     // Start all persistent recvs + sends for rho
//     if (!hypre_persist_rho_reqs.empty()) {
//         MPI_Startall(static_cast<int>(hypre_persist_rho_reqs.size()),
//         hypre_persist_rho_reqs.data());
//         MPI_Waitall(static_cast<int>(hypre_persist_rho_reqs.size()),
//         hypre_persist_rho_reqs.data(), MPI_STATUSES_IGNORE);
//     }

//     // Accumulate into rho from send_buff (received data placed there)
//     for (int r : grid.neighbours.all) {
//         for (int j = 0; j < hypre.size_s[r]; ++j) {
//             rho[hypre.V_s_idx[r][j]] += hypre.send_buff[r][j];
//         }
//     }
// }

void hypre_rho_update() {

  int i;
  for (int x_i = 0; x_i < hypre.nx; x_i++) {
    i = x_i * (grid.Ncy + 1);
    for (int y_i = 0; y_i < hypre.ny; y_i++) {
      rho_h[y_i * hypre.nx + x_i] = rho[i];
      i++;
    }
  }

  hypre_subtract_global_mean(rho_h, hypre.size);
  if (hypre_spd_form_enabled())
    for (int i = 0; i < hypre.size; ++i)
      rho_h[i] = -rho_h[i];
  if (hypre_pin_gauge_enabled() && hypre.ilower[0] <= 0 &&
      hypre.iupper[0] >= 0 && hypre.ilower[1] <= 0 && hypre.iupper[1] >= 0) {
    const int local_x = -hypre.ilower[0];
    const int local_y = -hypre.ilower[1];
    rho_h[local_y * hypre.nx + local_x] = 0.0;
  }

  // correct boundary entries
  // just dirichlet, periodic and open are already okay!
  double b_value;
  if (hypre.bcs[0] == 1) { // Equipotential
    b_value = dx2_inv * grid.boundaries[0].C;
    for (int i = 0; i < hypre.ny; i++) {
      rho_h[i * hypre.nx] = b_value;
    }
  }
  if (hypre.bcs[1] == 1) { // Equipotential
    b_value = dx2_inv * grid.boundaries[1].C;
    for (int i = 0; i < hypre.ny; i++) {
      rho_h[(i + 1) * hypre.nx - 1] = b_value;
    }
  }
  if (hypre.bcs[2] == 1) { // Equipotential
    b_value = dx2_inv * grid.boundaries[2].C;
    for (int i = 0; i < hypre.nx; i++) {
      rho_h[i] = b_value;
    }
  }
  if (hypre.bcs[3] == 1) { // Equipotential
    b_value = dx2_inv * grid.boundaries[3].C;
    int j = hypre.nx * (hypre.ny - 1);
    for (int i = 0; i < hypre.nx; i++) {
      rho_h[j + i] = b_value;
    }
  }
}

void hypre_source_update() {
  // send, receive then write
  // hypre_exchange_ghost_rho();

  if (!compact_skip_rho_halo_exchange)
    hypre_finish_rho_halo();

  hypre_rho_update();

  // finalize vector
  HYPRE_StructVectorSetBoxValues(b_h, hypre.ilower, hypre.iupper, rho_h);
  HYPRE_StructVectorAssemble(b_h);
}

void hypre_cleanup() {
  free(V_h);
  free(rho_h);
  free(V_local);
  // Free persistent MPI requests if created
  for (auto &req : hypre_persist_v_reqs) {
    MPI_Request_free(&req);
  }
  for (auto &req : hypre_persist_rho_reqs) {
    MPI_Request_free(&req);
  }
  hypre_persist_v_reqs.clear();
  hypre_persist_rho_reqs.clear();

  HYPRE_StructPFMGDestroy(pfmg);

  HYPRE_StructMatrixDestroy(A_h);
  HYPRE_StructVectorDestroy(b_h);
  HYPRE_StructVectorDestroy(x_h);
  HYPRE_StructGridDestroy(grid_h);
  HYPRE_StructStencilDestroy(stencil_h);
}
