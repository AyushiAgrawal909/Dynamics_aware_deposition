/**
 * SPDX-License-Identifier: EUPL-1.2
 * SPDX-FileCopyrightText: 2025 LeCAD-PEG
 *
 * @file kde_deposition.cpp
 * @brief KDE (wrapped-Gaussian kernel) deposition and gather for HEXAPIC.
 *
 * Algorithm reference: Wu & Qin, arXiv:1804.11174.
 *
 * Deposition: each particle at global (x_p, y_p) position in cell units
 * deposits charge to global node (ig, jg) with separable periodic weights
 *
 *   W_x(x_p, ig) = K_h(x_p - ig) / sum_{ig'} K_h(x_p - ig')
 *   W_y(y_p, jg) = K_h(y_p - jg) / sum_{jg'} K_h(y_p - jg')
 *
 * where K_h is the wrapped Gaussian kernel of bandwidth h (cell units) on a
 * periodic domain. Separate bandwidths h_x and h_y are used in x and y.
 *
 * Gather: the electric field is derived from the gradient of phi using the
 * same kernels, ensuring exact energy conservation (variational consistency).
 *
 * Bandwidth: updated every KDE_UPDATE_INTERVAL steps via least-squares
 * cross-validation (LSCV) on random subsamples of particle positions in both
 * x and y, with an Anderson-Darling uniformity pre-test to skip LSCV when a
 * coordinate is nearly uniform. The runtime keeps separate h_x and h_y values.
 */

#include "hexapic.hpp"
#include "kde_deposition.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace hexapic::kde {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int    KDE_UPDATE_INTERVAL = 10;      // steps between h updates
static constexpr int    KDE_SUBSAMPLE_MAX   = 2048;    // max particles for LSCV
static constexpr double KDE_H_MIN_CELLS     = 0.3;
static constexpr double KDE_H_MAX_CELLS     = 20.0;
static constexpr double KDE_H_DEFAULT       = 3.0;
static constexpr double KDE_AD_THRESHOLD    = 0.05;    // p-value threshold for AD test
static constexpr int    KDE_GOLDEN_ITERS    = 50;
static constexpr double KDE_SMOOTH_ALPHA    = 0.05;    // EMA coefficient for h update
static constexpr int    KDE_RADIUS_SIGMAS   = 4;       // stencil radius = ceil(h * SIGMAS)

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
static double s_hx       = KDE_H_DEFAULT;
static double s_hy       = KDE_H_DEFAULT;
static int    s_span_x   = 0;   // periodic_span_x = Nxg - 1
static int    s_span_y   = 0;   // periodic_span_y = Nyg - 1
static bool   s_init     = false;
static int    s_steps    = 0;

// Working buffers (resized on first use)
static std::vector<double> s_rho_send;
static std::vector<double> s_rho_sum;

struct AuditSigmaSplit {
  bool enabled = false;
  double left = 0.0;
  double right = 0.0;
};

static const AuditSigmaSplit &audit_sigma_split() {
  static const AuditSigmaSplit split = [] {
    AuditSigmaSplit result;
    const char *value = std::getenv("HEXAPIC_KDE_SPLIT_SIGMA_CELLS");
    if (value == nullptr)
      return result;
    char trailing = '\0';
    if (::sscanf(value, "%lf,%lf%c", &result.left, &result.right,
                 &trailing) != 2 ||
        !std::isfinite(result.left) || !std::isfinite(result.right) ||
        result.left <= 0.0 || result.right <= 0.0) {
      if (!rmpi)
        std::cerr << "WARNING: invalid HEXAPIC_KDE_SPLIT_SIGMA_CELLS='"
                  << value << "'; split disabled\n";
      return AuditSigmaSplit{};
    }
    result.enabled = true;
    return result;
  }();
  return split;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Wrap distance to the nearest image on a periodic domain of length N.
static inline double wrap_dist(double d, int N) {
  double n = static_cast<double>(N);
  d -= std::round(d / n) * n;
  return d;
}

// Wrap index to [0, N-1].
static inline int wrap_idx(int i, int N) {
  i %= N;
  if (i < 0) i += N;
  return i;
}

// Flat global index for node (ig, jg) on a periodic domain.
// Returns -1 if out of range after wrapping (should not happen for valid inputs).
static inline int global_flat(int ig, int jg) {
  int ii = wrap_idx(ig, s_span_x);
  int jj = wrap_idx(jg, s_span_y);
  return ii * grid.Nyg + jj;
}

// ---------------------------------------------------------------------------
// Wrapped-Gaussian kernel at distance d (cell units), bandwidth h (cell units)
// on a periodic domain of length span_x.
// ---------------------------------------------------------------------------
static double wgauss_periodic(double d, double h, int span) {
  d = wrap_dist(d, span);
  double sum = 0.0;
  for (int k = -3; k <= 3; k++) {
    double dd = d + static_cast<double>(k * span);
    sum += std::exp(-0.5 * dd * dd / (h * h));
  }
  return sum / (h * std::sqrt(2.0 * M_PI));
}

// ---------------------------------------------------------------------------
// KDE stencil radius (cells) for bandwidth h.
// ---------------------------------------------------------------------------
static inline double clamp_bandwidth(double h) {
  return std::max(KDE_H_MIN_CELLS, std::min(KDE_H_MAX_CELLS, h));
}

static double audit_bandwidth_for_cell(int i_global) {
  const AuditSigmaSplit &split = audit_sigma_split();
  const int wrapped_i = wrap_idx(i_global, s_span_x);
  return clamp_bandwidth(wrapped_i < s_span_x / 2 ? split.left : split.right);
}

static inline double active_bandwidth_x_for_cell(int i_global, int j_global) {
  if (audit_sigma_split().enabled)
    return audit_bandwidth_for_cell(i_global);
  if (controller_sigma_runtime_active())
    return clamp_bandwidth(controller_sigma_x_for_global_cell(i_global, j_global));
  if (deposition_sigma_cells > 0.0)
    return clamp_bandwidth(static_cast<double>(deposition_sigma_cells));
  return clamp_bandwidth(s_hx);
}

static inline double active_bandwidth_y_for_cell(int i_global, int j_global) {
  if (audit_sigma_split().enabled)
    return audit_bandwidth_for_cell(i_global);
  if (controller_sigma_runtime_active())
    return clamp_bandwidth(controller_sigma_y_for_global_cell(i_global, j_global));
  if (deposition_sigma_cells > 0.0)
    return clamp_bandwidth(static_cast<double>(deposition_sigma_cells));
  return clamp_bandwidth(s_hy);
}

static int kde_radius(double h, int span) {
  return std::max(
      1,
      std::min(static_cast<int>(std::ceil(h * KDE_RADIUS_SIGMAS)),
               std::max(span / 2 - 1, 1)));
}

// ---------------------------------------------------------------------------
// Fill normalised x-weights W_x[0..2*R] for a particle at cell-fractional
// position px (in [0,1)) within a cell whose global corner is at i_global.
// The stencil covers nodes at offsets ox in [-R, R] from i_global.
//
// weights_out must have space for 2*R+1 entries.
// Returns R (stencil radius).
// ---------------------------------------------------------------------------
static int fill_w_periodic(int i_global, double p, double h, int span,
                           double *weights_out) {
  int R = kde_radius(h, span);
  double xg = static_cast<double>(i_global) + p;
  double norm = 0.0;
  for (int ox = -R; ox <= R; ox++) {
    double d = xg - static_cast<double>(i_global + ox);
    double w = wgauss_periodic(d, h, span);
    weights_out[ox + R] = w;
    norm += w;
  }
  if (norm > 0.0) {
    double inv = 1.0 / norm;
    for (int k = 0; k < 2 * R + 1; k++)
      weights_out[k] *= inv;
  }
  return R;
}

// ---------------------------------------------------------------------------
// Fill normalised x-weights AND their derivative w.r.t. x_p (cell units).
//
// dW_x/dx_p = W_x[i] * (d_i_w - mu_w) / h^2
// where d_i_w = wrap(x_p - ig_i) is the wrapped distance from particle to
// node and mu_w = sum_i W_x[i] * d_i_w is the weighted mean wrapped distance.
// ---------------------------------------------------------------------------
static int fill_w_deriv_periodic(int i_global, double p, double h, int span,
                                 double *weights_out, double *dweights_out) {
  int R = kde_radius(h, span);
  double xg = static_cast<double>(i_global) + p;
  int n = 2 * R + 1;
  double norm = 0.0;

  // Unnormalised weights and wrapped distances
  std::vector<double> raw(static_cast<size_t>(n));
  std::vector<double> dist_w(static_cast<size_t>(n)); // wrapped distances
  for (int ox = -R; ox <= R; ox++) {
    int k = ox + R;
    double d = xg - static_cast<double>(i_global + ox);
    double dw = wrap_dist(d, span);
    dist_w[static_cast<size_t>(k)] = dw;
    raw[static_cast<size_t>(k)] = wgauss_periodic(d, h, span);
    norm += raw[static_cast<size_t>(k)];
  }

  if (norm <= 0.0) {
    for (int k = 0; k < n; k++) {
      weights_out[k]  = 1.0 / n;
      dweights_out[k] = 0.0;
    }
    return R;
  }

  // Normalise weights and compute weighted-mean wrapped distance
  double mu_w = 0.0;
  double inv = 1.0 / norm;
  for (int k = 0; k < n; k++) {
    weights_out[k] = raw[static_cast<size_t>(k)] * inv;
    mu_w += weights_out[k] * dist_w[static_cast<size_t>(k)];
  }

  // Derivative: dW_x[k]/dx_p = W_x[k] * (dist_w[k] - mu_w) / h^2
  // Sign: E_x = -(1/dx) * sum dW/dx_p * phi, with dx_p in cell units
  double inv_h2 = 1.0 / (h * h);
  for (int k = 0; k < n; k++) {
    dweights_out[k] = weights_out[k] *
                      (dist_w[static_cast<size_t>(k)] - mu_w) * inv_h2;
  }
  return R;
}

// ---------------------------------------------------------------------------
// LSCV objective function.
// xs[0..n-1] are particle positions in cell units on [0, span).
// Silverman 1986 / Stone 1984 LSCV:
//   LSCV(h) = (1/N^2) sum_{i,j} G_{h*sqrt2}(d_ij) - (2/N(N-1)) sum_{i!=j} G_h(d_ij)
// where G_sigma(d) = 1/(sigma*sqrt(2pi)) * exp(-d^2/(2*sigma^2)) [wrapped].
// ---------------------------------------------------------------------------
static double lscv(const double *xs, int n, double h, int span) {
  if (n < 2 || h <= 0.0) return 1e30;
  double term1 = 0.0; // (1/N^2) sum_{i,j} G_{h*sqrt2}(d_ij)
  double term2 = 0.0; // (2/N(N-1)) sum_{i!=j} G_h(d_ij)
  double h2 = h * std::sqrt(2.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double d = wrap_dist(xs[i] - xs[j], span);
      term1 += wgauss_periodic(d, h2, span);
      if (i != j) term2 += wgauss_periodic(d, h, span);
    }
  }
  term1 /= static_cast<double>(n * n);
  term2 *= 2.0 / static_cast<double>(n * (n - 1));
  return term1 - term2;
}

// Golden-section search for minimum of LSCV on [h_lo, h_hi].
static double lscv_optimal_h(const double *xs, int n, double h_lo, double h_hi,
                             int span) {
  const double phi = (std::sqrt(5.0) - 1.0) * 0.5; // 1/golden_ratio
  double a = h_lo, b = h_hi;
  double c = b - phi * (b - a);
  double d = a + phi * (b - a);
  double fc = lscv(xs, n, c, span);
  double fd = lscv(xs, n, d, span);
  for (int iter = 0; iter < KDE_GOLDEN_ITERS; iter++) {
    if (fc < fd) {
      b = d; d = c; fd = fc;
      c = b - phi * (b - a);
      fc = lscv(xs, n, c, span);
    } else {
      a = c; c = d; fc = fd;
      d = a + phi * (b - a);
      fd = lscv(xs, n, d, span);
    }
    if ((b - a) < 1e-4) break;
  }
  return 0.5 * (a + b);
}

// ---------------------------------------------------------------------------
// Anderson-Darling statistic for uniformity on [0, span_x).
// xs_sorted must be pre-sorted in ascending order, n >= 2.
// ---------------------------------------------------------------------------
static double ad_statistic(const double *xs_sorted, int n, int span) {
  double L = static_cast<double>(span);
  double A2 = 0.0;
  for (int i = 0; i < n; i++) {
    double u = xs_sorted[i] / L; // map to [0,1)
    if (u <= 0.0) u = 1e-12;
    if (u >= 1.0) u = 1.0 - 1e-12;
    double log_u = std::log(u);
    double log_1mu = std::log(1.0 - u);
    int ri = 2 * (i + 1) - 1; // 2i-1, 1-indexed
    A2 += static_cast<double>(ri) * (log_u + log_1mu);
  }
  A2 = -static_cast<double>(n) - A2 / static_cast<double>(n);
  return A2;
}

// Approximate p-value for Anderson-Darling test (uniform distribution).
// Piecewise linear approximation of tabulated values.
static double ad_pvalue(double A2) {
  // significance levels: A2 thresholds for p = 0.25, 0.10, 0.05, 0.025, 0.01
  // from standard tables (asymptotic).
  static const double A2_table[] = {0.341, 0.470, 0.561, 0.660, 0.781};
  static const double p_table[]  = {0.25,  0.10,  0.05,  0.025, 0.01};
  if (A2 <= A2_table[0]) return 1.0;
  if (A2 >= A2_table[4]) return 0.005;
  for (int i = 0; i < 4; i++) {
    if (A2 >= A2_table[i] && A2 < A2_table[i + 1]) {
      double t = (A2 - A2_table[i]) / (A2_table[i + 1] - A2_table[i]);
      return p_table[i] + t * (p_table[i + 1] - p_table[i]);
    }
  }
  return 0.005;
}

static double estimate_bandwidth_axis(const std::vector<double> &coords,
                                      int span,
                                      double fallback_h) {
  int total_n = static_cast<int>(coords.size());
  if (total_n < 4 || span <= 1)
    return clamp_bandwidth(fallback_h);

  int ns = std::min(total_n, KDE_SUBSAMPLE_MAX);
  std::vector<double> sub(static_cast<size_t>(ns));
  for (int i = 0; i < ns; i++) {
    int src = static_cast<int>(static_cast<long long>(i) * total_n / ns);
    sub[static_cast<size_t>(i)] = coords[static_cast<size_t>(src)];
  }

  std::vector<double> sub_sorted = sub;
  std::sort(sub_sorted.begin(), sub_sorted.end());
  double A2 = ad_statistic(sub_sorted.data(), ns, span);
  double pval = ad_pvalue(A2);

  double h_new = fallback_h;
  if (pval > KDE_AD_THRESHOLD) {
    double mean = 0.0;
    for (double x : sub) mean += x;
    mean /= ns;
    double var = 0.0;
    for (double x : sub) var += (x - mean) * (x - mean);
    var /= ns;
    double sigma = std::sqrt(var);
    double iqr = sub_sorted[3 * ns / 4] - sub_sorted[ns / 4];
    double bw_ref = std::min(sigma, iqr / 1.34);
    if (bw_ref <= 0.0) bw_ref = sigma > 0.0 ? sigma : 1.0;
    h_new = 0.9 * bw_ref * std::pow(static_cast<double>(total_n), -0.2);
  } else {
    h_new = lscv_optimal_h(sub.data(), ns, KDE_H_MIN_CELLS, KDE_H_MAX_CELLS,
                           span);
  }
  return clamp_bandwidth(h_new);
}

// ---------------------------------------------------------------------------
// Gather particle positions from all ranks and update h_x / h_y. Must be
// collective.
// ---------------------------------------------------------------------------
static void maybe_update_bandwidth() {
  if (controller_sigma_runtime_active())
    return;
  if (deposition_sigma_cells > 0.0) {
    double h = clamp_bandwidth(static_cast<double>(deposition_sigma_cells));
    s_hx = h;
    s_hy = h;
    return;
  }

  s_steps++;
  if (s_steps % KDE_UPDATE_INTERVAL != 0) return;

  // 1. Collect local particle positions (global cell units) for electrons.
  //    Use species[0] as electrons; skip if charge >= 0.
  std::vector<double> local_xs;
  std::vector<double> local_ys;
  for (int sp_i = 0; sp_i < static_cast<int>(species.size()); sp_i++) {
    if (species[sp_i].charge >= 0) continue; // electrons have negative charge
    for (const auto &cell : cells) {
      for (const auto &p : cell.part[sp_i]) {
        double xg = static_cast<double>(grid.N0x + cell.n0x) +
                    static_cast<double>(p.x - cell.n0x);
        double yg = static_cast<double>(grid.N0y + cell.n0y) +
                    static_cast<double>(p.y - cell.n0y);
        // wrap to [0, span_x)
        while (xg >= s_span_x) xg -= s_span_x;
        while (xg < 0.0) xg += s_span_x;
        while (yg >= s_span_y) yg -= s_span_y;
        while (yg < 0.0) yg += s_span_y;
        local_xs.push_back(xg);
        local_ys.push_back(yg);
      }
    }
    break; // first electron species only
  }

  // 2. Gather sizes from all ranks to rank 0.
  int local_n = static_cast<int>(local_xs.size());
  std::vector<int> all_counts(static_cast<size_t>(nmpi), 0);
  MPI_Gather(&local_n, 1, MPI_INT,
             all_counts.data(), 1, MPI_INT,
             0, comm);

  // 3. Rank 0 gathers data and estimates one bandwidth per axis.
  double h_new_x = s_hx;
  double h_new_y = s_hy;
  if (rmpi == 0) {
    int total_n = 0;
    for (int r = 0; r < nmpi; r++) total_n += all_counts[r];

    // Build displacements for MPI_Gatherv
    std::vector<int> displs(static_cast<size_t>(nmpi), 0);
    for (int r = 1; r < nmpi; r++)
      displs[static_cast<size_t>(r)] =
          displs[static_cast<size_t>(r - 1)] + all_counts[static_cast<size_t>(r - 1)];

    std::vector<double> all_xs(static_cast<size_t>(total_n));
    std::vector<double> all_ys(static_cast<size_t>(total_n));
    MPI_Gatherv(local_xs.data(), local_n, MPI_DOUBLE,
                all_xs.data(), all_counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);
    MPI_Gatherv(local_ys.data(), local_n, MPI_DOUBLE,
                all_ys.data(), all_counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (total_n >= 4) {
      h_new_x = estimate_bandwidth_axis(all_xs, s_span_x, s_hx);
      h_new_y = estimate_bandwidth_axis(all_ys, s_span_y, s_hy);
    }
  } else {
    // Non-zero ranks: just send data, don't gather all.
    MPI_Gatherv(local_xs.data(), local_n, MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                0, comm);
    MPI_Gatherv(local_ys.data(), local_n, MPI_DOUBLE,
                nullptr, nullptr, nullptr, MPI_DOUBLE,
                0, comm);
  }

  // 4. Broadcast optimal h_x/h_y from rank 0, apply EMA smoothing.
  MPI_Bcast(&h_new_x, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&h_new_y, 1, MPI_DOUBLE, 0, comm);
  s_hx = (1.0 - KDE_SMOOTH_ALPHA) * s_hx + KDE_SMOOTH_ALPHA * h_new_x;
  s_hy = (1.0 - KDE_SMOOTH_ALPHA) * s_hy + KDE_SMOOTH_ALPHA * h_new_y;
  s_hx = clamp_bandwidth(s_hx);
  s_hy = clamp_bandwidth(s_hy);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void kde_init() {
  s_span_x = std::max(grid.Nxg - 1, 1);
  s_span_y = std::max(grid.Nyg - 1, 1);
  s_hx     = clamp_bandwidth(KDE_H_DEFAULT);
  s_hy     = clamp_bandwidth(KDE_H_DEFAULT);
  s_steps  = 0;
  s_init   = true;
  s_rho_send.clear();
  s_rho_sum.clear();
  if (!rmpi && audit_sigma_split().enabled)
    std::cout << std::setprecision(17)
              << "MOMENTUM_AUDIT kde_sigma_left=" << audit_sigma_split().left
              << " kde_sigma_right=" << audit_sigma_split().right << "\n";
}

void part2grid_kde_nonlocal() {
  if (!s_init) kde_init();

  maybe_update_bandwidth();

  size_t total_nodes =
      static_cast<size_t>(grid.Nxg) * static_cast<size_t>(grid.Nyg);
  if (s_rho_send.size() != total_nodes)
    s_rho_send.assign(total_nodes, 0.0);
  if (s_rho_sum.size() != total_nodes)
    s_rho_sum.assign(total_nodes, 0.0);
  std::fill(s_rho_send.begin(), s_rho_send.end(), 0.0);

  std::vector<double> wx;
  std::vector<double> wy;

  for (const auto &cell : cells) {
    int i_global = grid.N0x + cell.n0x;
    int j_global = grid.N0y + cell.n0y;
    double hx = active_bandwidth_x_for_cell(i_global, j_global);
    double hy = active_bandwidth_y_for_cell(i_global, j_global);
    int Rx = kde_radius(hx, s_span_x);
    int Ry = kde_radius(hy, s_span_y);
    wx.assign(static_cast<size_t>(2 * Rx + 1), 0.0);
    wy.assign(static_cast<size_t>(2 * Ry + 1), 0.0);

    for (int sp_i = 0; sp_i < static_cast<int>(species.size()); sp_i++) {
      const Species &sp = species[sp_i];
      int q = sp.charge;
      if (q == 0) continue;
      double phys_q = np2c_global * static_cast<double>(q) * q_e;
      (void)phys_q; // used only for interface diagnostics if needed

      for (const auto &p : cell.part[sp_i]) {
        double px = static_cast<double>(p.x - cell.n0x);
        double py = static_cast<double>(p.y - cell.n0y);

        // Clamp to [0, 1)
        if (px < 0.0) px = 0.0;
        if (px >= 1.0) px = 1.0 - 1e-12;
        if (py < 0.0) py = 0.0;
        if (py >= 1.0) py = 1.0 - 1e-12;

        // KDE weights in x and y
        fill_w_periodic(i_global, px, hx, s_span_x, wx.data());
        fill_w_periodic(j_global, py, hy, s_span_y, wy.data());

        // Deposit
        for (int ox = -Rx; ox <= Rx; ox++) {
          double wx_k = wx[static_cast<size_t>(ox + Rx)];
          if (wx_k == 0.0) continue;
          int ii = wrap_idx(i_global + ox, s_span_x);
          for (int oy = -Ry; oy <= Ry; oy++) {
            double wy_k = wy[static_cast<size_t>(oy + Ry)];
            if (wy_k == 0.0) continue;
            int jj = wrap_idx(j_global + oy, s_span_y);
            size_t idx = static_cast<size_t>(ii * grid.Nyg + jj);
            s_rho_send[idx] += static_cast<double>(q) * wx_k * wy_k;
          }
        }
      }
    }
  }

  MPI_Allreduce(s_rho_send.data(), s_rho_sum.data(),
                static_cast<int>(total_nodes), MPI_DOUBLE, MPI_SUM, comm);

  // Copy result to local rho[]
  for (int i = 0; i < grid.Nx; i++) {
    for (int j = 0; j < grid.Ny; j++) {
      int local_idx = i * grid.Ny + j;
      int ii = wrap_idx(grid.N0x + i, s_span_x);
      int jj = wrap_idx(grid.N0y + j, s_span_y);
      rho[static_cast<size_t>(local_idx)] =
          Sconst * s_rho_sum[static_cast<size_t>(ii * grid.Nyg + jj)];
    }
  }

  compact_skip_rho_halo_exchange = true;
}

void gather_field_kde(int i0_local, int j0_local, REAL px_r, REAL py_r,
                      REAL *Ef) {
  if (!s_init) kde_init();
  ensure_phi_global();

  int i_global = grid.N0x + i0_local;
  int j_global = grid.N0y + j0_local;
  double px = static_cast<double>(px_r);
  double py = static_cast<double>(py_r);
  double hx = active_bandwidth_x_for_cell(i_global, j_global);
  double hy = active_bandwidth_y_for_cell(i_global, j_global);

  int Rx = kde_radius(hx, s_span_x);
  int Ry = kde_radius(hy, s_span_y);
  int n_wx = 2 * Rx + 1;
  int n_wy = 2 * Ry + 1;
  std::vector<double> wx(static_cast<size_t>(n_wx));
  std::vector<double> wy(static_cast<size_t>(n_wy));
  std::vector<double> dwx(static_cast<size_t>(n_wx));
  std::vector<double> dwy(static_cast<size_t>(n_wy));
  fill_w_deriv_periodic(i_global, px, hx, s_span_x, wx.data(), dwx.data());
  fill_w_deriv_periodic(j_global, py, hy, s_span_y, wy.data(), dwy.data());

  double Ex = 0.0, Ey = 0.0;
  for (int ox = -Rx; ox <= Rx; ox++) {
    int kx = ox + Rx;
    double wkx  = wx[static_cast<size_t>(kx)];
    double dwkx = dwx[static_cast<size_t>(kx)];
    for (int oy = -Ry; oy <= Ry; oy++) {
      int ky = oy + Ry;
      double wky  = wy[static_cast<size_t>(ky)];
      double dwky = dwy[static_cast<size_t>(ky)];
      double phi = get_phi_global(i_global + ox, j_global + oy);

      // E_x contribution: -(dW_x/dx_p) * W_y * phi / dx
      Ex += dwkx * wky * phi;

      // E_y contribution: -W_x * (dW_y/dy_p) * phi / dy
      Ey += wkx * dwky * phi;
    }
  }

  Ef[0] = static_cast<REAL>(-Ex / grid.dx);
  Ef[1] = static_cast<REAL>(-Ey / grid.dy);
}

} // namespace hexapic::kde
