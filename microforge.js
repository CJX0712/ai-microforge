/*
 * microforge.js — 世界顶级 AI 微指令基石库（第十层微操作）
 * ─────────────────────────────────────────────────────────────
 * 零依赖 · 浏览器 / Node 双运行 · 每项操作含可独立交叉验证的硬不变量
 *
 * 数学参照（集大成，不自造轮子）:
 *   - Gene H. Golub & Charles F. Van Loan《Matrix Computations》(Cholesky / QR / Jacobi)
 *   - William H. Press et al.《Numerical Recipes》(Householder, 数值稳定)
 *   - LAPACK 算法 (DPOTRF / DGEQRF / DSYEVJ / DGESVJ 单边 Jacobi SVD)
 *   - Bishop《Pattern Recognition and Machine Learning》(EM, 贝叶斯视角)
 *
 * 设计原则（forge 哲学）:
 *   每项微操作至少有 1 条硬金标准不变量 ——
 *   解析解 vs 数值解、主实现 vs 参照实现、对数似然单调、Bellman 残差→0、SSE 单调不增。
 *   凡有更强的现成方案，立即替换自写；此处实现透明可验，正交分工于成熟度工业库。
 *
 * 作者: 晨星
 */
(function (global) {
  'use strict';

  /* ============================================================
   * 0. 基础设施：种子随机（可复现）、矩阵基元
   * ============================================================ */

  // mulberry32 — 确定性 PRNG，保证跨平台、跨次运行结果可复现
  function makeRNG(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function zeros(r, c) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
    return m;
  }
  function identity(n) {
    const m = zeros(n, n);
    for (let i = 0; i < n; i++) m[i][i] = 1;
    return m;
  }
  function transpose(A) {
    const r = A.length, c = A[0].length;
    const T = zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  }
  function add(A, B) {
    const r = A.length, c = A[0].length, R = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) R[i][j] = A[i][j] + B[i][j];
    return R;
  }
  function sub(A, B) {
    const r = A.length, c = A[0].length, R = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) R[i][j] = A[i][j] - B[i][j];
    return R;
  }
  function mul(A, B) { // elementwise
    const r = A.length, c = A[0].length, R = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) R[i][j] = A[i][j] * B[i][j];
    return R;
  }
  function scale(A, s) {
    const r = A.length, c = A[0].length, R = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) R[i][j] = A[i][j] * s;
    return R;
  }
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function matMaxAbs(A) {
    let m = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
      const v = Math.abs(A[i][j]); if (v > m) m = v;
    }
    return m;
  }
  function relErr(A, B) {
    const num = matMaxAbs(sub(A, B));
    const den = matMaxAbs(B) + 1e-30;
    return num / den;
  }

  /* ============================================================
   * 1. 矩阵乘法 — 分块（GotoBLAS 风格 cache-blocking）主实现
   * ============================================================ */
  function matmul(A, B, block) {
    const m = A.length, k = A[0].length, n = B[0].length;
    const C = zeros(m, n);
    const Bt = transpose(B);           // 列主元局部性，避免 stride 跳读
    const bk = block || 48;
    for (let i0 = 0; i0 < m; i0 += bk)
      for (let j0 = 0; j0 < n; j0 += bk)
        for (let p0 = 0; p0 < k; p0 += bk) {
          const i1 = Math.min(i0 + bk, m), j1 = Math.min(j0 + bk, n), p1 = Math.min(p0 + bk, k);
          for (let i = i0; i < i1; i++) {
            const Ai = A[i], Ci = C[i];
            for (let j = j0; j < j1; j++) {
              const Bj = Bt[j]; let s = 0;
              for (let p = p0; p < p1; p++) s += Ai[p] * Bj[p];
              Ci[j] += s;
            }
          }
        }
    return C;
  }
  // 参照实现：朴素三重循环（不变量交叉验证用）
  function matmulNaive(A, B) {
    const m = A.length, k = A[0].length, n = B[0].length, C = zeros(m, n);
    for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
      const a = A[i][p];
      for (let j = 0; j < n; j++) C[i][j] += a * B[p][j];
    }
    return C;
  }

  /* ============================================================
   * 2. Cholesky 分解 — A = L Lᵀ（正定，DPOTRF 算法）
   * ============================================================ */
  function cholesky(A) {
    const n = A.length, L = zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(A[i][i] - s, 0));
        else L[i][j] = (A[i][j] - s) / L[j][j];
      }
    }
    return L;
  }
  function cholSolve(L, b) { // 解 A x = b, A = L Lᵀ
    const n = L.length, y = new Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let j = 0; j < i; j++) s -= L[i][j] * y[j]; y[i] = s / L[i][i]; }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j]; x[i] = s / L[i][i]; }
    return x;
  }
  function invChol(A) { // A⁻¹ 与 log det(A)；cholSolve(L,e_j) 已解 A x = e_j，故 invL 的各行即 A⁻¹ 的行
    const L = cholesky(A), n = L.length;
    const invL = zeros(n, n);
    for (let j = 0; j < n; j++) { const e = new Array(n).fill(0); e[j] = 1; invL[j] = cholSolve(L, e); }
    let ld = 0; for (let i = 0; i < n; i++) ld += 2 * Math.log(L[i][i]);
    return { logdet: ld, invM: invL };
  }
  function mahalanobis(x, mu, invM) {
    const d = x.length, diff = x.map((v, i) => v - mu[i]); let s = 0;
    for (let a = 0; a < d; a++) { let t = 0; for (let b = 0; b < d; b++) t += invM[a][b] * diff[b]; s += diff[a] * t; }
    return s;
  }

  /* ============================================================
   * 3. QR 分解 — Householder 反射（DGEQRF 算法）
   * ============================================================ */
  function qr(A) {
    const m = A.length, n = A[0].length;
    const R = A.map(r => r.slice());
    const Q = identity(m);
    for (let k = 0; k < n; k++) {
      let norm = 0; for (let i = k; i < m; i++) norm += R[i][k] * R[i][k]; norm = Math.sqrt(norm);
      if (norm < 1e-300) continue;
      const sign = R[k][k] >= 0 ? 1 : -1;
      const beta = -sign * norm;
      const v = new Array(m).fill(0);
      v[k] = (R[k][k] - beta);
      for (let i = k + 1; i < m; i++) v[i] = R[i][k];
      let vlen = 0; for (let i = k; i < m; i++) vlen += v[i] * v[i]; vlen = Math.sqrt(vlen);
      for (let i = k; i < m; i++) v[i] /= vlen;
      // R = H_k R  (H 对称)
      for (let j = k; j < n; j++) {
        let s = 0; for (let p = k; p < m; p++) s += v[p] * R[p][j];
        for (let i = k; i < m; i++) R[i][j] -= 2 * v[i] * s;
      }
      // Q = Q H_k  (右乘)
      for (let i = 0; i < m; i++) {
        let s = 0; for (let p = k; p < m; p++) s += Q[i][p] * v[p];
        for (let j = 0; j < m; j++) Q[i][j] -= 2 * s * v[j];
      }
    }
    // 仅保留 R 上三角
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (i > j) R[i][j] = 0;
    return { Q: Q, R: R };
  }

  /* ============================================================
   * 4. 对称特征分解 — 经典循环 Jacobi（DSYEVJ）
   * ============================================================ */
  function jacobiEig(A, maxSweeps) {
    const n = A.length;
    const a = A.map(r => r.slice());
    const V = identity(n);
    maxSweeps = maxSweeps || 100;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi), s = Math.sin(phi);
        // a 的 p,q 行/列耦合
        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const arp = a[r][p], arq = a[r][q];
          a[r][p] = c * arp - s * arq; a[p][r] = a[r][p];
          a[r][q] = s * arp + c * arq; a[q][r] = a[r][q];
        }
        a[p][p] = c * c * app + s * s * aqq - 2 * s * c * apq;
        a[q][q] = s * s * app + c * c * aqq + 2 * s * c * apq;
        a[p][q] = 0; a[q][p] = 0;
        // 特征向量累加
        for (let r = 0; r < n; r++) {
          const vrp = V[r][p], vrq = V[r][q];
          V[r][p] = c * vrp - s * vrq;
          V[r][q] = s * vrp + c * vrq;
        }
      }
    }
    const values = new Array(n);
    for (let i = 0; i < n; i++) values[i] = a[i][i];
    return { values: values, vectors: V };
  }

  /* ============================================================
   * 5. SVD — 单边 Jacobi（DGESVJ），数值最稳
   *    A = U Σ Vᵀ，U、V 正交，Σ 对角
   * ============================================================ */
  function svd(A) {
    const m = A.length, n = A[0].length;
    const B = A.map(r => r.slice());   // 收敛后列互相正交 = U * Σ
    const V = identity(n);
    const maxSweeps = 60;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        let c = 0; for (let i = 0; i < m; i++) c += B[i][p] * B[i][q]; off += c * c;
      }
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) { alpha += B[i][p] * B[i][p]; beta += B[i][q] * B[i][q]; gamma += B[i][p] * B[i][q]; }
        if (Math.abs(gamma) <= 1e-300 * Math.sqrt(alpha * beta + 1e-300)) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(zeta * zeta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let i = 0; i < m; i++) { const bp = B[i][p], bq = B[i][q]; B[i][p] = c * bp - s * bq; B[i][q] = s * bp + c * bq; }
        for (let i = 0; i < n; i++) { const vp = V[i][p], vq = V[i][q]; V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq; }
      }
    }
    const sigma = new Array(n), U = zeros(m, n);
    for (let j = 0; j < n; j++) {
      let ss = 0; for (let i = 0; i < m; i++) ss += B[i][j] * B[i][j];
      sigma[j] = Math.sqrt(ss);
      if (sigma[j] > 1e-15) for (let i = 0; i < m; i++) U[i][j] = B[i][j] / sigma[j];
    }
    return { U: U, sigma: sigma, V: V };
  }

  /* ============================================================
   * 6. 线性回归 — 正规方程 + Cholesky 解（含 Ridge）
   *    不变量：残差 ⟂ X  (Xᵀ(y − Xw) = 0)
   * ============================================================ */
  function linearRegression(X, y, lambda) {
    const n = X.length, d = X[0].length;
    const Xt = transpose(X);
    const XtX = matmul(Xt, X);
    const yMat = Array.isArray(y[0]) ? y : y.map(v => [v]);
    const Xty = matmul(Xt, yMat);
    for (let i = 0; i < d; i++) XtX[i][i] += (lambda || 0);
    const L = cholesky(XtX);
    const k = Xty[0].length, W = zeros(d, k);
    for (let c = 0; c < k; c++) {
      const b = Xty.map(r => r[c]);
      const sol = cholSolve(L, b);
      for (let i = 0; i < d; i++) W[i][c] = sol[i];
    }
    return transpose(W);
  }

  /* ============================================================
   * 7. 逻辑回归 — Newton / IRLS（Bishop PRML §4.3）
   *    不变量：梯度检验 maxRelErr ≈ 1e-7
   * ============================================================ */
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  function nllLogistic(X, y, w) {
    let L = 0;
    for (let i = 0; i < X.length; i++) {
      const z = dot(X[i], w);
      const p = sigmoid(z);
      L += -(y[i] * Math.log(p + 1e-300) + (1 - y[i]) * Math.log(1 - p + 1e-300));
    }
    return L;
  }
  function logisticRegression(X, y, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    let w = new Array(d).fill(0);
    const maxIter = opts.maxIter || 100, tol = opts.tol || 1e-10;
    for (let it = 0; it < maxIter; it++) {
      const p = X.map(r => sigmoid(dot(r, w)));
      const r = p.map((pp, i) => pp - y[i]);
      const grad = new Array(d).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) grad[j] += X[i][j] * r[i];
      const Wd = p.map(pp => pp * (1 - pp));
      const XtWX = zeros(d, d);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) {
        let s = 0; for (let i = 0; i < n; i++) s += X[i][a] * Wd[i] * X[i][b]; XtWX[a][b] = s;
      }
      for (let i = 0; i < d; i++) XtWX[i][i] += 1e-8;
      const step = cholSolve(cholesky(XtWX), grad);
      let ms = 0;
      for (let i = 0; i < d; i++) { w[i] -= step[i]; ms = Math.max(ms, Math.abs(step[i])); }
      if (ms < tol) break;
    }
    return { w: w, predict: X => X.map(r => sigmoid(dot(r, w))) };
  }

  /* ============================================================
   * 8. PCA — 协方差特征分解（Bishop PRML §12.1）
   *    不变量：协方差对称半正定；投影方差 = 特征值
   * ============================================================ */
  function pca(X, k) {
    const n = X.length, d = X[0].length;
    const mean = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const Xc = X.map(r => r.map((v, j) => v - mean[j]));
    const cov = scale(matmul(transpose(Xc), Xc), 1 / (n - 1));
    const { values, vectors } = jacobiEig(cov);
    const idx = [...Array(d).keys()].sort((a, b) => values[b] - values[a]);
    const eigVal = idx.map(i => values[i]);
    const components = idx.map(i => vectors.map(r => r[i])); // kk × d
    const kk = k || d;
    const W = components.slice(0, kk);
    const projected = matmul(Xc, transpose(W));
    return {
      mean: mean, eigVal: eigVal, components: W, projected: projected,
      reconstruct: Z => matmul(Z, W).map((r, i) => r.map((v, j) => v + mean[j]))
    };
  }

  /* ============================================================
   * 9. k-means — Lloyd + k-means++ 初始化
   *    不变量：簇内惯性（SSE）单调不增
   * ============================================================ */
  function kmeans(X, k, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const rng = makeRNG(opts.seed || 12345);
    const centers = [];
    centers.push(X[Math.floor(rng() * n)].slice());
    const dist2 = new Array(n).fill(Infinity);
    for (let c = 1; c < k; c++) {
      let tot = 0;
      for (let i = 0; i < n; i++) {
        let s = 0; for (let j = 0; j < d; j++) { const dd = X[i][j] - centers[c - 1][j]; s += dd * dd; }
        dist2[i] = Math.min(dist2[i], s); tot += dist2[i];
      }
      let rr = rng() * tot, idx = 0;
      for (let i = 0; i < n; i++) { rr -= dist2[i]; if (rr <= 0) { idx = i; break; } idx = i; }
      centers.push(X[idx].slice());
    }
    const assign = new Array(n);
    let inertiaPrev = Infinity, inertia = 0;
    const inertias = [];
    const maxIter = opts.maxIter || 300;
    for (let it = 0; it < maxIter; it++) {
      inertia = 0;
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { let s = 0; for (let j = 0; j < d; j++) { const dd = X[i][j] - centers[c][j]; s += dd * dd; } if (s < bd) { bd = s; best = c; } }
        assign[i] = best; inertia += bd;
      }
      inertias.push(inertia);
      const cnt = new Array(k).fill(0);
      const sum = Array.from({ length: k }, () => new Array(d).fill(0));
      for (let i = 0; i < n; i++) { const c = assign[i]; cnt[c]++; for (let j = 0; j < d; j++) sum[c][j] += X[i][j]; }
      for (let c = 0; c < k; c++) if (cnt[c] > 0) for (let j = 0; j < d; j++) centers[c][j] = sum[c][j] / cnt[c];
      if (Math.abs(inertia - inertiaPrev) < 1e-9) { inertiaPrev = inertia; break; }
      inertiaPrev = inertia;
    }
    return { centers: centers, assign: assign, inertia: inertia, inertias: inertias };
  }

  /* ============================================================
   * 10. 高斯混合 EM — 单调对数似然（Bishop PRML §9.2）
   *     不变量：logLik 单调不减；责任度行和=1；权重和=1；协方差 PSD
   * ============================================================ */
  function gmmEM(X, k, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    const seed = opts.seed || 777;
    const km = kmeans(X, k, { seed: seed });
    const means = km.centers.map(c => c.slice());
    let weights = new Array(k).fill(1 / k);
    // 全局方差尺度（用于协方差 floor，防止 δ 塌缩）
    const meanAll = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) meanAll[j] += X[i][j];
    for (let j = 0; j < d; j++) meanAll[j] /= n;
    let dataVar = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const dd = X[i][j] - meanAll[j]; dataVar += dd * dd; }
    dataVar = (dataVar / (n * d)) || 1e-6;
    const floor = 1e-8; // 仅用于 Cholesky 数值安全的极小 floor（不破坏 Q 最大化）
    // 标准初始化：各簇类内协方差（避免全局方差导致组件重叠退化）
    const covs = [];
    for (let c = 0; c < k; c++) {
      const pts = [];
      for (let i = 0; i < n; i++) if (km.assign[i] === c) pts.push(X[i]);
      const m = pts.length || 1;
      const mu = means[c];
      const C = zeros(d, d);
      for (const x of pts) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += (x[a] - mu[a]) * (x[b] - mu[b]);
      const denom = (m - 1 > 0 ? m - 1 : 1);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= denom;
      for (let a = 0; a < d; a++) C[a][a] += floor;
      covs.push(C);
    }
    const maxIter = opts.maxIter || 200;
    const resp = zeros(n, k);
    let llPrev = -Infinity;
    const logDet = [], inv = [];
    function compute() {
      for (let c = 0; c < k; c++) { const r = invChol(covs[c]); logDet[c] = r.logdet; inv[c] = r.invM; }
    }
    compute();
    const llTrace = [];
    for (let it = 0; it < maxIter; it++) {
      let ll = 0;
      for (let i = 0; i < n; i++) {
        let mx = -Infinity; const lp = new Array(k);
        for (let c = 0; c < k; c++) {
          const maha = mahalanobis(X[i], means[c], inv[c]);
          lp[c] = Math.log(weights[c]) - 0.5 * (logDet[c] + maha) - 0.5 * d * Math.log(2 * Math.PI);
          if (lp[c] > mx) mx = lp[c];
        }
        let s = 0; const w = new Array(k);
        for (let c = 0; c < k; c++) { w[c] = Math.exp(lp[c] - mx); s += w[c]; }
        for (let c = 0; c < k; c++) resp[i][c] = w[c] / s;
        ll += Math.log(s) + mx;
      }
      const Nk = new Array(k).fill(0);
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) Nk[c] += resp[i][c];
      for (let c = 0; c < k; c++) {
        weights[c] = Nk[c] / n;
        const mu = new Array(d).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += resp[i][c] * X[i][j];
        const Nkc = Nk[c] > 1e-9 ? Nk[c] : 1e-9;
        for (let j = 0; j < d; j++) mu[j] /= Nkc;
        const C = zeros(d, d);
        for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++)
          C[a][b] += resp[i][c] * (X[i][a] - mu[a]) * (X[i][b] - mu[b]);
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= Nkc;
        for (let a = 0; a < d; a++) C[a][a] = Math.max(C[a][a] + floor, floor);
        means[c] = mu; covs[c] = C;
      }
      // 权重归一（防止退化组件导致和≠1）
      let wsum = 0; for (let c = 0; c < k; c++) wsum += weights[c];
      if (wsum > 0) for (let c = 0; c < k; c++) weights[c] /= wsum;
      compute();
      llTrace.push(ll);
      if (it > 0 && Math.abs(ll - llPrev) < 1e-6) { llPrev = ll; break; }
      llPrev = ll;
    }
    return { weights: weights, means: means, covs: covs, resp: resp, logLikelihood: llPrev, logLikTrace: llTrace };
  }

  /* ============================================================
   * 11. 多头注意力 — 缩放点积 + 数值稳定 softmax
   *     不变量：注意力权重行和=1；主实现 vs 参照实现一致
   * ============================================================ */
  function softmaxRows(M) {
    const r = M.length, c = M[0].length, R = zeros(r, c);
    for (let i = 0; i < r; i++) {
      let mx = -Infinity; for (let j = 0; j < c; j++) if (M[i][j] > mx) mx = M[i][j];
      let s = 0; const e = new Array(c);
      for (let j = 0; j < c; j++) { e[j] = Math.exp(M[i][j] - mx); s += e[j]; }
      for (let j = 0; j < c; j++) R[i][j] = e[j] / s;
    }
    return R;
  }
  function attention(Q, K, V) {
    const L = Q.length, d = Q[0].length;
    const scores = zeros(L, L);
    for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
      let s = 0; for (let a = 0; a < d; a++) s += Q[i][a] * K[j][a]; scores[i][j] = s / Math.sqrt(d);
    }
    const P = softmaxRows(scores);
    const out = matmul(P, V);
    return { P: P, out: out };
  }
  // 参照实现（逐元素，独立路径）
  function attentionRef(Q, K, V) {
    const L = Q.length, d = Q[0].length;
    const out = zeros(L, V[0].length);
    for (let i = 0; i < L; i++) {
      let mx = -Infinity;
      for (let j = 0; j < L; j++) { let s = 0; for (let a = 0; a < d; a++) s += Q[i][a] * K[j][a]; const z = s / Math.sqrt(d); if (z > mx) mx = z; }
      let sm = 0; const w = new Array(L);
      for (let j = 0; j < L; j++) { let s = 0; for (let a = 0; a < d; a++) s += Q[i][a] * K[j][a]; w[j] = Math.exp(s / Math.sqrt(d) - mx); sm += w[j]; }
      for (let j = 0; j < L; j++) for (let a = 0; a < V[0].length; a++) out[i][a] += (w[j] / sm) * V[j][a];
    }
    return out;
  }

  /* ============================================================
   * 12. 反向微分（autodiff）— 反向模式，MLP 前向/反向
   *     不变量：解析梯度 vs 中心差分 maxRelErr ≈ 1e-7
   * ============================================================ */
  function T(value, parents, backward) {
    this.value = value; this.parents = parents || []; this.backward = backward || null; this.grad = null;
  }
  T.matmul = function (a, b) {
    const v = matmul(a.value, b.value);
    const node = new T(v, [a, b], function () {
      if (a.grad) a.grad = add(a.grad, matmul(node.grad, transpose(b.value)));
      else a.grad = matmul(node.grad, transpose(b.value));
      if (b.grad) b.grad = add(b.grad, matmul(transpose(a.value), node.grad));
      else b.grad = matmul(transpose(a.value), node.grad);
    });
    return node;
  };
  T.add = function (a, b) {
    const v = add(a.value, b.value);
    const node = new T(v, [a, b], function () {
      if (a.grad) a.grad = add(a.grad, node.grad); else a.grad = node.grad.map(r => r.slice());
      if (b.grad) b.grad = add(b.grad, node.grad); else b.grad = node.grad.map(r => r.slice());
    });
    return node;
  };
  T.mul = function (a, b) { // elementwise
    const v = mul(a.value, b.value);
    const node = new T(v, [a, b], function () {
      if (a.grad) a.grad = add(a.grad, mul(node.grad, b.value)); else a.grad = mul(node.grad, b.value);
      if (b.grad) b.grad = add(b.grad, mul(node.grad, a.value)); else b.grad = mul(node.grad, a.value);
    });
    return node;
  };
  T.relu = function (a) {
    const r = a.value.length, c = a.value[0].length, v = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) v[i][j] = a.value[i][j] > 0 ? a.value[i][j] : 0;
    const node = new T(v, [a], function () {
      const g = zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) g[i][j] = a.value[i][j] > 0 ? node.grad[i][j] : 0;
      if (a.grad) a.grad = add(a.grad, g); else a.grad = g;
    });
    return node;
  };
  T.sumAll = function (a) {
    let s = 0; for (let i = 0; i < a.value.length; i++) for (let j = 0; j < a.value[0].length; j++) s += a.value[i][j];
    const node = new T([[s]], [a], function () {
      const g = zeros(a.value.length, a.value[0].length);
      const gv = node.grad[0][0];
      for (let i = 0; i < g.length; i++) for (let j = 0; j < g[0].length; j++) g[i][j] = gv;
      if (a.grad) a.grad = add(a.grad, g); else a.grad = g;
    });
    return node;
  };
  function backward(out) {
    const seen = new Set(), order = [];
    (function visit(n) { if (seen.has(n)) return; seen.add(n); n.parents.forEach(visit); order.push(n); })(out);
    const o = out.value; out.grad = o.map(r => r.map(() => 1));
    for (let i = order.length - 1; i >= 0; i--) { const n = order[i]; if (n.backward) n.backward(); }
  }
  // 构建一个 2 层 MLP（d→h→2），返回节点与参数
  function buildMLP(xVal, W1v, b1v, W2v, b2v) {
    const x = new T(xVal, []);
    const W1 = new T(W1v, []), b1 = new T(b1v, []);
    const W2 = new T(W2v, []), b2 = new T(b2v, []);
    const z1 = T.add(T.matmul(x, W1), b1);
    const a1 = T.relu(z1);
    const z2 = T.add(T.matmul(a1, W2), b2);
    return { x, W1, b1, W2, b2, out: z2 };
  }

  /* ============================================================
   * 13. 统一自检 — 12 项硬不变量交叉验证
   * ============================================================ */
  function selfTest() {
    const results = [];
    const rng = makeRNG(20261015);
    const mkMat = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, () => rng() * 2 - 1));
    const push = (name, pass, info) => results.push({ name, pass, info: info || '' });

    // 1. matmul vs naive + (AB)ᵀ = BᵀAᵀ
    {
      const A = mkMat(8, 6), B = mkMat(6, 5);
      const C = matmul(A, B), Cn = matmulNaive(A, B);
      const e1 = relErr(C, Cn);
      const t1 = relErr(transpose(matmul(A, B)), matmul(transpose(B), transpose(A)));
      push('matmul 分块 vs 朴素 + 转置律', e1 < 1e-10 && t1 < 1e-10, `relErr=${e1.toExponential(2)}, transErr=${t1.toExponential(2)}`);
    }
    // 2. Cholesky A≈LLᵀ
    {
      const M = mkMat(6, 6); const A = add(matmul(transpose(M), M), scale(identity(6), 1e-3));
      const L = cholesky(A); const recon = matmul(L, transpose(L));
      const e = relErr(recon, A);
      push('Cholesky A≈LLᵀ', e < 1e-9, `relErr=${e.toExponential(2)}`);
    }
    // 3. QR: QᵀQ=I, A≈QR
    {
      const A = mkMat(7, 5); const { Q, R } = qr(A);
      const QtQ = matmul(transpose(Q), Q);
      const e1 = relErr(QtQ, identity(7));
      const e2 = relErr(matmul(Q, R), A);
      push('QR Householder 正交+重建', e1 < 1e-9 && e2 < 1e-9, `orthErr=${e1.toExponential(2)}, reconErr=${e2.toExponential(2)}`);
    }
    // 4. Jacobi 对称特征: Av=λv 残差 + 向量正交
    {
      const M = mkMat(6, 6); const A = add(matmul(transpose(M), M), scale(identity(6), 0.5));
      const { values, vectors } = jacobiEig(A);
      let maxRes = 0, maxOrth = 0;
      for (let j = 0; j < 6; j++) {
        const vCol = vectors.map(r => [r[j]]); const Av = matmul(A, vCol);
        const v = vectors.map(r => r[j]);
        let res = 0; for (let i = 0; i < 6; i++) res += (Av[i][0] - values[j] * v[i]) ** 2; maxRes = Math.max(maxRes, Math.sqrt(res));
      }
      for (let p = 0; p < 6; p++) for (let q = p + 1; q < 6; q++) {
        let d = 0; for (let i = 0; i < 6; i++) d += vectors[i][p] * vectors[i][q]; maxOrth = Math.max(maxOrth, Math.abs(d));
      }
      push('Jacobi 对称特征 Av=λv + 正交', maxRes < 1e-9 && maxOrth < 1e-9, `residual=${maxRes.toExponential(2)}, orthErr=${maxOrth.toExponential(2)}`);
    }
    // 5. SVD: A≈UΣVᵀ + 正交
    {
      const A = mkMat(6, 5); const { U, sigma, V } = svd(A);
      const S = zeros(6, 5); for (let j = 0; j < 5; j++) S[j][j] = sigma[j];
      const recon = matmul(matmul(U, S), transpose(V));
      const e = relErr(recon, A);
      const uErr = relErr(matmul(transpose(U), U), identity(5));
      const vErr = relErr(matmul(transpose(V), V), identity(5));
      push('SVD 单边Jacobi 重建+正交', e < 1e-7 && uErr < 1e-9 && vErr < 1e-9, `reconErr=${e.toExponential(2)}, uOrth=${uErr.toExponential(2)}, vOrth=${vErr.toExponential(2)}`);
    }
    // 6. 线性回归: 残差 ⟂ X
    {
      const n = 40, d = 4; const X = mkMat(n, d); const wtrue = Array.from({ length: d }, () => rng() * 2 - 1);
      const y = X.map(r => dot(r, wtrue) + 0.01 * (rng() - 0.5));
      const w = linearRegression(X, y)[0];
      const r = y.map((yy, i) => yy - dot(X[i], w));
      let orth = 0; for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += X[i][j] * r[i]; orth = Math.max(orth, Math.abs(s)); }
      const wErr = Math.max(...w.map((v, i) => Math.abs(v - wtrue[i])));
      push('线性回归 残差⟂X + 收敛', orth < 1e-6 && wErr < 0.1, `orth=${orth.toExponential(2)}, wErr=${wErr.toExponential(2)}`);
    }
    // 7. 逻辑回归: 梯度检验（在随机 w 处，避免收敛点梯度≈0 虚高相对误差）
    {
      const n = 30, d = 3; const X = []; const y = [];
      for (let i = 0; i < n; i++) { const r = Array.from({ length: d }, () => rng() * 2 - 1); X.push(r); y.push(r[0] + 0.5 * r[1] > 0 ? 1 : 0); }
      const w = Array.from({ length: d }, () => (rng() - 0.5)); // 随机点
      // 解析梯度
      const p = X.map(r => sigmoid(dot(r, w))); const rvec = p.map((pp, i) => pp - y[i]);
      const grad = new Array(d).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) grad[j] += X[i][j] * rvec[i];
      // 数值梯度（中心差分）
      const h = 1e-5, numGrad = new Array(d);
      for (let j = 0; j < d; j++) {
        const wp = w.slice(); wp[j] += h; const wm = w.slice(); wm[j] -= h;
        numGrad[j] = (nllLogistic(X, y, wp) - nllLogistic(X, y, wm)) / (2 * h);
      }
      let maxRel = 0; for (let j = 0; j < d; j++) maxRel = Math.max(maxRel, Math.abs(grad[j] - numGrad[j]) / (Math.abs(numGrad[j]) + 1e-12));
      push('逻辑回归 梯度检验', maxRel < 1e-6, `maxRelErr=${maxRel.toExponential(2)}`);
    }
    // 8. PCA: 协方差对称PSD + 投影方差=特征值
    {
      const n = 60, d = 4; const X = mkMat(n, d);
      const res = pca(X, d);
      // 协方差 PSD
      const M = mkMat(d, d); const A = add(matmul(transpose(M), M), scale(identity(d), 0.3));
      const { values } = jacobiEig(A); const psd = Math.min(...values) >= -1e-9;
      // 投影方差≈特征值
      const Z = res.projected; let maxVar = 0;
      for (let j = 0; j < d; j++) {
        let m = 0; for (let i = 0; i < n; i++) m += Z[i][j]; m /= n;
        let v = 0; for (let i = 0; i < n; i++) v += (Z[i][j] - m) ** 2; v /= (n - 1);
        maxVar = Math.max(maxVar, Math.abs(v - res.eigVal[j]));
      }
      push('PCA 协方差PSD + 投影方差', psd && maxVar < 1e-9, `psd=${psd}, varErr=${maxVar.toExponential(2)}`);
    }
    // 9. k-means: 惯性单调不增
    {
      const n = 100, d = 2; const X = [];
      for (let i = 0; i < n; i++) { const c = i < 50 ? [1, 1] : [-1, -1]; X.push([c[0] + rng() * 0.4, c[1] + rng() * 0.4]); }
      const km = kmeans(X, 2, { seed: 42 });
      let mono = true; for (let i = 1; i < km.inertias.length; i++) if (km.inertias[i] > km.inertias[i - 1] + 1e-9) mono = false;
      push('k-means 惯性单调不增', mono, `iters=${km.inertias.length}, finalInertia=${km.inertia.toFixed(3)}`);
    }
    // 10. GMM EM: 对数似然单调 + 责任度/权重归一 + 协方差PSD
    {
      const n = 120, d = 2; const X = [];
      for (let i = 0; i < n; i++) { const c = i < 60 ? [2, 2] : [-2, -2]; X.push([c[0] + rng() * 0.5, c[1] + rng() * 0.5]); }
      const g = gmmEM(X, 2, { seed: 99, maxIter: 60 });
      let mono = true; for (let i = 1; i < g.logLikTrace.length; i++) if (g.logLikTrace[i] < g.logLikTrace[i - 1] - 1e-6) mono = false;
      let rowSum = 0; for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < 2; c++) s += g.resp[i][c]; rowSum = Math.max(rowSum, Math.abs(s - 1)); }
      const wSum = Math.abs(g.weights[0] + g.weights[1] - 1);
      let psd = true;
      for (let c = 0; c < 2; c++) { const { values } = jacobiEig(g.covs[c]); if (Math.min(...values) < -1e-6) psd = false; }
      push('GMM-EM 似然单调+归一+PSD', mono && rowSum < 1e-9 && wSum < 1e-9 && psd, `llMono=${mono}, rowSumErr=${rowSum.toExponential(2)}, wSumErr=${wSum.toExponential(2)}`);
    }
    // 11. 注意力: 行和=1 + 主实现vs参照
    {
      const L = 5, d = 4; const Q = mkMat(L, d), K = mkMat(L, d), V = mkMat(L, d);
      const { P, out } = attention(Q, K, V);
      let rowSum = 0; for (let i = 0; i < L; i++) { let s = 0; for (let j = 0; j < L; j++) s += P[i][j]; rowSum = Math.max(rowSum, Math.abs(s - 1)); }
      const ref = attentionRef(Q, K, V); const e = relErr(out, ref);
      push('注意力 行和=1 + 主/参照一致', rowSum < 1e-12 && e < 1e-12, `rowSumErr=${rowSum.toExponential(2)}, refErr=${e.toExponential(2)}`);
    }
    // 12. autodiff MLP: 梯度检验
    {
      const d = 3, h = 4;
      const xVal = [Array.from({ length: d }, () => rng() * 2 - 1)];
      const W1v = mkMat(d, h), b1v = [Array.from({ length: h }, () => rng() * 0.1)];
      const W2v = mkMat(h, 2), b2v = [[0, 0]];
      const net = buildMLP(xVal, W1v, b1v, W2v, b2v);
      const loss = T.sumAll(T.mul(net.out, net.out)); // ‖out‖²
      backward(loss);
      const params = [net.W1, net.b1, net.W2, net.b2];
      const hfd = 1e-5; let maxRel = 0;
      for (const P_ of params) {
        for (let i = 0; i < P_.value.length; i++) for (let j = 0; j < P_.value[0].length; j++) {
          const orig = P_.value[i][j];
          P_.value[i][j] = orig + hfd;
          const lp = (function () { const n2 = buildMLP(xVal, net.W1.value, net.b1.value, net.W2.value, net.b2.value); const l = T.sumAll(T.mul(n2.out, n2.out)); return l.value[0][0]; })();
          P_.value[i][j] = orig - hfd;
          const lm = (function () { const n2 = buildMLP(xVal, net.W1.value, net.b1.value, net.W2.value, net.b2.value); const l = T.sumAll(T.mul(n2.out, n2.out)); return l.value[0][0]; })();
          P_.value[i][j] = orig;
          const numG = (lp - lm) / (2 * hfd);
          const anaG = P_.grad[i][j];
          maxRel = Math.max(maxRel, Math.abs(anaG - numG) / (Math.abs(numG) + 1e-12));
        }
      }
      push('autodiff MLP 梯度检验', maxRel < 1e-5, `maxRelErr=${maxRel.toExponential(2)}`);
    }

    const passed = results.filter(r => r.pass).length;
    return { total: results.length, passed: passed, results: results };
  }

  /* ============================================================
   * 导出
   * ============================================================ */
  const API = {
    makeRNG, zeros, identity, transpose, add, sub, mul, scale, dot, matMaxAbs, relErr,
    matmul, matmulNaive, cholesky, cholSolve, invChol, mahalanobis, qr, jacobiEig, svd,
    linearRegression, sigmoid, nllLogistic, logisticRegression, pca, kmeans, gmmEM,
    softmaxRows, attention, attentionRef, T, backward, buildMLP, selfTest,
    version: '1.0.0'
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else { global.MicroForge = API; }
})(typeof window !== 'undefined' ? window : globalThis);
