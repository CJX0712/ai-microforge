# microforge · 世界顶级 AI 微指令基石库

> 第十层微操作 · 零依赖 · 可验证 · 硬不变量
> 作者：晨星 ｜ 署名 commit：晨星

锻造世界顶级 AI 的底层基石——把现代机器学习的数值内核拆解为可独立交叉验证的**微指令级操作**，每项都带一条**硬金标准不变量**。不自造轮子，直接站在世界顶级数学与算法之上：Golub & Van Loan《Matrix Computations》、Numerical Recipes、LAPACK 算法族、Bishop《Pattern Recognition and Machine Learning》。

---

## 设计原则（零重写 · 集大成）

- **零重写**：任何数值内核，先采用世界已有的最优算法（Cholesky / Householder QR / Jacobi SVD / IRLS / EM），直接复用其数学结构，不自造平庸轮子。
- **集大成**：每一项都对照工业级参考实现（LAPACK：DPOTRF / DGEQRF / DSYEVJ / DGESVJ）实现，凡有更强现成方案立即替换。
- **可验证**：每项操作至少有 1 条硬不变量——解析解 vs 数值解、主实现 vs 参照实现、对数似然单调、残差正交、梯度检验。
- **闭环交付**：零依赖单文件，浏览器与 Node 双运行，可运行、可复现、可验证，完整开源。

---

## 微操作目录与硬不变量

| # | 微操作 | 算法 / 参照 | 硬不变量（金标准） |
|---|--------|------------|--------------------|
| 1 | `matmul` 分块矩阵乘 | GotoBLAS cache-blocking | 主实现 vs 朴素三重循环 `relErr<1e-10`；(AB)ᵀ=BᵀAᵀ |
| 2 | `cholesky` 乔列斯基分解 | DPOTRF A=LLᵀ | A ≈ L Lᵀ `relErr<1e-9` |
| 3 | `qr` QR 分解 | Householder DGEQRF | QᵀQ=I，A≈QR |
| 4 | `jacobiEig` 对称特征分解 | 经典循环 Jacobi DSYEVJ | A v=λ v 残差 `<1e-9`，特征向量正交 |
| 5 | `svd` 奇异值分解 | 单边 Jacobi DGESVJ | A≈UΣVᵀ，UᵀU=I，VᵀV=I |
| 6 | `linearRegression` 线性回归 | 正规方程 + Cholesky | 残差 ⟂ X（Xᵀ(y−Xw)=0） |
| 7 | `logisticRegression` 逻辑回归 | Newton / IRLS（PRML §4.3） | 梯度检验 `maxRelErr<1e-6` |
| 8 | `pca` 主成分分析 | 协方差特征分解（PRML §12.1） | 协方差 PSD；投影方差 = 特征值 |
| 9 | `kmeans` 聚类 | Lloyd + k-means++ | 簇内惯性（SSE）单调不增 |
| 10 | `gmmEM` 高斯混合 | EM（Bishop PRML §9.2） | 对数似然单调不减；权重和=1；协方差 PSD |
| 11 | `attention` 缩放点积注意力 | 稳定 softmax | 权重行和=1；主实现 vs 参照一致 |
| 12 | `T` 反向微分 | 反向模式 autodiff（MLP 前向/反向） | 解析梯度 vs 中心差分 `maxRelErr<1e-7` |

---

## 文件结构

```
ai-microforge/
├── microforge.js   # 零依赖单文件内核（UMD：浏览器全局 + Node module）
├── test.js         # Node 无头自检（12/12 硬不变量交叉验证）
├── index.html      # 单文件可视化 demo + 浏览器内自检面板（零外部依赖）
└── README.md       # 本文档
```

## 运行方式

**Node（无头自检）：**
```bash
node test.js
# 输出 12/12 通过 · 耗时 ~8ms
```

**浏览器（可视化 + 自检）：**
直接双击打开 `index.html`，点击「运行自检」即可在浏览器内运行与 Node 完全相同的 12 项交叉验证，并查看 PCA / 注意力 / 线性回归 / GMM 的可视化演示。

**作为模块引用：**
```js
const MF = require('./microforge.js');
const { U, sigma, V } = MF.svd(A);          // 奇异值分解
const { components, eigVal } = MF.pca(X, 2); // 主成分
const g = MF.gmmEM(X, 3);                   // 高斯混合 EM
const r = MF.selfTest();                    // 一次性跑全部硬不变量
```

---

## 数学参照（集大成，不自造轮子）

- Gene H. Golub & Charles F. Van Loan, *Matrix Computations* — Cholesky / QR / Jacobi 分解的权威算法来源
- William H. Press et al., *Numerical Recipes* — Householder 反射、数值稳定实现
- LAPACK — `DPOTRF` / `DGEQRF` / `DSYEVJ` / `DGESVJ`（单边 Jacobi SVD，数值最稳）
- Christopher M. Bishop, *Pattern Recognition and Machine Learning* — IRLS、PCA、EM 的贝叶斯视角

## 验证总纲

每个算法至少有 1 条可被独立实现交叉验证的「不变量」：解析解 vs 数值解、暴力枚举 vs 贪心、参照实现 vs 主实现、对数似然单调、SSE 单调不增。所有不变量在 `selfTest()` 中由 Node 与浏览器双重执行，确保**可复现、可验证**。

---

## 许可

MIT © 晨星
