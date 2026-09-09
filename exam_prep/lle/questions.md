# Day 3 — Canonical Correlation Analysis (CCA)

This is the next Tier 1 topic, and you have the actual slide deck for it. If you'd rather follow lecture order and take t-SNE first, say so and we'll switch.

One connection to hold onto: **CCA is the same skeleton as LLE.** Lagrangian → derivatives → generalised eigenvalue problem. And the same "degeneracy patches" you met in questions 1, 3, 4, 5. Keep that analogy in mind — it makes the derivation almost mechanical.

## 1. Theoretical foundations

### 1.1 The setting

Two "views" of the same $N$ objects: $X\in\mathbb R^{M\times N}$ and $Y\in\mathbb R^{N_y\times N}$ (columns are samples, as in the slides). Assumption: a latent variable $Z$ generates both views. The question from slide 4: *which representation of $X$ and $Y$ best reflects $Z$?*

CCA's answer: the one maximising the correlation between the views. Find directions $w_x\in\mathbb R^M$, $w_y\in\mathbb R^{N_y}$ such that the projections $w_x^\top X$ and $w_y^\top Y$ are maximally correlated.

The slide example: $X$ = (Displacement, Horsepower, Weight), $Y$ = (Acceleration, MPG), latent $Z$ = car type. After CCA the first canonical pair separates sports cars from commercial cars.

Hotelling, *Biometrika*, 1936.

### 1.2 The objective, and why constraints are needed

The natural target is the correlation coefficient:
$$
\rho(w_x,w_y)=\frac{w_x^\top C_{xy}w_y}{\sqrt{w_x^\top C_{xx}w_x}\;\sqrt{w_y^\top C_{yy}w_y}},
\qquad
C_{xy}=\tfrac1N XY^\top,\quad C_{xx}=\tfrac1N XX^\top
$$
(data centred: $\sum_i x_i=\sum_i y_i=0$).

$\rho$ is **scale-invariant**: $\rho(cw_x,w_y)=\rho(w_x,w_y)$. So the solution is determined only up to the length of the vectors — a degeneracy again, exactly as in LLE. The patch: fix the scale by constraints. Then the denominator equals 1 and the problem becomes
$$
\boxed{\;\operatorname*{argmax}_{w_x,w_y} w_x^\top X Y^\top w_y
\quad\text{s.t.}\quad
w_x^\top XX^\top w_x=1,\;\;
w_y^\top YY^\top w_y=1\;}
$$
which is precisely slides 5 and 9.

**Note the form of the constraints.** Not $\|w_x\|=1$ but $w_x^\top C_{xx}w_x=1$ — what is fixed is the *variance of the projection*, not the length of the vector. That is what makes CCA invariant to arbitrary invertible linear transformations of each view (not merely orthogonal ones). If you used $\|w\|=1$ you'd get PLS instead — a different method, maximising covariance rather than correlation.

### 1.3 The derivation (this is what you must be able to write from scratch)

Lagrangian (slide 9):
$$
\mathcal L=w_x^\top C_{xy}w_y-\tfrac12\alpha\big(w_x^\top C_{xx}w_x-1\big)-\tfrac12\beta\big(w_y^\top C_{yy}w_y-1\big)
$$
Partial derivatives:
$$
\frac{\partial\mathcal L}{\partial w_x}=C_{xy}w_y-\alpha C_{xx}w_x=0,
\qquad
\frac{\partial\mathcal L}{\partial w_y}=C_{yx}w_x-\beta C_{yy}w_y=0
$$

**The step everyone forgets:** left-multiply the first equation by $w_x^\top$ and the second by $w_y^\top$:
$$
w_x^\top C_{xy}w_y=\alpha\,\underbrace{w_x^\top C_{xx}w_x}_{=1}=\alpha,
\qquad
w_y^\top C_{yx}w_x=\beta\,\underbrace{w_y^\top C_{yy}w_y}_{=1}=\beta
$$
The left-hand sides are scalars that are transposes of each other, hence equal. Therefore
$$
\boxed{\alpha=\beta=\rho}
$$
And this isn't a technicality: **the Lagrange multiplier *is* the canonical correlation**, just as the eigenvalue in PCA *is* the variance. That's the detail marks are given for.

Substituting $\alpha=\beta$ and writing in block form (slide 11):
$$
\begin{bmatrix}0&C_{xy}\\ C_{yx}&0\end{bmatrix}
\begin{bmatrix}w_x\\w_y\end{bmatrix}
=\rho
\begin{bmatrix}C_{xx}&0\\0&C_{yy}\end{bmatrix}
\begin{bmatrix}w_x\\w_y\end{bmatrix}
$$
A generalised eigenvalue problem $A\phi=\rho B\phi$. The same template as PCA, kPCA, LLE, LDA.

### 1.4 Two equivalent solution forms

**Reduction to an ordinary eigenproblem.** From the second equation $w_y=\tfrac1\rho C_{yy}^{-1}C_{yx}w_x$; substitute into the first:
$$
C_{xx}^{-1}C_{xy}C_{yy}^{-1}C_{yx}\,w_x=\rho^2 w_x
$$
The eigenvalues are the **squared** canonical correlations. Downside: the matrix is non-symmetric, numerically worse.

**Via whitening + SVD (what's done in practice).** Substitute $u=C_{xx}^{1/2}w_x$, $v=C_{yy}^{1/2}w_y$:
$$
T=C_{xx}^{-1/2}C_{xy}C_{yy}^{-1/2},\qquad T=U\,\mathrm{diag}(\rho_1,\dots)\,V^\top
$$
The canonical correlations are the **singular values** of $T$, and $w_x^{(i)}=C_{xx}^{-1/2}u_i$, $w_y^{(i)}=C_{yy}^{-1/2}v_i$. Symmetric, stable, and gives all pairs at once. The number of pairs is $\le\min(M,N_y)$, and all $\rho_i\in[0,1]$.

### 1.5 Kernel CCA

Motivation from slide 15: covariance matrices are sometimes too large to compute (bag-of-words — potentially infinite-dimensional), and CCA doesn't capture non-linear dependencies.

Same idea as kPCA: the solution must lie in the span of the data, so $w_x=X\alpha_x$, $w_y=Y\alpha_y$. Substituting, with $K_x=X^\top X$, $K_y=Y^\top Y$ (slide 18):
$$
\begin{bmatrix}0&K_xK_y\\ K_yK_x&0\end{bmatrix}
\begin{bmatrix}\alpha_x\\\alpha_y\end{bmatrix}
=\rho
\begin{bmatrix}K_x^2&0\\0&K_y^2\end{bmatrix}
\begin{bmatrix}\alpha_x\\\alpha_y\end{bmatrix}
$$

**Critical: without regularisation kCCA is useless.** If $K_x$ is invertible (and for a Gaussian kernel with distinct points it essentially always is), you can find $\alpha_x,\alpha_y$ giving $\rho=1$ for *any* two views — even independent data. Perfect overfitting. Hence:
$$
K_x^2\;\to\;K_x^2+\kappa I,\qquad K_y^2\;\to\;K_y^2+\kappa I
$$
(visible on slide 48 in the canonical-trends formulas: $K_{\tilde Y}^2+I\kappa_y$).

That's the third degeneracy patch in this topic. Noticing the pattern?

### 1.6 Temporal kernel CCA

The problem (slide 21): if the variables are coupled **with a delay**, simultaneous samples are uncorrelated and standard (k)CCA finds nothing.

The fix: shift one view and maximise correlation over a sum across all lags:
$$
\operatorname*{argmax}_{w_x(\tau),\,w_y}\operatorname{Corr}\Big(\sum_\tau w_x(\tau)^\top x(t-\tau),\;w_y^\top y(t)\Big)
$$

The technical trick is **temporal embedding** (Takens 1981): stack shifted copies into one long vector
$$
\tilde X=\begin{bmatrix}X_{\tau_1}\\\vdots\\X_{\tau_T}\end{bmatrix},\qquad
\tilde w_x=\begin{bmatrix}w_x(\tau_1)\\\vdots\\w_x(\tau_T)\end{bmatrix}
$$
and the problem becomes an **ordinary CCA** on $\tilde X$. That's the whole idea: a hard problem reduced to a familiar one by changing the representation.

Terminology (slide 21):
- (k)CCA finds *canonical variates and correlation*
- tkCCA finds *canonical convolution and correlogram*

Applications from the slides: neuro-vascular coupling (simultaneous fMRI/BOLD + intracortical activity — $X$ and $Y$ of different dimensions, high-dimensional, non-instantaneous coupling), where tkCCA recovers the haemodynamic response function and the canonical correlogram. And canonical trend analysis: news-site content (bag-of-words) vs. geographic retweet locations with time lags.

### 1.7 Summary table

| | |
|---|---|
| **Hyperparameters** | number of components; kernel + its parameters (kCCA); regulariser $\kappa$; the lag set $\{\tau\}$ (tkCCA) |
| **Learned parameters** | $w_x,w_y$ (or duals $\alpha_x,\alpha_y$); canonical correlations $\rho_i$ |
| **Preprocessing** | centre both views (mandatory — the formulas assume $\sum x_i=0$); standardise features; for kCCA centre the kernel, $\tilde K=HKH$ |
| **Interpretation** | $\rho_i$ = strength of shared variability; $w_x,w_y$ = aligned subspaces; for text each canonical direction is a *topic* (De Bie & Cristianini 2004) |
| **Limitations** | linear unless kernelised; needs $N\gg M+N_y$, else $C_{xx}$ is singular; unregularised kCCA gives $\rho=1$ trivially; sensitive to the choice of $\kappa$; correlation $\ne$ causation |

### 1.8 Connections worth having ready

- **CCA $\to$ LDA:** if $Y$ is a one-hot encoding of class labels, CCA yields Fisher's linear discriminant.
- **CCA vs PLS:** PLS maximises covariance under $\|w\|=1$; CCA maximises correlation under $w^\top Cw=1$. PLS is more stable for small $N$; CCA is invariant to linear transformations.
- **CCA vs PCA:** PCA is one view, maximising variance; CCA is two views, maximising correlation. Both are generalised eigenvalue problems.
- **One-liner for MC:** *CCA finds projections of two views maximising their correlation subject to unit variance of each projection; the solution is a generalised eigenvalue problem in which the eigenvalue is the canonical correlation itself.*

---

## 2. Understanding questions

1. Why do the constraints take the form $w_x^\top C_{xx}w_x=1$ rather than $\|w_x\|=1$? Which degeneracy does this remove, and which invariance does it grant? (Compare with the role of $\sum_j w_j=1$ in LLE.)
2. Prove $\alpha=\beta$ in one line. Where exactly are the constraints used?
3. What does the Lagrange multiplier *mean* in CCA? Compare with the meaning of the eigenvalue in PCA.
4. Why does the reduced form $C_{xx}^{-1}C_{xy}C_{yy}^{-1}C_{yx}w_x=\rho^2w_x$ carry $\rho^2$ rather than $\rho$?
5. How many non-trivial canonical pairs exist, and why? What happens when $M=1$?
6. Show that the canonical correlations are unchanged under $X\to AX$, $Y\to BY$ for invertible $A,B$. Why is this desirable, and why does PCA lack such a property?
7. Why does unregularised kCCA give $\rho=1$ even for independent views? Argue via dimensions / invertibility of $K_x$.
8. Compare the role of $\kappa$ in kCCA with the role of $r$ in LLE. Both patch a degeneracy — but different degeneracies. Which ones?
9. In the neuro-vascular coupling problem: why would ordinary kCCA fail, and what exactly does the temporal embedding add?
10. Why does CCA require centred data? What breaks without it?
11. What is the difference between a *canonical variate* and a *canonical convolution*?
12. (MC drill) Which is the CCA objective? (a) minimise reconstruction error; (b) maximise correlation of projections of two views subject to unit variance of each; (c) maximise covariance subject to $\|w\|=1$; (d) maximise non-Gaussianity.
13. Data: $N=50$ samples, $M=200$, $N_y=300$. What happens to CCA, and what do you do about it?
14. Build the table: PCA, kPCA, LLE, CCA — for each, write $A$ and $B$ in $A\phi=\lambda B\phi$, and what $\lambda$ means. (This is the card you started on Day 1.)

---

## 3. Tasks

### A. Derivations (paper, closed-book, timed)

**A1 (12 min).** The full CCA derivation from scratch: constrained objective → Lagrangian → derivatives → proof that $\alpha=\beta$ → block generalised eigenproblem. Write it as if it were an exam answer.

**A2 (5 min).** Prove $\rho$'s invariance to $w_x\to cw_x$ and explain why that is exactly why constraints are needed. Same argument structure as LLE question 1 — state that structure explicitly.

**A3 (8 min).** Reduce the block problem to $C_{xx}^{-1}C_{xy}C_{yy}^{-1}C_{yx}w_x=\rho^2w_x$. Where precisely does the square appear?

**A4 (10 min).** Derive the whitening solution: substitute $u=C_{xx}^{1/2}w_x$ and show the problem becomes the SVD of $T=C_{xx}^{-1/2}C_{xy}C_{yy}^{-1/2}$.

**A5 (10 min).** Derive kCCA: substitute $w_x=X\alpha_x$, $w_y=Y\alpha_y$ into the block problem and obtain the form on slide 18.

**A6 (8 min).** Prove that canonical correlations are invariant to invertible linear transformations of the views (question 6).

**A7 (stretch, 15 min).** Put LLE and CCA side by side on one sheet: objective, degeneracy, patch, Lagrangian, final eigenvalue problem, meaning of $\lambda$. This is the most valuable artefact of the day — the generalised eigenvalue problem recurs in the exam every year.

**A8 (stretch).** Take $Y$ = one-hot labels for two classes and show CCA recovers the Fisher direction.

### B. Code (NumPy, no sklearn)

**B1.** Write `cca(X, Y, reg=1e-6)` via whitening + SVD, returning $\rho$, $W_x$, $W_y$. Comment the shapes before writing each line — the habit from last time.

```python
def cca(X, Y, reg=1e-6):
    # X: (N, M), Y: (N, Ny)  -- here rows are samples
    # centre; Cxx (M,M), Cyy (Ny,Ny), Cxy (M,Ny)
    # Cxx^{-1/2} via eigh
    # T = Cxx^{-1/2} Cxy Cyy^{-1/2};  U,S,Vt = svd(T)
    # rho = S; Wx = Cxx^{-1/2} @ U; Wy = Cyy^{-1/2} @ V
    ...
```

**B2 (validation).** Synthetic: latent $z\in\mathbb R^2$, $X=Az+\text{noise}$, $Y=Bz+\text{noise}$ with random $A,B$. Check the first two $\rho$ are large and the rest small. Check $\mathrm{corr}(Xw_x^{(1)}, Yw_y^{(1)})\approx\rho_1$ — that's the main correctness test.

**B3.** Reproduce the slide example on the Auto MPG dataset: $X$ = (displacement, horsepower, weight), $Y$ = (acceleration, mpg). Scatter $Xw_x$ against $Yw_y$ and compare your $w_x,w_y$ with the slide-12 numbers ($w_x\approx(0.0025,\,0.0202,\,-0.000025)$, $w_y\approx(-0.17,\,-0.092)$) — up to sign and scale.

**B4 (invariance).** Verify A6 numerically: multiply $X$ by a random invertible $A$, re-run, compare $\rho$. Should agree to $10^{-10}$.

**B5 (kCCA overfitting).** Generate **independent** $X$ and $Y$ ($N=50$, $M=N_y=100$). Run CCA without regularisation → you get $\rho_1\approx1$ on pure noise. Then with regularisation. Plot $\rho_1$ against $\kappa$ on a log scale. This is question 7 happening on screen.

**B6 (stretch).** kCCA with a Gaussian kernel on non-linearly related data (e.g. $y=\sin(x)$ + noise): show linear CCA gives low $\rho$ while kCCA gives high $\rho$.

**B7 (stretch).** tkCCA: generate $y(t)=\sum_\tau h(\tau)x(t-\tau)+\text{noise}$ with a known kernel $h$. Build the temporal embedding $\tilde X$, run ordinary CCA on it, and check that $w_x(\tau)$ recovers $h$. This reproduces the HRF logic of slide 30.

### C. Applied-template drill

**C1 (8 min, timed).** Fill the six parts for: *"A lab has simultaneous EEG recordings (64 channels) and behavioural measures (reaction time, accuracy, eye state) for 40 participants. They want to find which brain-activity patterns relate to behaviour."* Method + justification, hyperparameters, learned parameters, preprocessing, interpretation, limitations.

**C2 (5 min).** Same scenario, but the coupling has a 3–8 second delay. What changes in your answer?

### D. MC drill

**D1.** Add CCA to the Day 1 card: one-line objectives for LLE, PCA, kPCA, CCA in identical notation.

**D2.** Write one line on the role of each: $C_{xx}$, $C_{xy}$, $\rho$, $\alpha$, $\kappa$, $\tau$.

---

Order of attack: questions 1–5 verbally first (fast), then **A1** — the spine of the day — then **B1+B2**. Do **B5** without fail; it's the cheapest way to understand kCCA regularisation.

Post your answers to the questions or your A1 derivation here and I'll mark it the way the exam would, pointing out exactly where marks would be lost.