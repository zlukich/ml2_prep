Flagging before we move: `forward` and `sample_hmm` are the two functions that literally appeared on the 2026 exam paper. Everything else in the plan is inference; those two are attested. Worth ten minutes at some point.

Next by plan is **Days 4–5: Kernel Machines 1 (Structured Input)** — Tier 1, and the topic that carries exam question 4.

---

# Day 5 — Kernel Machines 1: Structured Input

## 1. Theoretical foundations

### 1.1 The problem kernels solve

You have an SVM, PCA, or CCA — all of which touch the data only through inner products $\langle x, x'\rangle$. Two limitations follow: the methods are linear, and they need $x\in\mathbb R^D$. But real data is often *structured*: DNA strings, molecule graphs, parse trees, sets. There is no natural vector representation.

The kernel trick: replace $\langle x,x'\rangle$ with $k(x,x')=\langle\phi(x),\phi(x')\rangle$ for some feature map $\phi:\mathcal X\to\mathcal H$. Two payoffs:

- **Non-linearity.** $\phi$ can be non-linear, so a linear method in $\mathcal H$ is non-linear in $\mathcal X$.
- **Structure.** $\mathcal X$ needn't be $\mathbb R^D$ at all. It can be the set of all strings, all graphs, all sets. You only ever need to *compare* two objects, never represent one.

And you never compute $\phi(x)$ — only $k(x,x')$. $\mathcal H$ can be infinite-dimensional (Gaussian kernel) and the computation stays finite.

### 1.2 Definition and the PSD condition

$k:\mathcal X\times\mathcal X\to\mathbb R$ is a **positive semidefinite (PSD) kernel** if it is symmetric and for every finite $\{x_1,\dots,x_n\}\subset\mathcal X$ and every $c\in\mathbb R^n$:
$$
\sum_{i,j}c_ic_j\,k(x_i,x_j)\;\ge\;0
$$
Equivalently: the **Gram matrix** $K$ with $K_{ij}=k(x_i,x_j)$ is PSD for any finite sample.

**Why PSD matters.** Mercer's theorem: $k$ is PSD $\iff$ there exists a Hilbert space $\mathcal H$ and a map $\phi$ with $k(x,x')=\langle\phi(x),\phi(x')\rangle$. So PSD is exactly the condition for "this is a legitimate inner product in some space." Without it, $\|w\|^2$ in the SVM dual can go negative, the optimisation stops being convex, and the whole framework breaks.

**The easy direction (memorise it).** If $k(x,x')=\langle\phi(x),\phi(x')\rangle$ then
$$
\sum_{i,j}c_ic_jk(x_i,x_j)=\sum_{i,j}c_ic_j\langle\phi(x_i),\phi(x_j)\rangle=\Big\|\sum_i c_i\phi(x_i)\Big\|^2\ge0
$$
That's a complete proof in three lines. Any time you can exhibit a feature map, PSD follows for free — which is why exam question 4 asked for *both* the proof and the feature map.

### 1.3 Closure properties — the toolbox

These are what exam question 4 tests. Given PSD kernels $k_1,k_2$:

| construction | PSD? | feature map |
|---|---|---|
| $\alpha k_1$, $\alpha\ge0$ | ✓ | $\sqrt\alpha\,\phi_1$ |
| $k_1+k_2$ | ✓ | concatenation $(\phi_1,\phi_2)$ |
| $k_1\cdot k_2$ | ✓ | tensor product $\phi_1\otimes\phi_2$ |
| $f(x)f(x')$, any $f:\mathcal X\to\mathbb R$ | ✓ | $\phi(x)=f(x)$ (1-D) |
| $k_1(\psi(x),\psi(x'))$ for any map $\psi$ | ✓ | $\phi_1\circ\psi$ |
| $p(k_1)$, polynomial with coefficients $\ge0$ | ✓ | from sum + product |
| $\exp(k_1)$ | ✓ | limit of the above |
| $k_1\otimes k_2$ on $\mathcal X_1\times\mathcal X_2$ | ✓ | $\phi_1\otimes\phi_2$ |
| $\dfrac{k_1(x,x')}{\sqrt{k_1(x,x)k_1(x',x')}}$ | ✓ | $\phi_1(x)/\|\phi_1(x)\|$ |
| $k_1-k_2$ | ✗ | — |

**The product case deserves care** — it's the one on the exam. Two proofs:

*Schur product theorem.* The Hadamard (elementwise) product of two PSD matrices is PSD. Since $(K_1\odot K_2)_{ij}=k_1(x_i,x_j)k_2(x_i,x_j)$, the product kernel's Gram matrix is a Hadamard product of PSD matrices, hence PSD.

*Feature-map construction.* With $\phi_1(x)\in\mathbb R^{m}$, $\phi_2(x)\in\mathbb R^{n}$, define $\Phi(x)=\phi_1(x)\otimes\phi_2(x)\in\mathbb R^{mn}$, i.e. $\Phi_{ab}(x)=\phi_{1,a}(x)\phi_{2,b}(x)$. Then
$$
\langle\Phi(x),\Phi(x')\rangle=\sum_{a,b}\phi_{1,a}(x)\phi_{2,b}(x)\phi_{1,a}(x')\phi_{2,b}(x')
=\Big(\sum_a\phi_{1,a}(x)\phi_{1,a}(x')\Big)\Big(\sum_b\cdots\Big)=k_1k_2
$$
The second is more useful in exams because it gives the feature map as a by-product.

### 1.4 Standard kernels on $\mathbb R^D$

**Linear:** $k(x,x')=x^\top x'$, $\phi=\mathrm{id}$.

**Polynomial:** $k(x,x')=(x^\top x'+c)^p$. Feature map = all monomials up to degree $p$, dimension $\binom{D+p}{p}$. Derive the $p=2$, $D=2$, $c=0$ case by hand at least once:
$$
(x_1x_1'+x_2x_2')^2 = (x_1^2)(x_1'^2)+2(x_1x_2)(x_1'x_2')+(x_2^2)(x_2'^2)
\;\Rightarrow\;\phi(x)=(x_1^2,\sqrt2x_1x_2,x_2^2)
$$

**Gaussian/RBF:** $k(x,x')=\exp(-\|x-x'\|^2/2\sigma^2)$. Infinite-dimensional $\mathcal H$. PSD proof: expand
$$
\exp\!\Big(-\tfrac{\|x\|^2}{2\sigma^2}\Big)\exp\!\Big(\tfrac{x^\top x'}{\sigma^2}\Big)\exp\!\Big(-\tfrac{\|x'\|^2}{2\sigma^2}\Big)
$$
The middle factor is $\exp$ of a PSD kernel (✓ by closure), the outer two form $f(x)f(x')$ (✓). Product of PSD kernels (✓). Done — this is a nice exam-length proof.

### 1.5 Kernels on structured inputs

**Spectrum kernel** (strings). $\phi_k(s)$ counts occurrences of every $k$-mer:
$$
k_{\text{spec}}(s,s')=\sum_{u\in\Sigma^k}\#_u(s)\,\#_u(s')
$$
PSD trivially — it's an explicit inner product of count vectors. Dimension $|\Sigma|^k$ (for DNA, $4^k$), but computable in $O(|s|+|s'|)$ with a hash map, so you never build the vector.

**Mismatch kernel.** Same, but a $k$-mer counts toward $u$ if it differs in at most $m$ positions. Tolerates mutations. Still an explicit (if larger) feature map, so PSD.

**Weighted-degree kernel** (Rätsch & Sonnenburg — the splice-site work referenced in your Kernels 2 lecture):
$$
k_{\text{WD}}(s,s')=\sum_{d=1}^{D}\beta_d\sum_{i=1}^{L-d+1}\mathbb 1\big[s_{i:i+d}=s'_{i:i+d}\big]
$$
The key difference from the spectrum kernel: matches must occur at the **same position** $i$. That's the right assumption for aligned biological sequences, where position carries meaning. PSD because each indicator is a matching kernel and the outer sums have $\beta_d\ge0$.

**Bag-of-words** (the kCCA slide). $\phi(\text{doc})$ = word counts. Same structure as the spectrum kernel with words instead of $k$-mers.

**Graph kernels.** Random-walk kernel (count matching walks in the product graph), shortest-path kernel, Weisfeiler–Lehman subtree kernel. All built as inner products of substructure-count vectors — same recipe: *decompose the object into countable parts, and the kernel is an inner product of count vectors.* That's the general design principle for structured data (R-convolution kernels, Haussler 1999).

**Fisher kernel.** Given a generative model $p(x\mid\theta)$ — for instance an HMM — define $\phi(x)=\nabla_\theta\log p(x\mid\theta)$ (the Fisher score) and $k(x,x')=\phi(x)^\top F^{-1}\phi(x')$. Turns any generative model into a kernel. PSD since $F^{-1}$ is PSD. Worth knowing: this is the bridge between your HMM lecture and this one.

### 1.6 The indicator/matching kernel — 2026 Q4

$k_\delta(y,y')=\mathbb 1[y=y']$ on any set $\mathcal Y$. Feature map: $\phi_\delta(y)=e_y$, the one-hot basis vector. Then $\langle e_y,e_{y'}\rangle=\mathbb 1[y=y']$. ✓

The exam's $k_2((x,y),(x',y'))=k_1(x,x')\mathbb 1[y=y']$ is precisely a **product of two PSD kernels**, giving feature map $\Phi(x,y)=\phi_1(x)\otimes e_y$ — a block vector with $\phi_1(x)$ in block $y$ and zeros elsewhere. Full solution is in your plan document §7.

Interpretation worth stating: this kernel makes examples with different labels **orthogonal**. It's the standard construction for multi-class / structured-output problems, which is exactly why it bridges into your Kernels 2 lecture.

### 1.7 Summary table

| | |
|---|---|
| **Hyperparameters** | kernel choice; $\sigma$ (RBF), degree $p$ and offset $c$ (polynomial), $k$-mer length and $m$ (string kernels), $\beta_d$ (WD); SVM $C$ |
| **Learned parameters** | dual coefficients $\alpha_i$, bias $b$, support vectors |
| **Preprocessing** | scale features (RBF is scale-sensitive); normalise the kernel; fixed alphabet for strings; centre the Gram matrix for kPCA |
| **Interpretation** | support vectors = boundary-defining examples; WD weights reveal which positions matter |
| **Limitations** | $O(n^2)$ memory / $O(n^2$–$n^3)$ time in sample size; kernel choice is a modelling decision with no learning; poor interpretability without extra machinery (LRP); needs careful hyperparameter tuning |

### 1.8 Connections

- **Representer theorem:** the minimiser of a regularised risk lies in $\mathrm{span}\{\phi(x_i)\}$, so $w=\sum_i\alpha_i\phi(x_i)$. This is what makes kPCA, kCCA, SVM all expressible in $\alpha$ — the same substitution you made in A5 on Day 3.
- Kernels are the shallow half of the shallow↔deep table: kernel SVM ↔ deep net, kPCA ↔ autoencoder, one-class SVM ↔ deep SVDD.
- **MC one-liner:** *A kernel is an inner product in an implicit feature space; PSD is necessary and sufficient for such a space to exist, and closure properties let you build kernels for structured objects without ever computing the feature map.*

---

## 2. Understanding questions

1. State the PSD condition. Why is it necessary for the SVM's convexity?
2. Prove in three lines that any $k(x,x')=\langle\phi(x),\phi(x')\rangle$ is PSD.
3. Give the feature map for $k_1+k_2$ and for $k_1\cdot k_2$. Why is the second one *not* concatenation?
4. Show that $k_1-k_2$ need not be PSD, with a $2\times2$ counterexample.
5. Prove the Gaussian kernel is PSD using only closure properties.
6. What is the dimension of the polynomial kernel's feature space for degree $p$ in $\mathbb R^D$? Why does the kernel trick matter here?
7. What does the normalised kernel $k/\sqrt{k(x,x)k(x',x')}$ do geometrically, and why is it PSD?
8. Difference between the spectrum kernel and the weighted-degree kernel. When is each appropriate?
9. Why is the spectrum kernel PSD without any theorem?
10. What does $k_\delta(y,y')=\mathbb 1[y=y']$ do geometrically to examples with different labels?
11. State the representer theorem and explain why it makes kPCA/kCCA possible.
12. (MC drill) Which is *not* guaranteed PSD? (a) $k_1k_2$; (b) $k_1+k_2$; (c) $k_1-k_2$; (d) $\exp(k_1)$.
13. Given $\phi_1(x)\in\mathbb R^3$ and $\phi_2(x)\in\mathbb R^4$, what is the dimension of the feature map of $k_1+k_2$? Of $k_1k_2$?
14. Why does a Gram matrix need to be PSD *for every finite sample*, not just one?

---

## 3. Tasks

### A. Derivations (paper, closed-book, timed)

**A1 (10 min).** The full 2026 Q4: prove $k_2((x,y),(x',y'))=k_1(x,x')\mathbb 1[y=y']$ is PSD **two ways** (Gram-matrix grouping by label, and Schur product), then give the explicit feature map. This is the single highest-value derivation of the day.

**A2 (8 min).** Prove the closure list: $\alpha k_1$, $k_1+k_2$, $k_1k_2$, $f(x)f(x')$, $k_1(\psi(x),\psi(x'))$ — each with its feature map.

**A3 (8 min).** Prove the Gaussian kernel is PSD via the decomposition in §1.4.

**A4 (5 min).** Expand $(x^\top x')^2$ in $\mathbb R^2$ and read off the explicit feature map.

**A5 (5 min).** Prove the normalised kernel is PSD and show $|k_{\text{norm}}|\le1$.

**A6 (8 min).** Prove the weighted-degree kernel is PSD by exhibiting its feature map.

**A7 (stretch).** Prove the Schur product theorem for PSD matrices (hint: write $K_1=\sum_i\lambda_iv_iv_i^\top$ and use that $vv^\top\odot K_2=\mathrm{diag}(v)K_2\mathrm{diag}(v)$).

### B. Code

**B1.** `gram(X, kernel)` returning $K$; implement linear, polynomial, RBF. Verify PSD numerically: `np.linalg.eigvalsh(K).min() > -1e-10`.

**B2.** Verify closure numerically: build $K_1$ (RBF), $K_2$ (polynomial), check the minimum eigenvalue of $K_1+K_2$, $K_1\odot K_2$, and $K_1-K_2$. The last should go negative — question 4, live.

**B3.** Implement the spectrum kernel for DNA strings with a `dict` of $k$-mer counts. Test on short sequences; verify the Gram matrix is PSD.

**B4.** Implement $k_2$ from the exam: given $k_1$ = RBF and labels $y$, build the joint Gram matrix and confirm PSD. Verify the block structure: entries between different labels are exactly zero.

**B5.** Kernel centring $\tilde K=HKH$ with $H=I-\frac1n\mathbf 1\mathbf 1^\top$. Verify the centred features have zero mean, i.e. $\tilde K\mathbf 1\approx0$.

**B6 (stretch).** Weighted-degree kernel on aligned sequences; compare against the spectrum kernel on a task where position matters (e.g. a motif always at position 10).

### C. Applied-template drill

**C1 (8 min).** *"Classify 5,000 protein sequences of varying length into functional families."* Six parts, with explicit justification of the kernel choice.

### D. MC drill

**D1.** Write the closure table from memory — construction, PSD yes/no, feature map.

**D2.** Add to your topic card: kernel methods are the *representer-theorem* branch, distinct from the eigenvalue branch (LLE/CCA/PCA) and the EM branch (HMM). Although note kPCA/kCCA sit in both.

---

Order: questions 1–5, then **A1** (it's the attested exam question), then A2, then **B2 and B4**.

Post A1 and I'll mark it — including whether you'd get full marks for the feature map, which is where most answers are thin.