# ML2 (TU Berlin, Müller) — A 3-Week Exam Preparation Plan

## TL;DR
- The ML2 written exam is a **120-minute graded written exam** (Moses module 40551 v5) with a **recurring five-question skeleton**: one multiple-choice question spanning several topics, one "applied ML" scenario with a fixed six-part sub-structure, one derivation/proof (LLE + Lagrange in 2026), one kernel-theory proof (PSD + feature map in 2026), and one Python/NumPy programming task (HMM in 2026) written on paper — so prepare to *reproduce derivations from scratch and write NumPy by hand*.
- Prioritise the **eigenvalue/Lagrange spine** (LLE, CCA/kCCA, PCA/kPCA, HMM) and the **kernel spine** (PSD proofs, feature maps, one-class SVM↔deep SVDD) — these carry the derivation, kernel-proof, and programming questions and recur across years; t-SNE, ICA, structured-output kernels, and the three Deep Learning lectures are the most likely rotation for the MC and applied questions.
- Follow the day-by-day plan below (≈4–6 h/day, 21 days with a 14-day compression), ending in two full mock exams; master the six-part applied-ML template and the attested HMM code, and you will cover the highest-probability surface of the exam.

## Key Findings

**Course and exam facts (confirmed).** Machine Learning 2 (module 40551, 9 CP; ML2-X 41173, 12 CP) is taught in English by Prof. Klaus-Robert Müller with Dr. Jacob Kauffmann (SoSe 2026, ISIS course 48434). The Moses module description (module 40551, version 5) states the assessment is a *Schriftliche Prüfung* (written exam), *Benotet* (graded), *Dauer/Umfang 120 Minuten*. Exam admission historically requires ≥50% of exercise points plus (for some versions) a seminar presentation. The course workload is split into concepts/theory, exercises, and programming (Python/NumPy in Jupyter notebooks), confirmed by the two public coursework repos (moritz-gerster/machine_learning2, WGierke/machine_learning_2), each of which contains an analytical PDF homework plus a programming `.ipynb` per week ("Assignments include both analytical derivations and programming tasks"; language breakdown ≈98–99% Jupyter Notebook).

**The 2026 topic list** (in lecture order): (1) LLE, (2) t-SNE, (3) CCA, (4) HMMs, (5) Kernel Machines 1 (structured input), (6) Kernel Machines 2 (structured output/bioinformatics), (7) Component Analysis 2 (blind source separation/ICA), (8) Kernel Machines 3 (anomaly detection), (9) Component Analysis 3 (representation learning), (10) Deep Learning 1 (structured input), (11) Deep Learning 2 (structured output), (12) Deep Learning 3 (anomaly detection). The official 2026 ML Group page also lists Bioinformatics and Explainable AI as themes, consistent with the Müller group's emphasis on LRP/explainable anomaly detection.

**Exam structure evidence.** The 2026 memory protocol (Q1 MC; Q2 applied HMM; Q3 LLE+Lagrange; Q4 kernel PSD+feature map; Q5 HMM programming) matches the SoSe 2021 protocol structure almost exactly (1. Multiple Choice; 2. Practical ML; 3. String Kernels; 4. Conditional RBM; 5. Programming — an HMM forward-algorithm + simulation task in NumPy, taught that year with Grégoire Montavon). This confirms a **stable five-part template** where the *slots* are fixed but the *topics filling them rotate*. The programming slot in both attested years was an HMM.

**Aids/open-book status: UNCONFIRMED.** No TU Berlin ML2-specific statement of permitted aids (Hilfsmittel) could be found in public sources; cheat-sheet/calculator rules found online belong to other universities (Tübingen, Chemnitz, KIT-Zöllner) and must not be assumed here. Plan for a **closed-book, no-aids** exam (the safe assumption) and confirm on ISIS 48434 / the exam cover sheet. Note the SoSe 2021 ML2 exam was conducted online, so aid rules can differ by year.

## Details

### 1. Diagnostic reading of the 2026 protocol

The five questions map to five distinct skills, and the weighting is implicit in the format:

- **Q1 — Multiple Choice (cross-topic recognition).** In 2026 it touched LLE's objective, the CCA objective, and the roles of forward/Viterbi/Baum–Welch. MC rewards *crisp one-line characterisations* of every method's objective and of each algorithm's role. This is the cheapest question to secure points on if you have memorised the "one-sentence identity" of each of the 12 topics. Because MC samples broadly, **no topic is safe to skip for recognition-level knowledge.**
- **Q2 — Applied ML (six-part scenario).** The fixed sub-structure — (1) choose method, (2) hyperparameters, (3) learned parameters, (4) preprocessing, (5) interpretation of model/latent states, (6) limitations — is a *template you can rehearse*. In 2026 the scenario (latent-state customer-interaction sequences) mapped to an HMM. In other years the same skeleton could map to t-SNE (visualisation), one-class SVM/deep SVDD (anomaly detection), ICA (source separation), CCA (two-view data), a string/graph kernel SVM (sequence/graph data), or an autoencoder (representation learning). Prepare a filled skeleton for each.
- **Q3 — Derivation/proof.** 2026: state what LLE optimises, then a Lagrange-multiplier derivation of the constrained-weight solution and its objective value. This is the "eigenvalue/Lagrange spine" question. It rotates among CCA, PCA/kPCA, Fisher LDA, one-class SVM/SVDD duals — all Lagrangian + (generalised) eigenvalue problems.
- **Q4 — Kernel theory.** 2026: prove a product/indicator kernel is PSD and construct its feature map. This tests kernel closure properties and explicit feature-map construction. It rotates among string/spectrum/WD kernels, Fisher kernels, graph kernels, and joint input–output kernels.
- **Q5 — Programming (NumPy on paper).** 2026 and 2021 were both HMMs (validation, forward algorithm, simulation). The programming slot is *the most predictable*: HMM inference is the modal task, but LLE weights, PCA/kPCA, k-means, and CCA are plausible.

**Hedge for a different year.** Since the *slots* are fixed and the *topics* rotate, the robust strategy is: (i) memorise one-line identities for all 12 topics (covers MC); (ii) rehearse the six-part applied template on ≥6 candidate methods; (iii) be able to reproduce ~8 core derivations (below); (iv) be able to prove PSD + build feature maps for the standard kernel constructions; (v) be able to hand-write HMM/LLE/PCA/k-means in NumPy.

### 2. Topic prioritisation into tiers

**Tier 1 — High probability & high effort (master fully; ~55% of effort).**
- **LLE** (Q3 2026; recurs in MC and programming) — Roweis, S.T. & Saul, L.K., "Nonlinear Dimensionality Reduction by Locally Linear Embedding," *Science* 290(5500):2323–2326, 22 Dec 2000 (DOI: 10.1126/science.290.5500.2323); plus Saul & Roweis 2003 (JMLR) for the algorithm appendix.
- **HMM** (Q2 + Q5 2026; Q5 2021; forward/Viterbi/Baum–Welch in MC) — Rabiner, L.R., "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proceedings of the IEEE* 77(2):257–286, 1989 (DOI: 10.1109/5.18626).
- **CCA / kCCA / tkCCA** (MC 2026; canonical eigenvalue derivation; provided slide deck) — Hotelling 1936.
- **Kernel machines 1 — structured input** (Q4 2026; PSD proofs, feature maps, string kernels) — Shawe-Taylor & Cristianini, *Kernel Methods for Pattern Analysis*; Schölkopf & Smola, *Learning with Kernels*.
- **PCA/kPCA** (the backbone connecting LLE, CCA, autoencoders, anomaly detection) — Schölkopf, B., Smola, A. & Müller, K.-R., "Nonlinear Component Analysis as a Kernel Eigenvalue Problem," *Neural Computation* 10(5):1299–1319, July 1998 (DOI: 10.1162/089976698300017467).

**Tier 2 — Medium probability (be exam-ready; ~30%).**
- **t-SNE** (natural MC/applied fit for visualisation) — van der Maaten, L. & Hinton, G., "Visualizing Data using t-SNE," *JMLR* 9:2579–2605, 2008.
- **ICA / blind source separation** (applied source-separation scenario; MC on ambiguities) — Hyvärinen & Oja 2000.
- **Kernel anomaly detection — one-class SVM / SVDD** (applied + derivation) — Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. & Williamson, R.C., "Estimating the Support of a High-Dimensional Distribution," *Neural Computation* 13(7):1443–1471, July 2001; Tax, D.M.J. & Duin, R.P.W., "Support Vector Data Description," *Machine Learning* 54(1):45–66, 2004.
- **Kernel machines 2 — structured output / bioinformatics** (Q4-style kernel proof; splice-site application) — Tsochantaridis et al. 2005 (JMLR); Sonnenburg/Rätsch weighted-degree kernels.

**Tier 3 — Lower probability (recognition + one applied skeleton; ~15%).**
- **Representation learning** (autoencoders, dictionary learning, NMF, self-supervised) — Goodfellow, Bengio & Courville, *Deep Learning*.
- **Deep learning 1/2/3** (CNN/GNN invariances; seq2seq/energy-based; deep SVDD + LRP) — Ruff et al., "Deep One-Class Classification," ICML, PMLR 80:4393–4402, 2018; Ruff, L., Kauffmann, J.R., Vandermeulen, R.A., Montavon, G., Samek, W., Kloft, M., Dietterich, T.G. & Müller, K.-R., "A Unifying Review of Deep and Shallow Anomaly Detection," *Proceedings of the IEEE* 109(5):756–795, May 2021 (DOI: 10.1109/JPROC.2021.3052449) — note co-author Jacob R. Kauffmann is the ML2 course instructor, so this review is a high-value primary source; Montavon, Binder, Lapuschkin, Samek & Müller, "Layer-Wise Relevance Propagation: An Overview," Springer LNCS 11700, 2019.

### 3. Day-by-day plan (21 days, ≈4–6 h/day)

Each day: **[Derive]** = reproduce on paper from scratch; **[State]** = explain verbally/one line; **[Recall]** = active-recall self-test; **[Code]** = NumPy from memory. Compressed 14-day variant marked **‡** (do only ‡ days, merging review into adjacent days).

**Week 1 — Eigenvalue/Lagrange spine + kernels core**

- **Day 1 ‡ — PCA/kPCA foundations.** [Derive] PCA as variance maximisation → eigen-decomposition of covariance; equivalence to reconstruction-error minimisation $\min_U \|X-XUU^\top\|_F^2$ s.t. $U^\top U=I$; dual/kernel PCA via centred Gram matrix $\tilde K v=\lambda v$. [State] why kPCA needs double-centering. [Code] `pca(X,d)` via `np.linalg.eigh`. [Recall] write the generalised-eigenvalue template $A\phi=\lambda B\phi$ and list every ML2 method that instantiates it.
- **Day 2 ‡ — LLE (Q3 anchor).** [Derive] the full LLE pipeline: kNN graph; reconstruction weights minimising $\sum_i\|x_i-\sum_j W_{ij}x_j\|^2$ s.t. $\sum_j W_{ij}=1$; the local Gram matrix $C_{jk}=(x_i-\eta_j)^\top(x_i-\eta_k)$; the closed-form $w^*=C^{-1}\mathbf 1/(\mathbf 1^\top C^{-1}\mathbf 1)$; regularisation of $C$; embedding as bottom non-zero eigenvectors of $M=(I-W)^\top(I-W)$ (discard the all-ones eigenvector). [Code] `lle_weights(x_i, neighbors)`. [Recall] reproduce the full worked Q3 (Section 7).
- **Day 3 ‡ — CCA / kCCA / tkCCA.** [Derive] CCA maximise $w_x^\top X Y^\top w_y$ s.t. $w_x^\top XX^\top w_x=1$, $w_y^\top YY^\top w_y=1$; Lagrangian → $\alpha=\beta$ → block generalised eigenvalue problem; whitening/SVD solution. [Derive] kCCA with $K_x,K_y$ and its regularisation; recovery $w_x=X\alpha_x$. [State] tkCCA time-lags, canonical convolution, correlogram; neuro-vascular/fMRI-BOLD and Twitter-trend applications (from the provided slide deck). [Code] `cca(X,Y)` via generalised eig.
- **Day 4 ‡ — Kernels 1: PSD theory + feature maps (Q4 anchor).** [Derive] kernel = inner product of feature maps; Gram matrix PSD proof; closure under sum, positive scaling, product (Schur/Hadamard), tensor/direct product, composition with power series of non-negative coefficients, pointwise limit; the indicator kernel $\mathbb 1[y=y']$. [Derive] the full Q4 (Section 7). [State] Mercer's condition. [Recall] build feature maps for sum, scaling, product.
- **Day 5 ‡ — Kernels 1 continued: string/sequence kernels.** [State] spectrum kernel, mismatch kernel, weighted-degree (WD) kernel (position-dependent k-mer co-occurrence, Rätsch & Sonnenburg), Fisher kernel; graph kernels. [Derive] why a normalised kernel $k/\sqrt{k(x,x)k(x',x')}$ is PSD. [Recall] MC one-liners for each kernel.
- **Day 6 — HMM part 1: model + evaluation/decoding.** [Derive] HMM as $\lambda=(A,B,\pi)$; the three problems (evaluation, decoding, learning); forward recursion $\alpha_t(j)=\big[\sum_i\alpha_{t-1}(i)A_{ij}\big]B_{j,x_t}$, init $\alpha_1(i)=\pi_i B_{i,x_1}$; backward recursion; Viterbi (max instead of sum + backpointers). [State] scaling to avoid underflow. [Code] `forward(A,B,pi,obs)` with and without scaling; `viterbi`.
- **Day 7 — Review + spaced repetition (Week 1).** [Recall] blank-paper reproduce: PCA eigenproblem, LLE weight solution + objective value, CCA generalised eig, kernel PSD closure list, forward/Viterbi recursions. Redo any derivation that took >2 tries. [Code] re-implement `forward` and `pca` from memory, timed.

**Week 2 — Probabilistic learning, component analysis, anomaly detection**

- **Day 8 ‡ — HMM part 2: learning + applied template.** [Derive] Baum–Welch/EM: $\gamma_t(i)$, $\xi_t(i,j)$, re-estimation of $A,B,\pi$; scaled forward-backward. [Apply] the six-part template to the 2026 customer-interaction scenario (Section 5). [Code] `sample_hmm(A,B,pi,T)`; `validate(A,B,pi)`.
- **Day 9 ‡ — t-SNE.** [Derive] SNE conditional $p_{j|i}$; symmetric SNE joint $p_{ij}=(p_{j|i}+p_{i|j})/2n$; Student-t low-dim $q_{ij}\propto(1+\|y_i-y_j\|^2)^{-1}$; KL gradient $\frac{\partial C}{\partial y_i}=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+\|y_i-y_j\|^2)^{-1}$. [State] crowding problem, heavy-tail fix, perplexity as a smooth measure of the effective number of neighbours (per van der Maaten & Hinton 2008, "typical values are between 5 and 50" and SNE is "fairly robust to changes in the perplexity"), Barnes–Hut $O(n\log n)$. [Apply] six-part template for visualisation.
- **Day 10 ‡ — ICA / blind source separation.** [Derive] linear mixing $x=As$; whitening then rotation; non-Gaussianity via kurtosis and negentropy $J(y)=H(y_{gauss})-H(y)$; FastICA fixed-point; InfoMax; JADE (4th-order cumulants); TDSEP/SOBI (time-lagged decorrelation). [State] ambiguities (scale/sign, permutation) and identifiability (≤1 Gaussian source). [Apply] six-part template for source separation.
- **Day 11 ‡ — Kernel anomaly detection.** [Derive] one-class SVM (Schölkopf et al. 2001): $\min \tfrac12\|w\|^2+\tfrac1{\nu n}\sum_i\xi_i-\rho$ s.t. $w^\top\phi(x_i)\ge\rho-\xi_i$; the $\nu$-property. [Derive] SVDD (Tax & Duin 2004): min enclosing hypersphere $\min R^2+C\sum\xi_i$; equivalence to OC-SVM for Gaussian kernel. [State] KDE vs OC-SVM; kPCA reconstruction-error scoring. [Apply] six-part template for anomaly detection. [Code] OC-SVM scoring given dual $\alpha$.
- **Day 12 — Kernels 2: structured output + bioinformatics.** [Derive] joint feature map $\Psi(x,y)$; structural SVM $\max_y \Delta(y,\hat y)+\langle w,\Psi(x,\hat y)-\Psi(x,y)\rangle$ (margin/slack rescaling); separation oracle; kernel dependency estimation (output kernel + pre-image). [State] splice-site prediction, protein classification. [Recall] why $k((x,y),(x',y'))=k_X(x,x')k_Y(y,y')$ is PSD (tensor product).
- **Day 13 — Representation learning (Component Analysis 3).** [Derive] linear autoencoder ≡ PCA (no nonlinearity, tied weights); relation kPCA↔autoencoder. [State] dictionary learning/sparse coding, NMF (non-negativity), self-supervised objectives. [Code] one-hidden-layer linear autoencoder training loop (NumPy, gradient step).
- **Day 14 — Review + spaced repetition (Week 2) + Mock Exam #1 (light).** [Recall] reproduce Baum–Welch re-estimation, t-SNE gradient, FastICA steps, OC-SVM primal, structural-SVM constraint. Take a 120-min self-made mock using 2026 questions.

**Week 3 — Deep learning, integration, mock exams**

- **Day 15 ‡ — Deep learning 1 (structured input).** [State] CNN weight sharing ⇒ translation equivariance; local connectivity/parameter efficiency; pooling ⇒ invariance; GNN message passing $h_v^{(k)}=\text{UPDATE}(h_v,\text{AGG}\{h_u\})$; permutation invariance/equivariance; sequence models. [Derive] parameter-count argument for weight sharing. [Recall] shallow↔deep parallels table.
- **Day 16 ‡ — Deep learning 2 & 3 (structured output + anomaly).** [State] seq2seq, energy-based models, structured prediction with NNs. [Derive] deep SVDD objective (Ruff et al. 2018) $\min_{\mathcal W}\tfrac1n\sum_i\|\phi(x_i;\mathcal W)-c\|^2+\tfrac\lambda2\sum_l\|W^l\|_F^2$ and soft-boundary variant with $R^2$ and $\nu$; autoencoder reconstruction-error scoring; likelihood-based deep AD. [State] the Ruff et al. 2021 unifying view (density/one-class/reconstruction × shallow/deep) and LRP/deep-Taylor for explaining anomalies. [Apply] six-part template for deep SVDD.
- **Day 17 ‡ — Connective threads day.** [Recall] build the two master tables: (a) every generalised-eigenvalue method (PCA, kPCA, CCA, kCCA, LLE, Fisher LDA, spectral clustering) with its $A,B$; (b) every shallow↔deep pair (kPCA↔autoencoder, OC-SVM/SVDD↔deep SVDD, KDE↔likelihood NN). [Derive] the generic Lagrangian for $\min w^\top A w$ s.t. $w^\top Bw=1$ → $Aw=\lambda Bw$.
- **Day 18 ‡ — Programming drills.** [Code] from memory, timed on paper then verify: HMM `forward` (scaled + unscaled), `backward`, `viterbi`, `sample_hmm`, `validate(A,B,pi)`; `lle_weights`; `pca`/`kpca`; `cca`; `kmeans`; OC-SVM scoring; linear autoencoder. (Reference code in Section 6.)
- **Day 19 ‡ — Mock Exam #2 (full, strict 120 min).** Simulate the five-slot exam with rotated topics: MC across all 12; applied = t-SNE or ICA; derivation = CCA; kernel proof = string/WD kernel PSD; programming = LLE weights or PCA. Grade yourself against Sections 7–8.
- **Day 20 — Error remediation.** Re-derive every item you missed in Mock #2; re-write failed code. Re-read Rabiner §III and the LLE appendix for any shaky step.
- **Day 21 — Final consolidation.** [Recall] one-line identities for all 12 topics; the six-part template; the eigenvalue and shallow/deep tables. Light re-write of HMM forward + LLE weights. Rest before exam.

**14-day compression:** do only ‡ days (1–5, 8–11, 15–19) plus a merged review/mock on the final two days; fold HMM learning (Day 8) and programming drills (Day 18) together; treat Deep Learning as recognition-level only.

### 4. Per-topic "derive / state / code" checklists

**LLE.** *Derive:* weight solution $w^*=C^{-1}\mathbf 1/(\mathbf 1^\top C^{-1}\mathbf 1)$; objective value $1/(\mathbf 1^\top C^{-1}\mathbf 1)$; embedding = bottom non-zero eigenvectors of $M=(I-W)^\top(I-W)$. *State:* three steps; why weights are invariant to rotation/translation/scaling; regularisation when $k>D$. *Code:* `lle_weights`.

**t-SNE.** *Derive:* KL cost; Student-t $q_{ij}$; gradient. *State:* crowding, heavy tail, perplexity, symmetric SNE, Barnes–Hut. *Code:* pairwise affinity matrix.

**CCA.** *Derive:* constrained covariance maximisation → block generalised eig; $\alpha=\beta$; kCCA + regularisation. *State:* correlation = eigenvalue; two-view applications; tkCCA. *Code:* `cca` via generalised eig.

**HMM.** *Derive:* forward/backward/Viterbi/Baum–Welch; scaling. *State:* three problems; forward=likelihood, Viterbi=decoding, Baum–Welch=learning. *Code:* `forward` (±scaling), `backward`, `viterbi`, `sample_hmm`, `validate`.

**Kernels 1 (structured input).** *Derive:* Gram PSD; closure (sum/scale/product/tensor/composition/limit); indicator kernel; normalised kernel. *State:* Mercer; spectrum/mismatch/WD/Fisher/graph kernels. *Code:* Gram matrix of a spectrum kernel.

**Kernels 2 (structured output).** *Derive:* structural SVM margin constraint; joint kernel tensor-product PSD. *State:* $\Psi(x,y)$; separation oracle; KDE; splice-site/protein apps. *Code:* argmax inference for a linear-chain score.

**ICA/BSS.** *Derive:* whitening + rotation; negentropy/kurtosis; FastICA fixed point. *State:* InfoMax, JADE, TDSEP/SOBI, SSA; ambiguities; identifiability. *Code:* whitening + one FastICA update.

**Kernel anomaly detection.** *Derive:* OC-SVM primal/dual + $\nu$-property; SVDD hypersphere; equivalence. *State:* KDE vs OC-SVM; kPCA reconstruction error. *Code:* score $= \sum_i\alpha_i k(x,x_i)$.

**Representation learning.** *Derive:* linear AE ≡ PCA. *State:* dictionary learning, NMF, self-supervised; kPCA↔AE. *Code:* linear autoencoder step.

**DL structured input.** *Derive:* weight-sharing parameter count. *State:* CNN equivariance, pooling invariance, GNN message passing, permutation symmetry. *Code:* 1D convolution in NumPy.

**DL structured output.** *State:* seq2seq, energy-based models, structured prediction. *Derive:* softmax cross-entropy gradient. *Code:* softmax + CE loss.

**DL anomaly detection.** *Derive:* deep SVDD objective (+ soft-boundary). *State:* Ruff et al. 2021 unifying view; AE reconstruction scoring; LRP/deep-Taylor explanation. *Code:* deep SVDD loss + anomaly score.

### 5. The "applied ML question" six-part template

Reusable skeleton: **(1) Method + one-line justification; (2) Hyperparameters; (3) Learned parameters; (4) Preprocessing; (5) Interpretation of model/latent states; (6) Limitations.**

**HMM (2026 scenario: latent-state customer-interaction sequences).**
1. *Method:* discrete-emission HMM — sequential data with an unobserved discrete latent state generating observed interactions; Markov assumption fits session dynamics.
2. *Hyperparameters:* number of hidden states $h$; number of observation symbols $d$; #EM restarts; convergence tolerance; regularisation/smoothing of $A,B$.
3. *Learned parameters:* transition matrix $A\in\mathbb R^{h\times h}$, emission matrix $B\in\mathbb R^{h\times d}$, initial distribution $\pi\in\mathbb R^h$ (via Baum–Welch).
4. *Preprocessing:* tokenise interactions into a finite symbol alphabet; segment into per-user sessions; handle variable lengths; encode as integer indices; optionally merge rare events.
5. *Interpretation:* latent states = behavioural phases (e.g. "browsing", "comparing", "checkout"); $A$ = phase-transition dynamics; $B$ = which actions each phase emits; Viterbi decodes a user's phase trajectory.
6. *Limitations:* Markov/stationarity assumptions; must fix $h$ a priori; EM local optima; conditional independence of observations given state; no long-range memory.

**t-SNE (visualisation).** 1. Non-parametric 2-D embedding preserving local neighbourhoods. 2. Perplexity, learning rate, #iterations, early-exaggeration. 3. *No reusable parameters* — the map (point coordinates) is the only output (non-parametric). 4. Standardise/PCA-preprocess to ~30–50 dims; remove duplicates. 5. Clusters = local structure; inter-cluster distances & cluster sizes are *not* faithful. 6. No out-of-sample mapping; stochastic; distorts global geometry; perplexity-sensitive.

**One-class SVM / deep SVDD (anomaly detection).** 1. One-class boundary around normal data. 2. $\nu$ (outlier fraction), kernel bandwidth (OC-SVM); network architecture, $\lambda$, center $c$ (deep SVDD). 3. Dual $\alpha$ and $\rho$ (OC-SVM); network weights + radius $R$ (deep SVDD). 4. Scale/normalise features; for deep SVDD pre-train an autoencoder, set $c$ = mean encoding, remove bias units to avoid collapse. 5. Decision score = distance to boundary/center; support vectors describe the normal region. 6. Sensitive to contamination and bandwidth; kernel version scales poorly; deep version can collapse to trivial solution.

**ICA (source separation).** 1. Linear unmixing of statistically independent non-Gaussian sources. 2. #components, nonlinearity $g$ (logcosh/kurtosis), whitening choice. 3. Unmixing matrix $W$ (⇒ mixing $A=W^{-1}$). 4. Center, whiten (PCA), optionally band-pass. 5. Rows of $W$ = spatial filters; recovered sources = independent generators (e.g. EEG artefacts). 6. Scale/sign & permutation ambiguity; ≤1 Gaussian source; assumes linear instantaneous mixing and stationarity.

**CCA (two-view data).** 1. Find maximally correlated projections of two views. 2. #components; kCCA kernel + regularisation $\kappa$. 3. Canonical directions $w_x,w_y$ (or dual $\alpha$). 4. Center each view; standardise; regularise covariances. 5. Canonical correlations = shared-variability strength; directions = aligned subspaces. 6. Linear (unless kernelised); needs regularisation when $XX^\top$ singular; sensitive to sample size vs. dimension.

**String/graph kernel SVM (sequence/graph data).** 1. SVM with a structured kernel (spectrum/WD/graph). 2. k-mer length, mismatch/shift, WD degree, SVM $C$. 3. Dual $\alpha$, bias $b$, support vectors. 4. Fixed alphabet, sequence length handling; graph canonicalisation. 5. Support sequences/subgraphs; WD weights show position importance. 6. Kernel/quadratic memory; kernel-design dependent; interpretability limited without explanation methods.

**Autoencoder (representation learning).** 1. Nonlinear encoder–decoder minimising reconstruction error. 2. Latent dim, depth/width, activation, regularisation (sparsity/denoising), LR. 3. Encoder/decoder weights. 4. Normalise inputs; for images augment. 5. Latent code = compressed representation; linear AE recovers PCA subspace. 6. Reconstruction ≠ semantic quality; can learn identity if overcomplete without regularisation.

### 6. Programming preparation — NumPy reference code

Be able to write these from memory. The HMM routines are directly exam-attested (2026 and 2021).

```python
import numpy as np

# ---- HMM validation ----
def validate(A, B, pi, tol=1e-8):
    h = A.shape[0]
    assert A.shape == (h, h)
    assert B.shape[0] == h
    assert pi.shape == (h,)
    assert np.allclose(A.sum(axis=1), 1, atol=tol)
    assert np.allclose(B.sum(axis=1), 1, atol=tol)
    assert np.isclose(pi.sum(), 1, atol=tol)
    return True

# ---- Forward algorithm (unscaled): returns P(obs|lambda) ----
def forward(A, B, pi, obs):
    h = A.shape[0]; T = len(obs)
    alpha = np.zeros((T, h))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
    return alpha, alpha[-1].sum()

# ---- Forward algorithm (scaled): returns log P(obs) ----
def forward_scaled(A, B, pi, obs):
    h = A.shape[0]; T = len(obs)
    alpha = np.zeros((T, h)); c = np.zeros(T)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    return alpha, c, np.sum(np.log(c))

# ---- Backward algorithm ----
def backward(A, B, obs):
    h = A.shape[0]; T = len(obs)
    beta = np.zeros((T, h)); beta[-1] = 1.0
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])
    return beta

# ---- Viterbi (log domain) ----
def viterbi(A, B, pi, obs):
    h = A.shape[0]; T = len(obs)
    logA, logB, logpi = np.log(A+1e-300), np.log(B+1e-300), np.log(pi+1e-300)
    delta = np.zeros((T, h)); psi = np.zeros((T, h), dtype=int)
    delta[0] = logpi + logB[:, obs[0]]
    for t in range(1, T):
        scores = delta[t-1][:, None] + logA        # (h_prev, h_cur)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(h)] + logB[:, obs[t]]
    path = np.zeros(T, dtype=int); path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path, delta[-1].max()

# ---- Simulate/sample from an HMM ----
def sample_hmm(A, B, pi, T, rng=np.random.default_rng()):
    h = A.shape[0]; d = B.shape[1]
    states = np.zeros(T, dtype=int); obs = np.zeros(T, dtype=int)
    states[0] = rng.choice(h, p=pi)
    obs[0] = rng.choice(d, p=B[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(h, p=A[states[t-1]])
        obs[t] = rng.choice(d, p=B[states[t]])
    return states, obs

# ---- LLE reconstruction weights for one point ----
def lle_weights(xi, neighbors, reg=1e-3):
    # neighbors: (k, D) array of the k nearest neighbours of xi
    Z = neighbors - xi                    # (k, D)
    C = Z @ Z.T                           # local Gram (k, k)
    C += reg * np.trace(C) * np.eye(len(C))
    w = np.linalg.solve(C, np.ones(len(C)))
    return w / w.sum()

# ---- PCA ----
def pca(X, d):
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:d]
    return Xc @ vecs[:, idx], vals[idx]

# ---- Kernel PCA ----
def kpca(K, d):
    n = K.shape[0]; H = np.eye(n) - np.ones((n, n))/n
    Kc = H @ K @ H
    vals, vecs = np.linalg.eigh(Kc)
    idx = np.argsort(vals)[::-1][:d]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))

# ---- CCA via generalised eigenvalue problem ----
def cca(X, Y, reg=1e-6):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Sxx = np.cov(X, rowvar=False) + reg*np.eye(X.shape[1])
    Syy = np.cov(Y, rowvar=False) + reg*np.eye(Y.shape[1])
    Sxy = (X.T @ Y) / (len(X)-1)
    M = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
    vals, Wx = np.linalg.eig(M)
    idx = np.argsort(vals.real)[::-1]
    return vals.real[idx], Wx[:, idx].real

# ---- k-means ----
def kmeans(X, k, iters=100, rng=np.random.default_rng()):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        D = ((X[:, None, :] - C[None])**2).sum(-1)
        a = D.argmin(1)
        newC = np.array([X[a == j].mean(0) if np.any(a==j) else C[j] for j in range(k)])
        if np.allclose(newC, C): break
        C = newC
    return C, a

# ---- One-class SVM scoring given dual alphas + rho ----
def ocsvm_score(x, SV, alpha, rho, kernel):
    return sum(a * kernel(x, sv) for a, sv in zip(alpha, SV)) - rho  # >=0 normal

# ---- Simple linear autoencoder (one gradient step) ----
def ae_step(X, W1, W2, lr):
    H = X @ W1; Xhat = H @ W2
    err = Xhat - X
    gW2 = H.T @ err / len(X)
    gW1 = X.T @ (err @ W2.T) / len(X)
    return W1 - lr*gW1, W2 - lr*gW2, (err**2).mean()
```

### 7. Worked solutions to Q3 and Q4

**Q3 — LLE objective + Lagrange derivation.**

*What LLE optimises.* For each point $x_i$ with neighbours $\{\eta_j\}$, LLE minimises the local reconstruction error $E(w)=\|x_i-\sum_j w_j\eta_j\|^2$ subject to $\sum_j w_j=1$. Using the sum-to-one constraint, $x_i-\sum_j w_j\eta_j=\sum_j w_j(x_i-\eta_j)$, so
$$E(w)=\Big\|\sum_j w_j(x_i-\eta_j)\Big\|^2=\sum_{j,k}w_jw_k\underbrace{(x_i-\eta_j)^\top(x_i-\eta_k)}_{C_{jk}}=w^\top C w,$$
with $C$ the local Gram (covariance) matrix. Globally, LLE then finds embeddings $Y$ minimising $\sum_i\|y_i-\sum_j W_{ij}y_j\|^2=\mathrm{tr}(Y^\top M Y)$ with $M=(I-W)^\top(I-W)$, solved by the bottom non-zero eigenvectors of $M$.

*Constrained minimisation.* Solve $\min_w w^\top C w$ s.t. $\mathbf 1^\top w=1$. Lagrangian:
$$\mathcal L(w,\lambda)=w^\top C w-\lambda(\mathbf 1^\top w-1).$$
Stationarity: $\nabla_w\mathcal L=2Cw-\lambda\mathbf 1=0\Rightarrow w=\tfrac\lambda2 C^{-1}\mathbf 1$ (using $C=C^\top$, $C^{-1}$ exists). Impose the constraint:
$$\mathbf 1^\top w=\tfrac\lambda2\,\mathbf 1^\top C^{-1}\mathbf 1=1\Rightarrow \tfrac\lambda2=\frac{1}{\mathbf 1^\top C^{-1}\mathbf 1}.$$
Therefore
$$\boxed{\,w^*=\frac{C^{-1}\mathbf 1}{\mathbf 1^\top C^{-1}\mathbf 1}\,}.$$

*Objective value.* Substitute:
$$(w^*)^\top C w^*=\frac{(C^{-1}\mathbf 1)^\top C (C^{-1}\mathbf 1)}{(\mathbf 1^\top C^{-1}\mathbf 1)^2}=\frac{\mathbf 1^\top C^{-1}\mathbf 1}{(\mathbf 1^\top C^{-1}\mathbf 1)^2}=\boxed{\frac{1}{\mathbf 1^\top C^{-1}\mathbf 1}},$$
using $C^{-1}CC^{-1}=C^{-1}$ and symmetry. (In practice $C$ is regularised, $C\leftarrow C+\tfrac{\text{reg}}{k}\mathrm{tr}(C)I$, when the neighbourhood size exceeds the input dimension so that $C$ is singular.)

**Q4 — PSD proof + feature map for $k_2((x,y),(x',y'))=k_1(x,x')\,\mathbb 1[y=y']$.**

*Setup.* $k_1$ is PSD on the input space with feature map $\phi_1$ so that $k_1(x,x')=\langle\phi_1(x),\phi_1(x')\rangle$. The indicator $k_\delta(y,y')=\mathbb 1[y=y']$ is itself a PSD kernel with feature map $\phi_\delta(y)=e_y$ (the one-hot/indicator vector in $\mathbb R^{|\mathcal Y|}$, or the basis vector $\delta_y$ in $\ell^2(\mathcal Y)$), since $\langle e_y,e_{y'}\rangle=\mathbb 1[y=y']$.

*PSD proof (Gram-matrix argument).* Take any finite set $\{(x_i,y_i)\}_{i=1}^n$ and coefficients $c\in\mathbb R^n$. Then
$$\sum_{i,j}c_ic_j\,k_2((x_i,y_i),(x_j,y_j))=\sum_{i,j}c_ic_j\,k_1(x_i,x_j)\,\mathbb 1[y_i=y_j]=\sum_{\ell\in\mathcal Y}\sum_{i,j:\,y_i=y_j=\ell}c_ic_j\,k_1(x_i,x_j).$$
For each fixed label $\ell$, the inner double sum is $\sum_{i,j\in I_\ell}c_ic_j k_1(x_i,x_j)\ge 0$ because it is a quadratic form of the PSD kernel $k_1$ restricted to the index set $I_\ell=\{i:y_i=\ell\}$. A sum of non-negative terms is non-negative, so the whole expression is $\ge 0$. Hence $k_2$ is PSD. (Equivalently: $k_2$ is the product of two PSD kernels $k_1(x,x')$ and $\mathbb 1[y=y']$, and products of PSD kernels are PSD by the Schur product theorem.)

*Feature map.* $k_2$ is a product of kernels, so its feature map is the **tensor product** of the factor feature maps:
$$\Phi(x,y)=\phi_1(x)\otimes e_y,\qquad \langle\Phi(x,y),\Phi(x',y')\rangle=\langle\phi_1(x),\phi_1(x')\rangle\,\langle e_y,e_{y'}\rangle=k_1(x,x')\,\mathbb 1[y=y'].$$
Concretely, $\Phi(x,y)$ is a block vector indexed by labels whose $y$-th block equals $\phi_1(x)$ and all other blocks are zero: $\Phi(x,y)=(\,\mathbb 1[\ell=y]\,\phi_1(x)\,)_{\ell\in\mathcal Y}$. This makes the indicator explicit — two pairs interact only when their labels match.

### 8. Most likely proof/derivation questions across the 12 topics (with sketches)

1. **LLE weight solution + objective value** (Section 7). Lagrange under $\mathbf 1^\top w=1$.
2. **CCA generalised eigenvalue problem.** Lagrangian with two constraints → $\alpha=\beta$ → $\begin{pmatrix}0&S_{xy}\\S_{yx}&0\end{pmatrix}\begin{pmatrix}w_x\\w_y\end{pmatrix}=\lambda\begin{pmatrix}S_{xx}&0\\0&S_{yy}\end{pmatrix}\begin{pmatrix}w_x\\w_y\end{pmatrix}$.
3. **PCA as eigenproblem + reconstruction equivalence.** $\max w^\top\Sigma w$ s.t. $\|w\|=1$ → $\Sigma w=\lambda w$; and $\|X-XUU^\top\|_F^2=\|X\|_F^2-\mathrm{tr}(U^\top\Sigma U)$.
4. **Kernel PSD closure** (sum/scale/product/tensor) + a specific construction like Q4 or a normalised kernel.
5. **HMM forward recursion derivation** from $\alpha_t(j)=P(x_{1:t},q_t=j)$ by marginalising $q_{t-1}$.
6. **Viterbi** as the max-product analogue; correctness by induction.
7. **Baum–Welch re-estimation** via EM: derive $\gamma,\xi$ and the M-step formulas.
8. **t-SNE gradient** $\partial C/\partial y_i=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)q_{ij}Z$.
9. **FastICA / negentropy** as non-Gaussianity maximisation after whitening; kurtosis fixed point.
10. **OC-SVM dual / $\nu$-property**, and SVDD hypersphere dual; their equivalence for Gaussian kernels.
11. **Linear autoencoder = PCA** (subspace equality).
12. **Deep SVDD objective** and its relation to SVDD; why bias-free architecture avoids hypersphere collapse.
13. **Structural-SVM constraint** and separation-oracle formulation; joint-kernel tensor-product PSD.
14. **kPCA double-centering** derivation $\tilde K=HKH$.

### 9. Exam-day tactics

- **Time budget (120 min, 5 questions):** ~5 min triage reading all questions; then roughly MC 12 min, applied-ML 22 min, derivation 28 min, kernel proof 25 min, programming 25 min; keep ~3 min buffer. Do the MC and the question you know best *first* to bank points and build confidence.
- **Partial credit on proofs:** always (i) state the objective and constraints, (ii) write the Lagrangian, (iii) take derivatives and set to zero, (iv) impose constraints, (v) box the result and verify a limiting/symmetry case. Each labelled step earns marks even if the algebra stalls.
- **Programming on paper:** write the recursion/initialisation first as comments, then fill code; state array shapes; don't chase perfect syntax — correct indexing and the recurrence matter most. For HMM, get the initialisation $\alpha_1=\pi\odot B_{:,x_1}$ and the update exactly right.
- **When stuck:** write the definition of every symbol; try the smallest case ($n=1$, $T=2$, $2\times2$); state the general principle (e.g. "this is a generalised eigenvalue problem", "products of PSD kernels are PSD") to secure conceptual marks; move on and return.
- **Applied question:** always answer all six sub-parts explicitly with the sub-headers (method/hyperparameters/learned parameters/preprocessing/interpretation/limitations) — the structure itself is graded.

## Recommendations

- **Start now with Tier 1 (Days 1–6):** the eigenvalue/Lagrange spine and kernel PSD theory are the load-bearing skills — they carry Q3, Q4, and half of MC every year. Benchmark: by end of Week 1 you can reproduce the LLE weight derivation, the CCA generalised eig, and the Q4 PSD proof *cold* in <10 min each. If you cannot, extend Week 1 by two days before proceeding.
- **Rehearse the six-part applied template on ≥6 methods (Days 8–16):** HMM, t-SNE, OC-SVM/deep SVDD, ICA, CCA, autoencoder. Benchmark: you can fill all six parts for any of them in <8 min.
- **Drill the attested code (Days 6, 18):** HMM forward/backward/Viterbi/sample/validate must be hand-writable in <10 min total. This is the single most predictable question — treat it as guaranteed points.
- **Take both mock exams under strict 120-min conditions (Days 14, 19).** Benchmark to sit the real exam confidently: ≥80% on Mock #2 with all five slots attempted. If below, spend Day 20 exclusively on the weakest two slots.
- **Confirm exam logistics on ISIS 48434** in week 1: aids/open-book status (assume closed-book until confirmed), room, and whether the programming question is on paper (it was in 2021/2026). If open-book is confirmed, build a one-page derivation/formula sheet keyed to Sections 4, 7, 8.
- **Escalation triggers:** if by Day 10 you cannot reproduce ≥6 of the 14 core derivations, drop Tier 3 deep-learning derivations to recognition-only and reinvest the time in Tiers 1–2.

## Caveats

- **Exam aids/open-book status is unconfirmed** for TU Berlin ML2; cheat-sheet/calculator rules found online belong to other universities and were explicitly excluded. Verify on ISIS 48434 or the exam cover sheet.
- **The five-slot structure is inferred** from the 2026 memory protocol corroborated by the 2021 protocol; memory protocols are approximate and per-question point values are not publicly documented. Topics filling each slot rotate year to year.
- **Some derivations (t-SNE gradient constant, exact scaling conventions in Baum–Welch) vary by source/notation;** the versions above follow the standard papers (van der Maaten & Hinton 2008; Rabiner 1989) but the lecture slides may use slightly different constants — prefer the course slides' convention where they differ.
- **The provided CCA slide deck** (tkCCA, canonical convolution, Swiss-constitution and Twitter examples) indicates the course goes beyond textbook CCA; expect application-flavoured MC/applied items drawn from such lecture-specific examples that generic references won't fully cover.
- **Deep-learning lectures (10–12) are the least documented** for this specific course; the plan treats them at recognition + one-applied-skeleton depth, which is the rational allocation given they did not appear as major written questions in the two attested years but could surface in MC or the applied slot.