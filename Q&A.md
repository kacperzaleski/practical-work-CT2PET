# Q&A notes — radiological / implementation explainers

Plain-language explainers, written so you can defend the protocol without having
to look anything up. Kept here so they don't leak into the formal text.

---

## 1. FDG — what it actually is, and why PET shows uptake

FDG = **F**luorodeoxy**g**lucose, specifically [18F]-FDG: a glucose molecule
where one hydroxyl group is replaced by radioactive fluorine-18. Workflow:
inject into the patient, wait ~60 min, scan.

Why it shows anything in PET: cells absorb FDG the same way they absorb normal
glucose (they "think" it's food). Once inside, an enzyme phosphorylates it —
but the modified structure can't continue down the metabolic pathway and can't
easily leave the cell. So FDG accumulates **proportionally to how much glucose
the cell is consuming**. Tumours, brain tissue, inflamed regions and heart
muscle all eat a lot of glucose, so they "light up". Fat and resting muscle
don't.

The fluorine-18 nucleus decays by emitting a positron; the positron annihilates
with a nearby electron, producing two 511 keV gamma photons travelling in
opposite directions. The PET scanner detects those coincident photon pairs and
reconstructs where the FDG (and therefore the glucose-hungry tissue) is. "FDG
uptake" = how much FDG ended up in a given region, which is the PET image
intensity (usually expressed as SUV, standardised uptake value).

---

## 2. Morphological closing, dilation, erosion

These are operations on **binary images** — every pixel is either 1
(foreground) or 0 (background). Forget medicine; the threshold mask is just a
black-and-white stencil.

- **Structuring element** = a small shape (here a 3×3 square of 1s) you slide
  over every pixel of the mask. It defines "the neighbourhood I look at around
  each pixel." A 3×3 square is the smallest isotropic 8-connected
  neighbourhood — the standard default.
- **Dilation** = at every pixel, if **any** pixel in its 3×3 neighbourhood is
  foreground, mark the centre as foreground. Effect: foreground blobs grow
  outward by one pixel, small gaps close.
- **Erosion** = the opposite: a pixel stays foreground only if **all** pixels
  in its 3×3 neighbourhood are foreground. Effect: blobs shrink inward by one
  pixel, thin protrusions vanish.
- **Closing** = dilation **followed by** erosion. The dilation step fills
  single-pixel holes and bridges narrow gaps between nearby blobs; the erosion
  step then shrinks everything back, so the outer boundaries end up close to
  the original size. Net effect: same overall blob shape, but pinhole gaps
  sealed and near-touching regions merged.

Why we need it: thresholding raw PET at the 75th percentile produces speckly
masks (a single noisy pixel below threshold inside a hot region becomes a
"hole"). Closing cleans that up so the attention UNet has a smoother target to
learn.

---

## 3. Why we compute PET-derived masks on the fly but learn a CT-derived mask

We have paired (CT, PET) data on disk, already normalised. So:

- **Mask from PET**: just a thresholding rule on the PET intensities. No
  learning, no model — open the PET .npy, take the 75th percentile, threshold,
  close. Done in milliseconds per slice, in the dataloader, at training time.
  We *have* the PET, so why train anything to predict from PET what we already
  have?
- **Mask from CT**: at **inference** time (when CPDM is asked to synthesise PET
  for a *new* patient) we **don't have the PET** — that's the whole point,
  we're generating it. But CPDM's diffusion process needs an attention map as
  conditioning input. So we need a way to produce an "approximate hot-region
  mask" from CT alone. CT doesn't directly encode glucose metabolism (it sees
  anatomy: bone, soft tissue, air), so we can't write a simple thresholding
  rule. Instead we train the attention UNet to look at CT and **predict** what
  the hot-region mask would look like if we had PET. Training targets come
  from PET (computed on the fly, free), inputs are CT, so at inference we only
  need CT.

Short version: PET-derived masks are a rule, not a model — anyone can compute
them when PET is available. The CT-derived prediction is the part that
actually has to be learned, because the relationship between anatomy and
metabolism is non-trivial.

---

## 4. Why BCE stabilises gradients vs. pure Dice loss

**Dice loss** $= 1 - \frac{2 |X \cap Y|}{|X| + |Y|}$, where $X$ is the
predicted foreground probability map and $Y$ the binary target.

The problem in early training is that the model's predictions are essentially
uniform (~0.5 everywhere, or close to all-zero). In that regime:

- The numerator $|X \cap Y|$ is tiny — there's almost no overlap with the
  target.
- The Dice gradient is global: it depends on sums over the whole image, so
  the signal a single pixel sees is small and **divided across all pixels**.
  Individual pixels get a weak, diffuse push.
- Worse, until the prediction actually starts overlapping the target, the
  gradient direction can flip-flop slice to slice (one slice has 0 overlap, the
  next has a sliver), which makes training noisy.

**BCE loss** $= -[y \log p + (1-y)\log(1-p)]$ per pixel. The gradient w.r.t.
the pre-sigmoid logit $z$ is simply $\sigma(z) - y$ — a clean, **per-pixel,
local** signal that points each pixel directly at its target with no
dependence on what other pixels are doing.

So the blend $0.7 \cdot \text{Dice} + 0.3 \cdot \text{BCE}$ does two jobs:

- BCE provides strong, well-conditioned gradients during the cold-start phase,
  before there's any meaningful overlap for Dice to amplify.
- Dice dominates the loss (weight 0.7) so the optimisation target stays
  foreground-overlap, which is what we actually care about given the heavy
  class imbalance (small hot regions, mostly background).

Without BCE, pure Dice will still converge eventually but takes longer to get
out of the uniform-prediction regime and is more sensitive to initialisation.

---

## 5. How the VQGAN parameter count (55.3 M) was obtained

The standard PyTorch idiom:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

run on the instantiated `VQModel` configured from `config/VQGAN-autoPET.yaml`
(base channels 128, ch_mult (1, 2, 4), z_channels 4, codebook size 8192).
That sum aggregates:

- Encoder convolutions and residual blocks.
- Decoder convolutions and residual blocks (roughly mirrors the encoder).
- The quantiser codebook (8192 × 4 = 32 768 learnable embedding values — small
  next to the convs).
- The pre- and post-quantiser 1×1 convs (`quant_conv`, `post_quant_conv`).

Adds up to **55.3 M** trainable parameters. You can verify any time with the
one-liner above on the loaded checkpoint. Nothing was estimated by hand from
the architecture spec — it's the literal `numel()` sum.

---

## 6. SpatialRescaler — what it does, and whether we added it

**What it does.** SpatialRescaler is a tiny preprocessing module that takes a
2D conditioning input (the attention map or the attenuation map, both at
64 × 64 with 1 channel) and turns it into a tensor that matches the latent
resolution and channel count the UNet denoiser expects (16 × 16, 4 channels).
It does two things in sequence:

1. **Downsample** by repeated interpolation. With `n_stages=2` and
   `multiplier=0.5` it calls `F.interpolate(..., scale_factor=0.5)` twice
   (bilinear by default), taking 64 → 32 → 16.
2. **Remap channels.** A single 1 × 1 `nn.Conv2d(in_channels=1,
   out_channels=4)` projects the 1-channel resized map to 4 channels matching
   `z_channels`.

That's the entire forward pass — no nonlinearity, no attention, no normalisation.
It's intentionally minimal: the conditioning maps already encode useful
spatial structure, so all the rescaler has to do is geometry + a learned
channel mix. The two 1 × 1 convs (one for the attention rescaler, one for the
attenuation rescaler) are the only learnable parameters in the whole module
and are trained jointly with the diffusion UNet.

**Where it came from.** Not added by us. It's a stock LDM utility
(`ldm.modules.encoders.modules.SpatialRescaler` from Rombach et al. 2022) that
was inherited by BBDM and then by the CPDM repo. In our tree it lives at
`model/BrownianBridge/base/modules/encoders/modules.py:106` — same file path
as the LDM original, untouched. We only configure it from YAML (`CondStageParams`
in the CPDM config).

---

## 7. Attenuation map — full walkthrough with every number

**Context.** PET attenuation correction needs to know how much each tissue
absorbs 511 keV gammas — the linear attenuation coefficient $\mu_{511}$ [$cm^{-1}$] of the tissue. CT measures attenuation at much lower energies
(80–140 kVp X-rays), expressed in Hounsfield Units (HU). The standard
PET/CT correction trick is a **bilinear HU → $\mu_{511}$ conversion**,
parametrised by the CT tube voltage (kVp). CPDM uses this as a conditioning
input so the diffusion model knows the per-pixel attenuation context the
real PET scanner would have used.

The piecewise-linear formula (parameters from Burger et al., *Accuracy of
CT-based attenuation correction in PET/CT bone imaging*) anchors $\mu_{511}$
at four HU values:

| HU | $\mu_{511}$ |
|---:|:---|
| $-1000$ (air) | $b_0 - 1000 \, a_0$ |
| $0$ (water) | $b_1$ |
| $1000$ (dense soft tissue / cancellous bone) | $b_1 + 1000 \, a_1$ |
| $3000$ (cortical bone) | $b_1 + 1000 \, a_1 + 2000 \, a_2$ |

with $a = (a_0, a_1, a_2)$ and $b = (b_0, b_1, b_2)$ chosen for the actual
tube voltage. For **140 kVp** (the kVp we use):
$a = (9.3\!\times\!10^{-5},\; 5.59\!\times\!10^{-5},\; 0.698\!\times\!10^{-5})$,
$b = (0.093,\; 0.093,\; 0.142)$. Outside the four anchor points the function
extrapolates linearly. The four anchors $\{-1000, 0, 1000, 3000\}$ are
hard-coded HU values; they're where the slope of the piecewise-linear function
changes, not data.

The model code (`attenuationCT_to_511`) builds a 1-D lookup table on a 0.1-HU
grid from $-1000$ to $3000$ via `scipy.interpolate.interp1d`, then looks up
every CT pixel in that table. After the lookup, `get_attenuation_map` takes
$\exp(-\mu_{511})$ so the map represents transmission (a number in $(0, 1]$,
1 = no attenuation, smaller = more absorption) rather than the raw $\mu$
value. That transmission-style map is what gets handed to the SpatialRescaler.

### The CPU-budget patch — why we rearrange the input

The paper assumes the CT was preprocessed with a `pixel / 2047` scheme that
preserves HU directly. We use a different normalisation: in
`preprocess_autopet.normalize_ct`, the HU range is clipped to
$[-1000, 3071]$ and **mapped linearly to $[-1, 1]$**:

$$
\text{ct\_norm} \;=\; \frac{\text{HU} - (-1000)}{3071 - (-1000)} \cdot 2 - 1
\;=\; \frac{\text{HU} + 1000}{4071} \cdot 2 - 1.
$$

So when we hand a normalised slice to `attenuationCT_to_511`, the input is
already in $[-1, 1]$ and is **not** in HU. We have to invert the above to get
HU back before we can use the bilinear table:

$$
\text{ct\_norm} + 1 \;=\; \frac{\text{HU} + 1000}{4071} \cdot 2
$$
$$
\frac{\text{ct\_norm} + 1}{2} \;=\; \frac{\text{HU} + 1000}{4071}
$$
$$
(\text{ct\_norm} + 1) \cdot \tfrac{1}{2} \cdot 4071 \;=\; \text{HU} + 1000
$$
$$
\boxed{\text{HU} \;=\; (\text{ct\_norm} + 1) \cdot 0.5 \cdot (3071 - (-1000)) \;+\; (-1000)}
$$

Every number in that expression has a concrete role:

- **+1** shifts $[-1, 1] \to [0, 2]$.
- **× 0.5** rescales $[0, 2] \to [0, 1]$ (fraction of the way through the
  clipped HU window).
- **× (3071 − (−1000)) = × 4071** expands $[0, 1]$ to the **width** of the HU
  window in HU.
- **+ (−1000)** shifts the result so its origin is at the **bottom** of the HU
  window (HU = −1000 = air).

In code (`get_attenuation_map`):
```python
HU_MIN, HU_MAX = -1000.0, 3071.0
HU_map = (np_x_cond + 1.0) * 0.5 * (HU_MAX - HU_MIN) + HU_MIN
KVP = 140
attenuation_factors = self.attenuationCT_to_511(KVP, HU_map)
attenuation_factors = np.exp(-attenuation_factors)
```
Three lines, in order:

1. **Denormalise** the input from $[-1, 1]$ back to HU.
2. **Bilinear lookup** to $\mu_{511}$ at 140 kVp.
3. **Exponentiate** $-\mu$ to get a transmission factor in $(0, 1]$.

The only paper-specific thing we changed is *line 1* — the
denormalisation constants — to match our HU-window normalisation instead of
the paper's pixel/2047 scheme. Everything downstream (the bilinear table, the
kVp parameters, the $\exp(-\mu)$ transform, the SpatialRescaler) is
untouched.

---

## 8. 1000 forward steps vs. 200 reverse sampling steps — what these are and when they apply

The two counts belong to **different phases** and do different jobs.

- `num_timesteps = 1000` — the **training** discretisation of the Brownian
  Bridge. Every training step samples a random integer $t \in \{1, \dots, 1000\}$,
  builds the corresponding noisy interpolate
  $x_t = (1 - m_t)\, x_0 + m_t\, y + \sigma_t\, \varepsilon$, feeds it to the
  UNet, and regresses the training target (here the bridge gradient, since
  `objective: grad`). We never actually run 1000 steps in a row during
  training — we sample one $t$ per example. The **1000** is only the size of
  the noise-schedule grid the training loop draws from. Bigger 1000 = finer
  interpolation between source and target = smoother learning signal, at no
  extra per-step compute cost.
- `sample_step = 200` — the **inference / sampling** discretisation. When we
  actually generate a PET slice, we do run the reverse process step-by-step
  (200 UNet forward passes back-to-back, from $t = 1000$ down to $t = 0$),
  and each step costs one denoiser call. With `skip_sample: True` and
  `sample_type: linear`, those 200 steps are 200 evenly-spaced indices out of
  the 1000-step training grid — same schedule, just subsampled.

**Why the two numbers can differ.** Training only needs to *learn* the
denoiser $\varepsilon_\theta(x_t, t)$ at every $t$ on the grid; that's a
single-step regression. Inference has to *unroll* the reverse chain and pays
one UNet call per step. The whole point of DDIM-style skipped sampling is:
train on a fine 1000-step grid (better learning signal), sample on a coarse
200-step grid (5× cheaper inference), and let the trained network interpolate
across the skipped steps. Same idea BBDM inherited from DDIM
(Song et al., 2021) — it is the standard trick, not a CPDM invention.

So: 1000 is training-only, 200 is inference-only. Both are hyperparameters,
not properties of the physical process.

---

## 9. The monotone schedule $m_t$ and its alternatives

`m_t` (called the "monotone schedule" in BBDM) controls **how the
interpolation weight moves from source to target** over $t \in [0, T]$. From
the forward equation
$x_t = (1 - m_t)\, x_0 + m_t\, y + \sigma_t\, \varepsilon$:

- At $t = 0$: $m_t \approx 0 \Rightarrow x_t \approx x_0$ (the target,
  i.e. PET latent).
- At $t = T$: $m_t \approx 1 \Rightarrow x_t \approx y$ (the source,
  i.e. CT latent).
- In between: $m_t$ smoothly interpolates.

"Monotone" just means it never goes backwards — $m_t$ is a non-decreasing
function of $t$.

**What CPDM uses.** `mt_type: linear` with $m_{\min} = 0.001$, $m_{\max} = 0.999$
(hard-coded in `BrownianBridgeModel.py:43`), i.e.
$m_t = \operatorname{linspace}(0.001, 0.999, 1000)$. Straight line from ~0 to
~1.

**What else is implemented.** `mt_type: sin` in the same file uses an
exponential-then-normalised schedule ($1.0075^t$ then divided by its final
value, capped at 0.999). Despite the name it isn't literally $\sin(\cdot)$ —
it just curves gently, spending more of the schedule near $m_t \approx 1$
(closer to the source).

**Other options that BB / diffusion literature has explored.** BB and DDPM
literature commonly compares:

- **Linear** — what we use. Uniform speed from target to source. Simple, no
  free parameters, works well when the source and target are at comparable
  scales.
- **Cosine** (Nichol & Dhariwal, 2021, originally for DDPM's $\bar\alpha$) —
  spends more schedule near $m_t \approx 0$ (target) so early denoising
  timesteps get more coverage. Generally sharper samples on natural images.
- **Sigmoid** — S-shaped: slow near both endpoints, fast in the middle.
  Sometimes helps when the endpoints matter (which for image-to-image
  translation, they do).
- **Sqrt / power schedules** — $m_t = (t/T)^p$ for various $p$. A tuning knob
  that biases the schedule toward either endpoint.

We stayed with `linear` because it is the BBDM default, is what CPDM
reports, and any different choice would need a controlled ablation to justify
— which is out of scope for a CPU-budget reproduction.

---

## 10. `max_var` and `eta` — what they are, what else was plausible

Both are hyperparameters of the Brownian Bridge, both set to `1.0` in
`config/CPDM-autoPET.yaml`. They live at different points in the process.

### `max_var` — peak noise level of the forward bridge

From `BrownianBridgeModel.py:53`:
$$
\text{variance}_t \;=\; 2 \cdot (m_t - m_t^2) \cdot \texttt{max\_var}.
$$

The factor $m_t - m_t^2$ is a parabola that is zero at $m_t = 0$ (pure
target) and $m_t = 1$ (pure source), and peaks at $m_t = 0.5$ with a value of
$0.25$. So the maximum variance actually injected during training is
$2 \cdot 0.25 \cdot \texttt{max\_var} = 0.5 \cdot \texttt{max\_var}$.

- `max_var = 1.0` (our choice, and BBDM default): peak noise variance of
  $0.5$ in latent space. Standard.
- Larger `max_var` (e.g. 2.0 or 4.0): more noise injected mid-trajectory.
  Makes the reverse process explore more, at the cost of harder training.
- Smaller `max_var` (e.g. 0.25 or 0.5): quieter forward process. The model
  regresses closer to a deterministic interpolation, and samples end up more
  conservative (fewer high-frequency variations, more mode-collapsed).
  Occasionally useful for very-tight image-to-image tasks like registration,
  but risks losing sample diversity.

BBDM shows `max_var = 1.0` is a good default across image-to-image tasks and
CPDM inherits that. It's the standard go-to.

### `eta` — reverse-process stochasticity (the DDIM-style knob)

Used only at sampling time, in `BrownianBridgeModel.py:204`:
$$
\sigma_t^{\text{reverse}} \;=\; \sqrt{\sigma^2_t} \cdot \texttt{eta}.
$$

`eta` scales the variance of the *reverse* transition, matching DDIM's
$\eta$ parameter:

- `eta = 1.0` (our choice, and BBDM default): fully stochastic reverse
  process. Each sampling step injects the full posterior variance. Different
  seeds give different samples — meaningful sample diversity.
- `eta = 0.0`: fully deterministic reverse process (DDIM-style). Same CT
  input always produces exactly the same PET. Faster and often sharper on
  natural images, but destroys sample diversity.
- Values in between (e.g. `eta = 0.5`): compromise — some stochasticity,
  some sharpness.

We picked `eta = 1.0` because (a) it is what CPDM reports, (b) medical
image-to-image translation actually benefits from expressing predictive
uncertainty on the model's part (a deterministic map would over-commit), and
(c) again, any deviation would need an ablation to defend.

**Summary.** Both being `1.0` is the "vanilla BBDM" configuration. Neither
was tuned by us. Plausible alternatives exist but each one would need a
controlled comparison, which the CPU budget didn't allow.

---

## 11. Why `pixel / 2047`? — and is it in the code?

Short version: **it is not in the CPDM code we have.** The
`attenuationCT_to_511` function receives an HU-valued array directly and
looks it up in the piecewise-linear table; there is no `/2047` division
anywhere in `model/BrownianBridge/CT2PETDiffusionModel.py`. The reference to
"`pixel/2047`" appears only in our own comments (`report.py:34`,
`report.py:318`) as a shorthand for "the paper's dataset stored CT in a
different scale than autoPET does".

### What the `2047` most likely refers to

The paper trains on a private, curated CT–PET dataset (not autoPET). In many
DICOM CT export pipelines, CT is stored as a **12-bit unsigned integer** with
the actual HU range approximately $[-1024, 1023]$ — i.e. $2048$ discrete
levels, values $0 \ldots 2047$ on disk. To normalise such stored pixels into
$[-1, 1]$ or $[0, 1]$, a common shortcut is `pixel / 2047` (or `pixel / 2048`),
because it maps the on-disk integer range to a fixed float range without
computing HU explicitly. This is a **dataset-storage artefact**, not a
medical convention — it depends on how the CT was written out and has nothing
to do with the underlying physics.

Because their pipeline stored/loaded CT that way, when they call
`attenuationCT_to_511(kVp, ct_array)`, the `ct_array` they pass in **is
already HU-scaled** (the storage convention gives HU-ish values back after
the divide by 2047 and a rescale — again, dataset-specific). Our pipeline
does not use that storage; we use `nibabel`, read the DICOM/NIfTI header,
apply the RescaleSlope/Intercept to get true HU, then clip and normalise to
$[-1, 1]$ explicitly via `normalize_ct`. So we have to undo *that* mapping
before calling the same downstream function.

### Is any of this actually visible in the code?

No. The `attenuationCT_to_511` function itself is dataset-agnostic — it just
takes an HU array. The "adaptation" is entirely upstream, at the interface
between our preprocessing and the function's expected input. Concretely, the
only line that has any autoPET-specific numbers is the denormalisation in
`get_attenuation_map`:

```python
HU_map = (np_x_cond + 1.0) * 0.5 * (HU_MAX - HU_MIN) + HU_MIN
```

which inverts *our* `normalize_ct`. In the original CPDM code, the analogous
line would have been whatever inverse of their storage convention got their
CT array back to HU (e.g. `HU_map = np_x_cond * 2047 - 1024` or similar) —
but that line wasn't preserved verbatim in the code we inherited, because
their dataset loader did the un-normalisation earlier in the pipeline.

**Honest bottom line.** The "`pixel/2047`" phrase is my paraphrase for
"whatever dataset-specific scaling the paper used". I have not verified the
exact divisor from the CPDM paper text or the private dataset spec, and it
is not visible in the code. What matters for us is only that the downstream
`attenuationCT_to_511` needs HU, and our `get_attenuation_map` supplies that
by inverting `normalize_ct`. Everything else is upstream storage detail.
