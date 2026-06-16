# PPO from 0 to Hero — Design Spec

**Date:** 2026-06-15
**Repo:** https://github.com/hefeicoder/rl_mario (open source, MIT)
**Status:** Approved design, ready for implementation planning

## Goal

A chapter-based, **intuition-first** tutorial that teaches Proximal Policy
Optimization (PPO) "from 0 to hero," using the project's own Super Mario Bros
agent as the running example. Delivered as a single self-contained HTML
"study notes" document styled after
`/Users/xishu/cc/inference-technology.html`.

**Audience:** someone who finishes able to *explain* PPO, *tune* it, and
*rescue a stuck run* — not necessarily re-derive it from scratch. Math appears
but is not formally derived; diagrams and analogies carry the load.

**Success:** the document opens in any browser with no build step, renders its
math via KaTeX, navigates via a sticky sidebar TOC, and walks the reader up the
ladder of ideas that culminates in PPO — every concept anchored in the real
Mario project.

## Style reference

Match the design language of `/Users/xishu/cc/inference-technology.html`:

- Self-contained single HTML file; light gray background, white rounded cards,
  subtle shadows, green accent (`#1d7a45`), Apple-system font stack.
- **Collapsible numbered chapter cards** (`chapter`, `chapter-header`,
  `chapter-num`, `chapter-title`, `chapter-label`, `chapter-body`).
- **Sticky sidebar TOC** (`sidebar` + `nav > ol`) with chapter links and
  sub-section links, active-section highlighting on scroll.
- **KaTeX** for math: inline `$...$`, display `$$...$$` (CDN, auto-render on
  load), exactly as the reference configures it.
- Reuse the reference's content component vocabulary: `diagram-box` +
  `diagram-caption` for figures, `code-block`, `math-block`, glossary
  (`glossary-term` / `glossary-def`), and callout boxes for asides.
- Smooth scroll, `scroll-margin-top` offsets, responsive collapse of the
  sidebar under ~900px.

We author a fresh stylesheet in the same spirit (not a byte-for-byte copy);
class names mirror the reference so the look matches.

## Chapters (the ladder)

Problem-driven build-up: each chapter solves a problem and exposes the next,
until PPO is the natural answer. Mario anchors every concept.

1. **The problem: Mario as an RL loop.** Agent / env / state / action / reward
   / policy / return / discount γ, grounded in our setup (7 `SIMPLE_MOVEMENT`
   actions, the progress−time−death reward, `x_pos`). Frames the question:
   *how do we improve the policy?*
2. **Policy gradients: push up what worked (REINFORCE).** Make good actions
   more likely; the score-function intuition; one light equation
   ($\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)\, R]$).
   Ends on the catch: **insanely noisy**.
3. **Taming the noise: baselines, value functions & advantage.** Subtract
   "what we expected" → advantage $A = R - V(s)$; actor-critic; GAE as the
   bias/variance knob. Anchored in our `explained_variance` metric.
4. **The other catch: updates that blow up.** Why one large policy-gradient
   step destroys the policy (distribution shift); motivates limiting step size
   → trust regions (TRPO), intuition only — no Lagrangian/KKT math.
5. **PPO: the clip that changed everything.** The clipped surrogate objective
   explained with a picture (the clip function diagram), not a proof. "TRPO's
   benefit at a fraction of the complexity." The summit.
6. **PPO in practice: the full loop.** Collect rollouts (parallel envs) →
   compute advantages → clipped update for K epochs → repeat. Our
   hyperparameters as knobs: `n_steps`, `batch_size`, `n_epochs`, `clip_range`,
   `ent_coef`, `gamma`, `gae_lambda` (cite `src/config.py`).
7. **Reading a healthy run: our Mario agent.** Real curves — `mario/flag_rate`,
   `mario/max_x_pos`, `rollout/ep_rew_mean`, `ep_len_mean` — and the `train/`
   health metrics (`explained_variance`, `approx_kl`, `clip_fraction`,
   entropy). What healthy looks like; the burst/plateau/breakthrough pattern;
   non-monotonic noise; stochastic vs deterministic at eval time.
8. **When training gets stuck: diagnosing & escaping plateaus.** The chapter
   the project earned firsthand.
   - *Is it stuck or just slow?* Temporary plateau (entropy healthy, occasional
     pokes upward) vs dead one (entropy collapsed, flat 200k+ steps).
   - *Diagnose before tuning:* watch the agent play to see **where** it dies;
     read `approx_kl` / `explained_variance` / entropy.
   - *Escape levers, in order:* (1) patience/more steps, (2) exploration
     (`ent_coef`), (3) horizon (`gamma`), (4) learning rate / clip, (5) reward
     shaping, (6) curriculum.
   - *Decision flowchart* as a `diagram-box`.
   - **Centerpiece worked example:** our real runs — PPO_3 sat at ~800–850 from
     100k–300k, then broke out ~320k and reached ~1,610 by 598k (patience
     won); `ppo_1M_baseline_1` stuck flat at ~895. Two outcomes, side by side.

## Placement & integration

- File: `docs/ppo-from-zero-to-hero.html` (self-contained).
- Real screenshots: 1–2 actual TensorBoard captures saved under
  `docs/images/` and embedded in Ch 7–8 (the PPO_3 breakout chart for Ch 8).
- Link from the top-level `README.md` References section.

## Build approach

The document is large (reference is ~7k lines). Build incrementally so each
step is verifiable in a browser:

1. **Scaffold:** HTML shell, `<head>` with KaTeX, full CSS (cards, sidebar,
   components), empty sidebar TOC, page title/subtitle, one placeholder
   chapter — confirm it renders, math works, sidebar sticks, collapse works.
2. **Chapters 1–8, one at a time:** add each chapter's card + body content +
   its sidebar entries; verify rendering and math after each.
3. **Polish pass:** glossary entries, cross-links, real screenshots embedded,
   README link, final responsive/scroll check.

## Verification

No automated tests (it's a static document). Verification is per-chapter
browser inspection:

- File opens locally with no errors; KaTeX renders all `$`/`$$` blocks.
- Sidebar links scroll to the right chapter/section; active highlighting works.
- Chapters collapse/expand; responsive layout collapses the sidebar on narrow
  widths.
- All cited numbers/hyperparameters match `src/config.py` and the real runs.
- Embedded screenshots load.

## Out of scope

- Formal derivations/proofs (policy gradient theorem proof, TRPO monotonic
  improvement bound) — intuition-first by design.
- Interactive/animated widgets beyond collapse + smooth scroll.
- A general (non-Mario) version of the tutorial.
- Re-teaching DQN in depth (covered by Stage 02 and the main README).
