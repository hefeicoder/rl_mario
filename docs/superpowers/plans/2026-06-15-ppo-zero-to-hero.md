# PPO from 0 to Hero — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ppo-from-zero-to-hero.html` — a self-contained, intuition-first PPO tutorial in 8 collapsible chapters, styled like `/Users/xishu/cc/inference-technology.html`, with the project's Mario agent as the running example.

**Architecture:** One self-contained HTML file at the repo top level: KaTeX (CDN) for math, a hand-written stylesheet matching the reference's card/sidebar design, and a little vanilla JS for chapter collapse + active-section sidebar highlighting. A tiny Python validator (`scripts/check_doc.py`) checks structural integrity (anchors resolve, chapters present, KaTeX linked) after each chapter.

**Tech Stack:** HTML5, CSS3, vanilla JS, KaTeX 0.16 (CDN). Python 3 stdlib (`html.parser`) for the validator only.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `ppo-from-zero-to-hero.html` | The whole document (shell + CSS + JS + 8 chapters) |
| `scripts/check_doc.py` | Structural validator: parse HTML, verify anchors/chapters/KaTeX |
| `images/` | TensorBoard screenshots embedded in Ch 7–8 |
| `README.md` | Add a link to the tutorial in References |

## Component vocabulary (used by all chapter tasks)

The scaffold (Task 1) defines these CSS classes. Chapter tasks reuse them — do
not invent new ones without adding CSS:

- **Chapter card:** `<section class="chapter" id="ch-N">` → `chapter-header`
  (with `chapter-num`, `chapter-title`, `chapter-label`, `chevron`) →
  `chapter-body` (the content).
- **Section heading inside a chapter:** `<h2 class="section-title" id="ch-N-slug">`.
- **Figure:** `<div class="diagram-box"><pre>…ascii…</pre></div>` +
  `<p class="diagram-caption">…</p>`.
- **Display math:** `<div class="math-block">$$ … $$</div>`. Inline math: `$…$`.
- **Callout aside:** `<div class="callout"><span class="callout-label">WHY IT
  MATTERS</span> … </div>`.
- **Code:** `<pre class="code-block"><code>…</code></pre>`.
- **Glossary item:** `<span class="glossary-term">advantage</span>
  <span class="glossary-def">…</span>`.
- **Mario anchor box:** `<div class="callout mario"><span class="callout-label">
  IN OUR MARIO RUN</span> …</div>` — the recurring "here's how this shows up in
  the project" aside.

---

## Task 1: Scaffold — shell, CSS, JS, validator, one placeholder chapter

**Files:**
- Create: `ppo-from-zero-to-hero.html`
- Create: `scripts/check_doc.py`

- [ ] **Step 1: Write the validator `scripts/check_doc.py`**

```python
"""Structural validator for ppo-from-zero-to-hero.html.

Checks (no rendering): the file parses, every internal #anchor resolves to an
element id, KaTeX is linked, and the expected number of chapters is present.
Usage: python scripts/check_doc.py [expected_chapter_count]
"""
import sys
from html.parser import HTMLParser


class DocParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.anchors = []
        self.chapter_count = 0
        self.has_katex = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("href", "").startswith("#"):
            self.anchors.append(a["href"][1:])
        if tag == "section" and "chapter" in a.get("class", "").split():
            self.chapter_count += 1
        if tag == "link" and "katex" in a.get("href", ""):
            self.has_katex = True


def main():
    expected_chapters = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    html = open("ppo-from-zero-to-hero.html", encoding="utf-8").read()
    p = DocParser()
    p.feed(html)

    errors = []
    missing = [a for a in p.anchors if a and a not in p.ids]
    if missing:
        errors.append(f"broken anchors (no matching id): {sorted(set(missing))}")
    if not p.has_katex:
        errors.append("KaTeX stylesheet not linked")
    if expected_chapters and p.chapter_count != expected_chapters:
        errors.append(f"expected {expected_chapters} chapters, found {p.chapter_count}")

    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {p.chapter_count} chapters, {len(p.ids)} ids, "
          f"{len(set(p.anchors))} anchors all resolve, KaTeX linked")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the full HTML scaffold `ppo-from-zero-to-hero.html`**

Create the file with the complete shell below. It contains the `<head>` (KaTeX),
the full stylesheet, the layout (sidebar + container), the JS (collapse +
active-section observer), and exactly **one** placeholder chapter so structure
is verifiable before content exists.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PPO from 0 to Hero — Study Notes</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
        crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
          crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
          crossorigin="anonymous"
          onload="renderMathInElement(document.body, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$',  right: '$',  display: false}
            ], throwOnError: false });"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
           background: #f5f5f5; color: #1a1a1a; line-height: 1.65; padding: 40px 20px 80px; }
    .page-title { text-align: center; font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; color: #111; }
    .page-subtitle { text-align: center; font-size: 0.9rem; color: #777; margin-bottom: 32px; }
    .layout { display: flex; align-items: flex-start; max-width: 1180px; margin: 0 auto; gap: 28px; }
    .sidebar { flex: 0 0 256px; position: sticky; top: 24px; max-height: calc(100vh - 48px);
               overflow-y: auto; background: #fff; border-radius: 14px;
               box-shadow: 0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.06); padding: 18px; }
    .sidebar-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .1em;
                     color: #888; font-weight: 700; margin-bottom: 14px; }
    .sidebar nav ol { list-style: none; padding: 0; margin: 0; }
    .sidebar nav > ol > li { margin-bottom: 4px; }
    .sidebar nav > ol > li > a { display: block; font-weight: 700; color: #1a1a1a;
        text-decoration: none; padding: 5px 0; line-height: 1.35; font-size: 0.85rem; }
    .sidebar nav > ol > li > a:hover, .sidebar nav > ol > li > a.active { color: #1d7a45; }
    .sidebar nav ul { list-style: none; padding: 2px 0 6px; margin: 0; }
    .sidebar nav ul li a { display: block; padding: 3px 0 3px 12px; color: #777;
        text-decoration: none; font-size: 0.79rem; line-height: 1.4; border-left: 2px solid #eee; }
    .sidebar nav ul li a:hover { color: #1a1a1a; border-left-color: #bbb; }
    .sidebar nav ul li a.active { color: #1d7a45; border-left-color: #1d7a45; font-weight: 600; }
    html { scroll-behavior: smooth; }
    .chapter, .section-title[id] { scroll-margin-top: 18px; }
    .container { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 20px; }
    .chapter { background: #fff; border-radius: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.06); overflow: hidden; }
    .chapter-header { display: flex; align-items: center; justify-content: space-between;
        padding: 20px 24px; cursor: pointer; user-select: none; }
    .chapter-header-left { display: flex; align-items: center; gap: 16px; }
    .chapter-num { display: flex; align-items: center; justify-content: center; width: 40px;
        height: 40px; background: #1a1a1a; color: #fff; font-weight: 700; font-size: 1rem;
        border-radius: 8px; flex-shrink: 0; }
    .chapter-title-group { display: flex; flex-direction: column; gap: 2px; }
    .chapter-title { font-size: 1.05rem; font-weight: 700; color: #111; }
    .chapter-label { font-size: 0.68rem; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #1d7a45; }
    .chevron { width: 18px; height: 18px; flex-shrink: 0; transition: transform .2s; color: #999; }
    .chapter.collapsed .chevron { transform: rotate(-90deg); }
    .chapter-body { padding: 0 24px 24px; }
    .chapter.collapsed .chapter-body { display: none; }
    .chapter-body p { margin: 12px 0; }
    .chapter-body ul, .chapter-body ol { margin: 12px 0 12px 22px; }
    .chapter-body li { margin: 5px 0; }
    .section-title { font-size: 1rem; font-weight: 700; color: #111; margin: 26px 0 8px; }
    .diagram-box { background: #1a1a1a; color: #e8e8e8; border-radius: 10px; padding: 18px 20px;
        margin: 16px 0 6px; overflow-x: auto; }
    .diagram-box pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.8rem;
        line-height: 1.5; white-space: pre; }
    .diagram-caption { font-size: 0.8rem; color: #777; margin: 4px 0 16px; font-style: italic; }
    .math-block { background: #fafafa; border: 1px solid #eee; border-radius: 10px;
        padding: 14px 18px; margin: 16px 0; overflow-x: auto; }
    .callout { background: #f3f9f5; border-left: 3px solid #1d7a45; border-radius: 0 8px 8px 0;
        padding: 12px 16px; margin: 16px 0; font-size: 0.92rem; }
    .callout.mario { background: #fdf6f0; border-left-color: #e07b39; }
    .callout-label { display: block; font-size: 0.66rem; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #1d7a45; margin-bottom: 4px; }
    .callout.mario .callout-label { color: #e07b39; }
    .code-block { background: #1a1a1a; color: #e8e8e8; border-radius: 10px; padding: 14px 18px;
        margin: 16px 0; overflow-x: auto; font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.8rem; line-height: 1.5; }
    .glossary-term { font-weight: 700; color: #1d7a45; }
    .glossary-def { color: #444; }
    figure { margin: 16px 0; }
    figure img { max-width: 100%; border-radius: 10px; border: 1px solid #eee; display: block; }
    figure figcaption { font-size: 0.8rem; color: #777; margin-top: 6px; font-style: italic; }
    @media (max-width: 900px) {
      .layout { flex-direction: column; gap: 0; }
      .sidebar { position: static; flex: none; width: 100%; max-height: none; margin-bottom: 20px; }
    }
  </style>
</head>
<body>
  <div class="page-title">PPO from 0 to Hero</div>
  <div class="page-subtitle">An intuition-first tour of Proximal Policy Optimization, taught through a Super Mario Bros agent</div>

  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-title">Contents</div>
      <nav>
        <ol id="toc">
          <li><a href="#ch-1">1 · Placeholder</a></li>
        </ol>
      </nav>
    </aside>

    <main class="container" id="chapters">
      <section class="chapter" id="ch-1">
        <div class="chapter-header">
          <div class="chapter-header-left">
            <div class="chapter-num">1</div>
            <div class="chapter-title-group">
              <div class="chapter-label">Placeholder</div>
              <div class="chapter-title">Scaffold works</div>
            </div>
          </div>
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9l6 6 6-6"/></svg>
        </div>
        <div class="chapter-body">
          <p>If you can read this, collapse the chapter by clicking its header,
          and see math render: $a^2 + b^2 = c^2$ — the scaffold is working.</p>
        </div>
      </section>
    </main>
  </div>

  <script>
    // Collapse/expand a chapter when its header is clicked.
    document.querySelectorAll(".chapter-header").forEach(function (h) {
      h.addEventListener("click", function () {
        h.parentElement.classList.toggle("collapsed");
      });
    });
    // Highlight the sidebar link for the chapter/section currently in view.
    var links = {};
    document.querySelectorAll(".sidebar a").forEach(function (a) {
      links[a.getAttribute("href").slice(1)] = a;
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && links[e.target.id]) {
          document.querySelectorAll(".sidebar a.active").forEach(function (x) {
            x.classList.remove("active");
          });
          links[e.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-10% 0px -80% 0px" });
    document.querySelectorAll(".chapter, .section-title[id]").forEach(function (el) {
      observer.observe(el);
    });
  </script>
</body>
</html>
```

- [ ] **Step 3: Validate structure**

Run: `python scripts/check_doc.py 1`
Expected: `OK: 1 chapters, … ids, … anchors all resolve, KaTeX linked`

- [ ] **Step 4: Open in a browser and eyeball it**

Run: `open ppo-from-zero-to-hero.html`
Expected: centered title + subtitle, a white card "Chapter 1 / Scaffold works",
the Pythagorean math renders as a formula, clicking the header collapses/expands
the card, sidebar shows "1 · Placeholder".

- [ ] **Step 5: Commit**

```bash
git add ppo-from-zero-to-hero.html scripts/check_doc.py
git commit -m "feat: scaffold PPO tutorial (shell, CSS, JS, validator)"
git push
```

---

## Authoring contract for chapter tasks (Tasks 2–9)

Each chapter task does the same mechanical steps; the **content outline** is what
differs. For every chapter task:

1. **Replace/extend** the chapters in `#chapters` and the `<ol id="toc">` entries.
   (Task 2 replaces the placeholder Chapter 1; Tasks 3–9 append.)
2. Each chapter is a `<section class="chapter" id="ch-N">` using the component
   vocabulary above. Give every `section-title` a unique `id="ch-N-slug"` and a
   matching sidebar sub-link under the chapter's `<li>`.
3. Write the prose to the chapter's outline. Intuition-first: lead with the idea
   / analogy / diagram, then at most one or two light equations. Include the
   chapter's **"IN OUR MARIO RUN"** callout anchoring the concept to the project.
4. **Verify:** `python scripts/check_doc.py N` (N = chapters so far) → OK, then
   `open ppo-from-zero-to-hero.html` and confirm the new chapter renders, its
   math renders, and its sidebar links scroll correctly.
5. **Commit:** `git add … && git commit -m "feat: PPO tutorial Chapter N — <title>" && git push`.

Keep numbers/hyperparameters consistent with `src/config.py`
(`PPO_CONFIG`: `n_steps=512`, `batch_size=256`, `n_epochs=4`, `gamma=0.9`,
`gae_lambda=0.95`, `clip_range=0.1`, `ent_coef=0.01`; `N_ENVS=8`) and the real
runs (random baseline `x_pos`≈600–720; flagpole ≈3160; PPO_3 plateau ~800–850
from 100k–300k, breakout ~320k to ~1610 by 598k; `ppo_1M_baseline_1` stuck ~895).

---

## Task 2: Chapter 1 — The problem: Mario as an RL loop

**Files:** Modify `ppo-from-zero-to-hero.html` (replace placeholder chapter + TOC).

- [ ] **Step 1: Replace the placeholder with Chapter 1**

Content outline (author the prose to this):
- Hook: a human sees Mario and just plays; an agent sees 184k pixel numbers and
  must learn from a single scalar reward. That gap is what RL bridges.
- **The loop** — reuse the agent/environment diagram as a `diagram-box`:
  action out, (observation, reward) back.
- Define, intuition-first: `glossary-term` for **state**, **action**, **reward**,
  **policy** $\pi(a\mid s)$, **return** $R=\sum_t \gamma^t r_t$, **discount** $\gamma$.
  One math-block for the return.
- **IN OUR MARIO RUN** callout: state = 4×84×84 stacked frames; actions = the 7
  `SIMPLE_MOVEMENT` moves; reward = progress − time − death; `x_pos` is how we
  read progress; $\gamma=0.9$.
- Close by posing the driving question of the whole document: *given this loop,
  how do we make the policy better?* — which Chapter 2 answers.
- TOC: `<li><a href="#ch-1">1 · The RL loop</a><ul>…sub-links…</ul></li>`.

- [ ] **Step 2: Validate** — Run: `python scripts/check_doc.py 1` → Expected: OK.

- [ ] **Step 3: Browser check** — Run: `open ppo-from-zero-to-hero.html`
  Expected: Chapter 1 renders, return formula renders, sidebar sub-links scroll.

- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 1 — Mario as an RL loop"
git push
```

---

## Task 3: Chapter 2 — Policy gradients: push up what worked (REINFORCE)

**Files:** Modify `ppo-from-zero-to-hero.html` (append Chapter 2 + TOC entry).

- [ ] **Step 1: Add Chapter 2**

Content outline:
- Core intuition: after an episode, make the actions that led to high return
  **more probable**, and low-return actions **less probable**. That's policy
  gradient in one sentence.
- The score-function form as the single light equation (math-block):
  $\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, R\big]$.
  Explain each piece in words (the $\nabla\log\pi$ is "which way to nudge the
  weights to make this action more likely"; multiply by return = "and by how
  much / which sign").
- `diagram-box`: a trajectory with per-step returns, arrows up/down on action
  probabilities.
- The catch (sets up Ch 3): **variance**. The same policy can score wildly
  differently run to run, so the gradient estimate is extremely noisy — training
  would be slow and unstable.
- **IN OUR MARIO RUN** callout: this is why a single PPO episode's `max_x_pos`
  jumps around; the *signal* is buried in noise.
- Append TOC `<li>` with sub-links.

- [ ] **Step 2: Validate** — Run: `python scripts/check_doc.py 2` → Expected: OK.
- [ ] **Step 3: Browser check** — `open ppo-from-zero-to-hero.html`; Chapter 2 + math render.
- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 2 — policy gradients (REINFORCE)"
git push
```

---

## Task 4: Chapter 3 — Taming the noise: baselines, value functions & advantage

**Files:** Modify `ppo-from-zero-to-hero.html` (append Chapter 3 + TOC entry).

- [ ] **Step 1: Add Chapter 3**

Content outline:
- Idea: don't reward an action for the *total* return — reward it for being
  **better than expected**. Subtract a baseline.
- **Value function** $V(s)$ = "expected return from here." **Advantage**
  $A(s,a) = R - V(s)$ (math-block): positive = better than expected → push up;
  negative = worse → push down. Same direction, far less variance.
- **Actor-critic**: two networks (or heads) — actor = policy, critic = $V(s)$.
- **GAE** (generalized advantage estimation) as the **bias/variance knob**
  ($\lambda$): intuition only — small $\lambda$ = low variance/more bias, large
  $\lambda$ = the reverse. Cite `gae_lambda=0.95`.
- `diagram-box`: same trajectory as Ch 2 but arrows scaled by advantage, visibly
  calmer.
- **IN OUR MARIO RUN** callout: `train/explained_variance` (~0.86 in our run) is
  literally "how good is the critic's $V(s)$" — tie it to the advantage estimate.
- Append TOC `<li>`.

- [ ] **Step 2: Validate** — `python scripts/check_doc.py 3` → OK.
- [ ] **Step 3: Browser check** — render + math.
- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 3 — advantage & actor-critic"
git push
```

---

## Task 5: Chapter 4 — The other catch: updates that blow up (trust regions)

**Files:** Modify `ppo-from-zero-to-hero.html` (append Chapter 4 + TOC entry).

- [ ] **Step 1: Add Chapter 4**

Content outline:
- Problem: even with a clean gradient *direction*, taking too big a step is
  catastrophic. The policy changes, so the data you collected no longer reflects
  the new policy — one greedy update can destroy a good policy (the
  distribution-shift problem).
- `diagram-box`: a loss-landscape cartoon — small step climbs, giant step falls
  off a cliff.
- The fix in words: **limit how much the policy changes per update** — stay in a
  "trust region." Introduce **TRPO** as the rigorous-but-heavy version
  (constrains the KL divergence between old and new policy). Intuition only — no
  Lagrangian/KKT; one sentence that KL = "how different the new policy is from
  the old."
- Set up Ch 5: TRPO works but is complex; what if we could get the same
  step-limiting effect much more simply?
- **IN OUR MARIO RUN** callout: `train/approx_kl` (~0.01 healthy) is exactly this
  "how much did the policy move" measure; a spike means an unstable update.
- Append TOC `<li>`.

- [ ] **Step 2: Validate** — `python scripts/check_doc.py 4` → OK.
- [ ] **Step 3: Browser check.**
- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 4 — trust regions (TRPO intuition)"
git push
```

---

## Task 6: Chapter 5 — PPO: the clip that changed everything

**Files:** Modify `ppo-from-zero-to-hero.html` (append Chapter 5 + TOC entry).

- [ ] **Step 1: Add Chapter 5**

Content outline (the summit — give it room):
- The probability ratio $r(\theta) = \pi_\theta(a\mid s) / \pi_{\theta_{old}}(a\mid s)$
  = "how much more/less likely is this action now vs before" (math-block).
- The clipped surrogate objective (math-block):
  $L = \mathbb{E}\big[\min(r\,A,\ \mathrm{clip}(r, 1-\epsilon, 1+\epsilon)\,A)\big]$.
- Explain with a **picture, not a proof** (`diagram-box`): for a good action
  ($A>0$), reward goes up as $r$ rises but **flattens past $1+\epsilon$** — no
  incentive to overshoot. For a bad action ($A<0$), symmetric. The clip removes
  the reward for moving the policy too far → the trust-region effect, for free.
- Why it won: TRPO's stability benefit at a fraction of the complexity; works
  with ordinary SGD and minibatches. Cite `clip_range=0.1` ($\epsilon$).
- **IN OUR MARIO RUN** callout: `train/clip_fraction` = the fraction of samples
  where the ratio actually hit the clip (~0.3 in our run) — you can *see* the
  mechanism firing.
- Append TOC `<li>`.

- [ ] **Step 2: Validate** — `python scripts/check_doc.py 5` → OK.
- [ ] **Step 3: Browser check** — the clip figure + both equations render.
- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 5 — the clipped objective"
git push
```

---

## Task 7: Chapter 6 — PPO in practice: the full loop

**Files:** Modify `ppo-from-zero-to-hero.html` (append Chapter 6 + TOC entry).

- [ ] **Step 1: Add Chapter 6**

Content outline:
- The full training loop as a `diagram-box`: **collect** rollouts from N parallel
  envs → **compute** advantages (GAE) → **update** with the clipped objective for
  K epochs over minibatches → repeat.
- Walk the hyperparameters as the knobs they are (small table or list), each tied
  to `src/config.py`: `n_steps=512` (rollout length per env), `N_ENVS=8`
  (parallel envs), `batch_size=256`, `n_epochs=4` (reuse each batch K times),
  `gamma=0.9` (horizon), `gae_lambda=0.95`, `clip_range=0.1` ($\epsilon$),
  `ent_coef=0.01` (entropy bonus = keep exploring).
- A short `code-block` mirroring our `build_model`/`learn` call so the reader maps
  theory → the actual SB3 code in the repo.
- **IN OUR MARIO RUN** callout: parallel envs are why our throughput is ~207
  env-steps/s on 8 cores, and why training is emulator-bound (CPU), not
  GPU-bound.
- Append TOC `<li>`.

- [ ] **Step 2: Validate** — `python scripts/check_doc.py 6` → OK.
- [ ] **Step 3: Browser check.**
- [ ] **Step 4: Commit**

```bash
git add ppo-from-zero-to-hero.html
git commit -m "feat: PPO tutorial Chapter 6 — the full PPO loop in practice"
git push
```

---

## Task 8: Chapter 7 — Reading a healthy run

**Files:**
- Modify `ppo-from-zero-to-hero.html` (append Chapter 7 + TOC entry).
- Create: `images/healthy_run.png` (a real TensorBoard capture).

- [ ] **Step 1: Add the screenshot**

Save a real TensorBoard capture of a healthy run's `mario/max_x_pos` (or the
4-panel view) to `images/healthy_run.png`. If none is handy, capture one from
`tensorboard --logdir tb_logs`.

- [ ] **Step 2: Add Chapter 7**

Content outline:
- The metrics that matter, grouped: **progress** (`mario/flag_rate` = win rate,
  `mario/max_x_pos` toward 3160), **behavior** (`rollout/ep_rew_mean`,
  `ep_len_mean` — and the "length rises then falls = stopped dawdling, started
  running" insight), **health** (`explained_variance` ~0.86, `approx_kl` ~0.01,
  `clip_fraction` ~0.3, entropy decaying).
- `<figure>` embedding `images/healthy_run.png` with a caption.
- The shape of learning: **bursts, plateaus, breakthroughs**, and **non-monotonic
  noise** (rolling average + stochastic policy) — read the smoothed trend.
- Eval gotcha: **stochastic vs deterministic** — sampling explores (varied runs),
  greedy argmax is crisp for a trained agent but loops identically on the
  deterministic emulator.
- **IN OUR MARIO RUN** callout: point at the embedded figure's actual numbers.
- Append TOC `<li>`.

- [ ] **Step 3: Validate** — `python scripts/check_doc.py 7` → OK.
- [ ] **Step 4: Browser check** — screenshot loads, metrics sections render.
- [ ] **Step 5: Commit**

```bash
git add ppo-from-zero-to-hero.html images/healthy_run.png
git commit -m "feat: PPO tutorial Chapter 7 — reading a healthy run"
git push
```

---

## Task 9: Chapter 8 — When training gets stuck: diagnosing & escaping plateaus

**Files:**
- Modify `ppo-from-zero-to-hero.html` (append Chapter 8 + TOC entry).
- Create: `images/plateau_breakout.png` (the PPO_3-vs-baseline capture).

- [ ] **Step 1: Add the screenshot**

Save the real `mario/max_x_pos` capture showing PPO_3 breaking out (~1610 @ 598k)
while `ppo_1M_baseline_1` stays stuck (~895) to `images/plateau_breakout.png`.

- [ ] **Step 2: Add Chapter 8**

Content outline (the chapter the project earned):
- **Is it stuck, or just slow?** Signs of a *temporary* plateau (entropy still
  healthy, `max_x_pos` occasionally pokes up) vs a *dead* one (entropy collapsed,
  flat 200k+ steps, `approx_kl` near zero).
- **Diagnose before tuning:** (1) watch the agent play to see **where** it dies
  (which obstacle), (2) read entropy / `approx_kl` / `explained_variance`.
- **Escape levers, in order** (list): patience/more steps → exploration
  (`ent_coef`) → horizon (`gamma`) → learning rate / clip → reward shaping →
  curriculum learning. One line each on when to reach for it.
- **Decision flowchart** as a `diagram-box`: stuck? → entropy collapsed? → where
  does it die? → pick a lever.
- **Centerpiece worked example** with `<figure>` `images/plateau_breakout.png`:
  PPO_3 sat at ~800–850 (100k–300k), then **broke out ~320k** to ~1610 by 598k —
  *patience won, no tuning needed*; `ppo_1M_baseline_1` stuck flat at ~895 — *a
  genuinely stuck run*. The contrast teaches the judgment call: when to wait vs
  when to intervene.
- Close: ties back to the project's Stage 04 (tuning) and the README's failure-
  mode notes.
- Append TOC `<li>`.

- [ ] **Step 3: Validate** — `python scripts/check_doc.py 8` → OK.
- [ ] **Step 4: Browser check** — both screenshots load, flowchart renders.
- [ ] **Step 5: Commit**

```bash
git add ppo-from-zero-to-hero.html images/plateau_breakout.png
git commit -m "feat: PPO tutorial Chapter 8 — diagnosing & escaping plateaus"
git push
```

---

## Task 10: Polish — glossary, cross-links, README link, final pass

**Files:**
- Modify `ppo-from-zero-to-hero.html`
- Modify `README.md`

- [ ] **Step 1: Glossary + intro callout**

Add a short intro `callout` under the subtitle ("How to read this: 8 chapters,
each solving the problem the last one left — by Ch 5 you'll have built PPO"), and
sweep the chapters to ensure every first use of a key term uses `glossary-term`.

- [ ] **Step 2: Cross-links**

Where a chapter references another concept ("we'll fix this noise in Ch 3"), make
it an `<a href="#ch-N">` link so navigation is live.

- [ ] **Step 3: Link from the README**

In `README.md` §11 References, add:

```markdown
- **PPO from 0 to Hero** — an intuition-first PPO tutorial built on this
  project's Mario agent: [`ppo-from-zero-to-hero.html`](ppo-from-zero-to-hero.html)
```

- [ ] **Step 4: Final validation + full browser read-through**

Run: `python scripts/check_doc.py 8`
Expected: `OK: 8 chapters, … ids, … anchors all resolve, KaTeX linked`
Then `open ppo-from-zero-to-hero.html` and read top to bottom: all math renders,
all sidebar links work and highlight on scroll, both screenshots load, chapters
collapse/expand, responsive layout collapses sidebar when the window is narrow.

- [ ] **Step 5: Commit**

```bash
git add ppo-from-zero-to-hero.html README.md
git commit -m "feat: PPO tutorial polish — glossary, cross-links, README link"
git push
```

---

## Self-Review Notes

- **Spec coverage:** style/components (Task 1 scaffold), 8 chapters
  (Tasks 2–9, one per spec chapter), Mario-anchored via the "IN OUR MARIO RUN"
  callout in every chapter, intuition-first/light-math (authoring contract +
  per-chapter outlines cap equations), real screenshots in Ch 7–8 (Tasks 8–9),
  top-level placement + README link (Tasks 1 & 10), KaTeX/sidebar/collapse
  (Task 1). All covered.
- **Verification adaptation:** static doc → `scripts/check_doc.py` (anchors
  resolve, chapter count, KaTeX) + browser inspection each task, matching the
  spec's "no automated tests, per-chapter browser inspection."
- **Consistency:** chapter ids `ch-1`…`ch-8`; section ids `ch-N-slug`;
  components limited to the documented vocabulary; numbers pinned to
  `src/config.py` and the real runs (stated once in the authoring contract).
- **Screenshots are real artifacts** (Tasks 8–9 create them from `tb_logs`);
  if a capture is unavailable at execution time, that step says how to produce
  one rather than leaving a placeholder.
