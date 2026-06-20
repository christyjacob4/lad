# LAD — Learnable Advantage Density (interactive poster)

A self-contained, dependency-free static web build of the LAD research poster:
a single HTML page with nine interactive SVG charts, a live cohort-scoring
widget, and a KaTeX-rendered definition of the LAD metric. Everything runs
fully offline — KaTeX (CSS, JS, fonts) is vendored locally under
`vendor/katex/` and referenced with relative paths.

## Contents

```
web/
  index.html              # complete standalone HTML5 document (CSS + JS + RESULTS inline)
  README.md               # this file
  vendor/
    katex/                # KaTeX 0.16.11, vendored for offline math rendering
      katex.min.css
      katex.min.js
      fonts/              # KaTeX web fonts (woff2 / woff / ttf)
```

The charts are vanilla JavaScript with no external libraries. KaTeX is the only
third-party dependency, and it is bundled — there are no network requests at
runtime.

## Run it

Any static file server works. Pick one:

```bash
# Python (no install needed)
python3 -m http.server 8000
# then open http://localhost:8000

# Node (npx)
npx serve            # serves the current directory, prints the URL
```

Run the command from inside the `web/` directory. Opening `index.html` directly
via `file://` mostly works, but a static server is recommended so the relative
`vendor/katex/` paths and fonts resolve consistently across browsers.

To preview the page with every section's reveal animation already triggered,
append `#showall` to the URL (e.g. `http://localhost:8000/#showall`).

## Use it in Next.js

Two straightforward options:

1. **Drop into `public/`.** Copy the entire `web/` folder (or just the files
   you need) into your app's `public/` directory, e.g.
   `public/lad/index.html` + `public/lad/vendor/...`. It is then served as-is
   at `/lad/index.html`. The relative `vendor/katex/...` references resolve
   correctly because they are relative to the HTML file.

2. **Embed the markup in a page/component.** Move the contents of `<style>`
   into a CSS module (or a global stylesheet) and the chart `<script>` body into
   a client component. The chart code is already organized as small, idempotent
   init IIFEs that only touch elements by `id`, so it ports cleanly into a
   `useEffect`:

   ```tsx
   "use client";
   import { useEffect } from "react";
   import "katex/dist/katex.min.css";

   export default function LadPoster() {
     useEffect(() => {
       initLadCharts();   // the init function holding the chart IIFEs + katex.render calls
     }, []);
     return <div className="lad-root">{/* the poster markup */}</div>;
   }
   ```

   For the math you can either keep the vendored KaTeX or install the npm
   `katex` package and call `katex.render(...)` against the `#ladEquation` and
   `#ladEquationSub` target elements (same TeX strings used here).

   Wrap the init in a guard so it only runs once per mount, and remember the
   charts size themselves from each SVG's `viewBox` — no resize handler needed.

## Notes

- **Offline math.** KaTeX 0.16.11 is vendored under `vendor/katex/`. The page
  links `vendor/katex/katex.min.css` and loads `vendor/katex/katex.min.js`, both
  by relative path; the fonts referenced by the CSS live in
  `vendor/katex/fonts/`. No internet connection is required to render the
  equation.
- **The `RESULTS` object.** All figures read from a single `RESULTS` object
  defined near the top of the final `<script>` block in `index.html`. It is the
  one source of truth for every chart and stat.
- **The nine charts.** Derivation curve (γ slider + component toggles), headline
  predictive scatter, baseline comparison (expensive-baseline toggle), factor
  ablation, mechanistic scatter, reliability curve, causal-selection bars,
  LAD dose-response, the cost-vs-power Pareto frontier, plus the live
  score-a-cohort widget.
