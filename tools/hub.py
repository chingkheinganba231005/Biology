#!/usr/bin/env python3
"""Regenerate the home page (index.html) and README.md from tools/course.json."""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
course = json.load(open(os.path.join(ROOT, "tools", "course.json"), encoding="utf-8"))
PIP = "python3 -m pip install -U jupyter numpy pandas matplotlib scipy scikit-learn biopython rdkit requests"

cards = []
for tr in course["tracks"]:
    items = []
    for c in tr["chapters"]:
        kind = c.get("kind", "")
        tag = f'<span class="kind">{html.escape(kind)}</span>' if kind else ""
        inner = (f'<span class="num">{html.escape(c["n"])}</span>{tag}'
                 f'<h3>{html.escape(c["title"])}</h3><p>{html.escape(c["desc"])}</p>')
        if c["ready"]:
            items.append(f'<li><a class="card" href="{c["path"]}">{inner}<span class="go">Open →</span></a></li>')
        else:
            items.append(f'<li><div class="card soon">{inner}<span class="go">Coming next</span></div></li>')
    cards.append(f'<section class="track"><h2>{html.escape(tr["name"])}</h2><p class="blurb">{html.escape(tr["blurb"])}</p>'
                 f'<ul class="cards">{"".join(items)}</ul></section>')

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cells to Code</title>
<meta name="description" content="Biology from zero to computational biology: molecular biology, bioinformatics, omics, machine learning, simulation, drug discovery and cancer research, with interactive lessons and end-to-end projects on real data.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/guide.css">
<script>try{{var t=localStorage.getItem('c2c-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<style>
.hub{{max-width:1100px;margin:0 auto;padding:0 max(16px,env(safe-area-inset-right)) 80px max(16px,env(safe-area-inset-left))}}
.hero{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:32px;align-items:center;padding:52px 0 20px}}
@media (max-width:820px){{.hero{{grid-template-columns:1fr;padding-top:30px}}}}
.hero h1{{font-size:clamp(2.6rem,7vw,4.2rem);margin:.2rem 0 .6rem;letter-spacing:-.03em;line-height:1}}
.hero h1 em{{font-style:normal;color:var(--accent2)}}
.hero p{{font-size:1.18rem;color:var(--muted);max-width:60ch}}
.helix{{width:100%;max-width:420px;height:auto;justify-self:center}}
.path{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px;font:600 .85rem var(--sans)}}
.path span{{background:var(--accent-soft);color:var(--accent);border-radius:999px;padding:5px 12px}}
.track h2{{border-top:1px solid var(--line)}}.blurb{{color:var(--muted)}}
.cards{{list-style:none;padding:0;margin:1rem 0;display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}}
.card{{display:flex;flex-direction:column;height:100%;border:1px solid var(--line);border-radius:14px;padding:16px 18px;background:var(--surface);color:var(--ink);text-decoration:none;font-family:var(--sans);transition:border-color .2s,transform .2s}}
a.card:hover{{border-color:var(--accent);transform:translateY(-2px)}}
.card h3{{margin:.4rem 0 .3rem;font-size:1.08rem}}.card p{{margin:0 0 .8rem;font-size:.92rem;color:var(--muted);line-height:1.5;flex:1}}
.num{{font:700 .75rem var(--mono);color:var(--accent)}}.kind{{margin-left:8px;font:600 .72rem var(--sans);text-transform:uppercase;letter-spacing:.06em;color:var(--accent2)}}
.go{{font:600 .85rem var(--sans);color:var(--accent)}}.soon{{opacity:.6}}.soon .go{{color:var(--muted)}}
.how{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin:1rem 0}}
.how div{{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--surface);font:.92rem/1.5 var(--sans)}}
.how b{{display:block;margin-bottom:2px}}
</style></head><body>
<header class="topbar"><a class="brand" href="index.html"><span class="logo" aria-hidden="true"></span>Cells to Code</a><span class="crumb">Home</span><button class="themebtn" type="button" aria-label="Switch light or dark theme">Theme</button></header>
<main class="hub">
<section class="hero"><div><p class="eyebrow">Biology from zero to computational biology</p>
<h1>Cells to <em>Code</em></h1>
<p>Start with no biology at all. Finish able to analyse genomes, single cells and tumours, simulate molecules and cells, and run a computational drug-discovery project, on real data, in Python.</p>
<div class="path"><span>0 · Biology from zero</span><span>1 · Bioinformatics</span><span>2 · Omics and ML</span><span>3 · Structure and drug discovery</span><span>4 · Systems and simulation</span><span>5 · Cancer research</span></div></div>
<svg class="helix" viewBox="0 0 420 260" role="img" aria-label="A DNA double helix">
  <g fill="none" stroke-width="5" stroke-linecap="round">
    <path d="M10 130 C 60 20, 110 20, 160 130 S 260 240, 310 130 S 410 20, 410 20" stroke="var(--accent)"/>
    <path d="M10 130 C 60 240, 110 240, 160 130 S 260 20, 310 130 S 410 240, 410 240" stroke="var(--accent2)"/>
  </g>
  <g stroke="var(--line)" stroke-width="4" stroke-linecap="round">
    <line x1="35" y1="80" x2="35" y2="180"/><line x1="60" y1="55" x2="60" y2="205"/><line x1="85" y1="48" x2="85" y2="212"/><line x1="110" y1="55" x2="110" y2="205"/><line x1="135" y1="80" x2="135" y2="180"/>
    <line x1="185" y1="80" x2="185" y2="180"/><line x1="210" y1="55" x2="210" y2="205"/><line x1="235" y1="48" x2="235" y2="212"/><line x1="260" y1="55" x2="260" y2="205"/><line x1="285" y1="80" x2="285" y2="180"/>
    <line x1="335" y1="80" x2="335" y2="180"/><line x1="360" y1="55" x2="360" y2="205"/><line x1="385" y1="48" x2="385" y2="212"/>
  </g>
</svg></section>
<section><h2>How every chapter works</h2><div class="how">
<div><b>Plain words first</b>Every idea starts with a like-you're-five explanation, then the real biology, then the maths and code.</div>
<div><b>Interactive models</b>Playgrounds where you translate genes, fold proteins, run PCR or grow a tumour and watch what happens.</div>
<div><b>Real data</b>Genomes from NCBI, proteins from UniProt and the PDB, drugs from ChEMBL, tumours from TCGA and real single-cell data, saved so everything also runs offline.</div>
<div><b>A notebook you run</b>Each chapter is also a Jupyter notebook that reproduces every number and figure.</div></div></section>
{"".join(cards)}
<section><h2>Running the notebooks</h2>
<pre class="code"><code>{PIP}
jupyter notebook</code></pre>
<p>Each chapter folder holds <code>index.html</code> (the guide), a notebook, <code>images/</code>, <code>datasets/</code> and <code>animations/</code> (playgrounds). Guides and notebooks are generated from one source, <code>chapter.md</code>, by <code>tools/build.py</code>.</p>
<p class="meta">Educational material. Nothing here is medical advice.</p></section>
</main>
<script src="assets/guide.js"></script>
</body></html>
"""
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(page)

lines = ["# Cells to Code", "",
         "**Biology from zero to computational biology**: molecular biology, bioinformatics, omics, machine learning, "
         "simulation, drug discovery and cancer research, with interactive lessons and end-to-end projects on real data.", "",
         "**Start here:** open [`index.html`](index.html) in a browser, or the GitHub Pages site of this repository.", ""]
for tr in course["tracks"]:
    lines += [f"## {tr['name']}", "", tr["blurb"], "", "| # | Chapter | What you learn |", "|---|---|---|"]
    for c in tr["chapters"]:
        t = f"[{c['title']}]({c['path']})" if c["ready"] else f"{c['title']} *(coming)*"
        lines.append(f"| {c['n']} | {t} | {c['desc']} |")
    lines.append("")
lines += ["## Running the notebooks", "", "```bash", PIP, "jupyter notebook", "```", "",
          "Each chapter folder contains `index.html` (the guide), a Jupyter notebook that reproduces every number and figure, "
          "`images/`, `datasets/` and `animations/` (interactive playgrounds). Guides and notebooks are generated from a single "
          "literate source, `chapter.md`, with `python3 tools/build.py <chapter-folder>`; this page and the home page with "
          "`python3 tools/hub.py`.", "",
          "*Educational material. Nothing here is medical advice.*", ""]
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write("\n".join(lines))
print("hub and README written")
