#!/usr/bin/env python3
"""
Enhanced TDG visualization with bidirectional linking:
  - Click a graph node -> highlights the fact row + source text span
  - Click a fact row   -> highlights the graph node + source text span
  - Click highlighted text -> highlights the graph node + fact row
  - Hover shows dependency tooltips

Usage:
    python visualize_linked.py sample_output.json -o linked.html
    python visualize_linked.py --demo -o linked.html
"""

import argparse
import json
import os
import sys

_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TDG -- Linked Visualizer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0d13; --surface: #12141c; --border: #1c1f2b;
  --text: #b8bfcc; --text-dim: #5a6173; --text-bright: #e4e8f0;
  --accent: #6366f1; --accent-dim: #6366f122;
  --start: #34d399; --start-dim: #34d39918; --start-bg: #0d2818;
  --end: #f87171; --end-dim: #f8717118; --end-bg: #2a0f0f;
  --dur: #818cf8; --dur-dim: #818cf818; --dur-bg: #14143a;
  --contains: #fbbf24; --contains-dim: #fbbf2418; --contains-bg: #2a2410;
  --unknown: #6b7280; --unknown-dim: #6b728018;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

/* -- Header ----------------------------------- */
header { padding: 28px 36px 20px; border-bottom: 1px solid var(--border); }
header h1 { font-size: 18px; font-weight: 700; color: var(--text-bright); }
header p { font-size: 12px; color: var(--text-dim); margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }

/* -- Tabs ------------------------------------- */
.tabs { display: flex; padding: 0 36px; border-bottom: 1px solid var(--border); background: #0e1018; overflow-x: auto; }
.tab { padding: 11px 18px; font-size: 13px; font-weight: 600; color: var(--text-dim); cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; transition: .15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--text-bright); border-bottom-color: var(--accent); }
.badge { display: inline-block; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; padding: 2px 7px; border-radius: 3px; margin-left: 7px; vertical-align: middle; }
.badge-historical { background: #1e293b; color: #93c5fd; }
.badge-legal { background: #1c1917; color: #fbbf24; }
.badge-medical { background: #0d2818; color: #6ee7b7; }
.badge-corporate { background: #1e1b2e; color: #c4b5fd; }
.badge-biographical { background: #2a0f0f; color: #fca5a5; }
.badge-unknown { background: #1a1a1a; color: #9ca3af; }

/* -- Layout ----------------------------------- */
.doc-panel { display: none; padding: 28px 36px; }
.doc-panel.active { display: block; }
.layout { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto auto; gap: 20px; }
@media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }

/* -- Cards ------------------------------------ */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; transition: border-color .15s; }
.card-head { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .7px; color: var(--text-dim); display: flex; align-items: center; justify-content: space-between; }
.card-body { padding: 16px; }
.count { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); }

/* Span full width */
.full { grid-column: 1 / -1; }

/* -- Source Text Card ------------------------- */
.source-text {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 15px;
  line-height: 1.85;
  color: var(--text);
}
.timex-span {
  cursor: pointer;
  border-radius: 3px;
  padding: 1px 4px;
  margin: 0 1px;
  transition: all .15s;
  position: relative;
}
.timex-span[data-role="START"]    { background: var(--start-dim); border-bottom: 2px solid var(--start); }
.timex-span[data-role="END"]      { background: var(--end-dim); border-bottom: 2px solid var(--end); }
.timex-span[data-role="DURATION"] { background: var(--dur-dim); border-bottom: 2px solid var(--dur); }
.timex-span[data-role="CONTAINS"] { background: var(--contains-dim); border-bottom: 2px solid var(--contains); }
.timex-span[data-role="UNKNOWN"]  { background: var(--unknown-dim); border-bottom: 2px solid var(--unknown); }

.timex-span.highlighted {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  box-shadow: 0 0 12px var(--accent-dim);
}
.timex-span:hover { filter: brightness(1.3); }

/* Role legend below source */
.legend { display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-dim); }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

/* -- Graph ------------------------------------ */
.graph-wrap { height: 360px; position: relative; background: radial-gradient(ellipse at 50% 50%, #6366f108 0%, transparent 70%); }
.graph-wrap svg { width: 100%; height: 100%; }
.node-g { cursor: pointer; }
.node-g:hover .node-ring { opacity: 1; }
.node-g.highlighted .node-ring { opacity: 1; stroke-width: 3; }
.node-ring { opacity: 0; transition: opacity .15s; }

/* -- Facts Table ------------------------------ */
.fact-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px; margin: 0 -10px;
  border-radius: 6px; cursor: pointer; transition: background .1s;
  font-size: 13px;
}
.fact-row:hover { background: #ffffff06; }
.fact-row.highlighted { background: var(--accent-dim); }
.fid { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-dim); min-width: 26px; }
.ftag { font-size: 10px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; padding: 2px 7px; border-radius: 3px; min-width: 68px; text-align: center; }
.ftag-START    { background: var(--start-bg); color: var(--start); }
.ftag-END      { background: var(--end-bg); color: var(--end); }
.ftag-DURATION { background: var(--dur-bg); color: var(--dur); }
.ftag-CONTAINS { background: var(--contains-bg); color: var(--contains); }
.ftag-UNKNOWN  { background: #1a1a1a; color: var(--unknown); }
.fval { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--text-bright); }
.fsig { margin-left: auto; font-size: 11px; color: var(--text-dim); font-style: italic; white-space: nowrap; }

/* -- Dependencies ----------------------------- */
.dep-row {
  display: flex; align-items: center; gap: 7px;
  padding: 7px 0; border-bottom: 1px solid #ffffff06;
  font-size: 12px; font-family: 'IBM Plex Mono', monospace;
}
.dep-row:last-child { border-bottom: none; }
.darr { color: var(--accent); }
.dtype { font-size: 9px; padding: 2px 5px; border-radius: 3px; background: #1e1b3a; color: #818cf8; }
.dcheck { color: var(--start); font-size: 11px; }
.dexpr { margin-left: auto; color: var(--text-dim); font-size: 10px; max-width: 240px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* -- Scenarios -------------------------------- */
.scen { padding: 12px 0; border-bottom: 1px solid #ffffff06; }
.scen:last-child { border-bottom: none; }
.scen-edit { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #fbbf24; }
.scen-casc { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #818cf8; padding-left: 14px; margin-top: 2px; }
.scen-meta { font-size: 10px; color: var(--text-dim); margin-top: 4px; padding-left: 14px; }
.empty { color: var(--text-dim); font-size: 12px; padding: 12px 0; }
</style>
</head>
<body>

<header>
  <h1>TDG -- Linked Visualizer</h1>
  <p id="sub"></p>
</header>
<div class="tabs" id="tabs"></div>
<div id="panels"></div>

<script>
const DATA = %DATA%;

const tabsEl = document.getElementById('tabs');
const panelsEl = document.getElementById('panels');
document.getElementById('sub').textContent =
  `${DATA.length} docs · ${DATA.reduce((a,d)=>a+d.facts.length,0)} facts · ${DATA.reduce((a,d)=>a+d.dependencies.length,0)} deps`;

// -- Build tabs + panels -----------------------
DATA.forEach((doc, i) => {
  const tab = document.createElement('div');
  tab.className = 'tab' + (i === 0 ? ' active' : '');
  tab.dataset.i = i;
  const dt = doc.document_type || 'unknown';
  tab.innerHTML = `${doc.document_id}<span class="badge badge-${dt}">${dt}</span>`;
  tab.onclick = () => activate(i);
  tabsEl.appendChild(tab);

  const p = document.createElement('div');
  p.className = 'doc-panel' + (i === 0 ? ' active' : '');
  p.id = 'p' + i;
  p.innerHTML = panel(doc, i);
  panelsEl.appendChild(p);
});

function activate(i) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', +t.dataset.i === i));
  document.querySelectorAll('.doc-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('p' + i).classList.add('active');
  drawGraph(i);
}

// -- Annotated source text ---------------------
function annotateText(doc) {
  const text = doc.source_text || '';
  if (!text) return '<span class="empty">No source text available</span>';

  // Build spans sorted by start_char descending (so we can insert from end)
  const marks = [];
  doc.facts.forEach(f => {
    // Find the raw_text in source to get offsets
    const raw = f.raw_text || f.value || '';
    if (!raw) return;
    const idx = text.indexOf(raw);
    if (idx === -1) return;
    marks.push({ start: idx, end: idx + raw.length, id: f.id, role: f.role, raw });
  });
  marks.sort((a, b) => a.start - b.start);

  // Merge overlapping spans, build HTML
  let result = '';
  let cursor = 0;
  marks.forEach(m => {
    if (m.start < cursor) return; // skip overlapping
    result += escHtml(text.slice(cursor, m.start));
    result += `<span class="timex-span" data-fid="${m.id}" data-role="${m.role}" title="${m.id}: ${m.role}">${escHtml(m.raw)}</span>`;
    cursor = m.end;
  });
  result += escHtml(text.slice(cursor));
  return result;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// -- Panel builder -----------------------------
function panel(doc, idx) {
  let h = '<div class="layout">';

  // Source text (full width)
  h += `<div class="card full"><div class="card-head">Source Text</div>
    <div class="card-body"><div class="source-text" id="src-${idx}">${annotateText(doc)}</div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:var(--start)"></div>START</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--end)"></div>END</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--dur)"></div>DURATION</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--contains)"></div>CONTAINS</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--unknown)"></div>UNKNOWN</div>
    </div></div></div>`;

  // Graph (full width)
  h += `<div class="card full"><div class="card-head">Dependency Graph<span class="count">click nodes to link</span></div>
    <div class="card-body"><div class="graph-wrap" id="gw-${idx}"></div></div></div>`;

  // Facts
  h += `<div class="card"><div class="card-head">Facts<span class="count">${doc.facts.length}</span></div><div class="card-body">`;
  doc.facts.forEach(f => {
    const sig = [f.signal_verb, f.signal_prep].filter(Boolean).join(', ') || '';
    h += `<div class="fact-row" data-fid="${f.id}" data-panel="${idx}" onclick="selectFact('${f.id}',${idx})">
      <span class="fid">${f.id}</span>
      <span class="ftag ftag-${f.role}">${f.role}</span>
      <span class="fval">${f.value || f.raw_text || '--'}</span>
      ${sig ? `<span class="fsig">${sig}</span>` : ''}
    </div>`;
  });
  h += '</div></div>';

  // Deps
  h += `<div class="card"><div class="card-head">Dependencies<span class="count">${doc.dependencies.length}</span></div><div class="card-body">`;
  if (!doc.dependencies.length) h += '<div class="empty">No dependencies</div>';
  doc.dependencies.forEach(d => {
    h += `<div class="dep-row">
      <span>${d.from_id}</span><span class="darr">-></span><span>${d.to_id}</span>
      <span class="dtype">${d.constraint_type}</span>
      ${d.verified ? '<span class="dcheck">✓</span>' : ''}
      <span class="dexpr" title="${d.constraint_expr}">${d.constraint_expr}</span>
    </div>`;
  });
  h += '</div></div>';

  // Scenarios
  h += `<div class="card full"><div class="card-head">Edit Scenarios<span class="count">${doc.edit_scenarios.length}</span></div><div class="card-body">`;
  if (!doc.edit_scenarios.length) h += '<div class="empty">No cascading scenarios</div>';
  doc.edit_scenarios.slice(0, 3).forEach(s => {
    h += `<div class="scen"><div class="scen-edit">✎ ${s.edit.role} ${s.edit.target_id}: ${s.edit.old_value} -> ${s.edit.new_value}</div>`;
    s.expected_cascades.forEach(c => {
      const n = c.note ? ` · ${c.note}` : '';
      h += `<div class="scen-casc">↳ ${c.role} ${c.fact_id}: ${c.old_value} -> ${c.new_value}${n}</div>`;
    });
    h += `<div class="scen-meta">depth ${s.ripple_depth} · breadth ${s.ripple_breadth}</div></div>`;
  });
  if (doc.edit_scenarios.length > 3) h += `<div class="empty">+ ${doc.edit_scenarios.length - 3} more</div>`;
  h += '</div></div></div>';
  return h;
}

// -- Selection / highlighting ------------------
let currentSelection = { fid: null, panel: null };

function selectFact(fid, panelIdx) {
  clearHighlights(panelIdx);
  if (currentSelection.fid === fid && currentSelection.panel === panelIdx) {
    currentSelection = { fid: null, panel: null };
    return;
  }
  currentSelection = { fid: fid, panel: panelIdx };

  // Highlight fact row
  const row = document.querySelector(`#p${panelIdx} .fact-row[data-fid="${fid}"]`);
  if (row) row.classList.add('highlighted');

  // Highlight source text span
  const spans = document.querySelectorAll(`#src-${panelIdx} .timex-span[data-fid="${fid}"]`);
  spans.forEach(s => {
    s.classList.add('highlighted');
    s.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  // Highlight graph node
  const node = document.querySelector(`#gw-${panelIdx} .node-g[data-fid="${fid}"]`);
  if (node) node.classList.add('highlighted');
}

function clearHighlights(panelIdx) {
  document.querySelectorAll(`#p${panelIdx} .highlighted`).forEach(el => el.classList.remove('highlighted'));
}

// Source text click handler (delegated)
document.addEventListener('click', e => {
  const span = e.target.closest('.timex-span');
  if (span) {
    const fid = span.dataset.fid;
    const panel = span.closest('.doc-panel');
    if (panel) {
      const idx = parseInt(panel.id.replace('p', ''));
      selectFact(fid, idx);
    }
  }
});

// -- SVG Graph ---------------------------------
const RC = { START: '#34d399', END: '#f87171', DURATION: '#818cf8', CONTAINS: '#fbbf24', UNKNOWN: '#6b7280' };
const CC = { additive: '#6366f1', ordering: '#f59e0b', interval: '#10b981', periodic: '#c084fc' };

function drawGraph(idx) {
  const doc = DATA[idx];
  const el = document.getElementById('gw-' + idx);
  if (!el) return;
  const W = el.clientWidth || 700, H = el.clientHeight || 360;
  const facts = doc.facts, deps = doc.dependencies;
  if (!facts.length) { el.innerHTML = '<svg></svg>'; return; }

  // Layout
  const sp = Math.min((W - 160) / Math.max(facts.length - 1, 1), 160);
  const sx = (W - sp * (facts.length - 1)) / 2;
  const pos = {};
  facts.forEach((f, i) => {
    let y = 0;
    if (f.role === 'START') y = -45;
    else if (f.role === 'END') y = 45;
    else if (f.role === 'DURATION') y = 0;
    else y = 65;
    pos[f.id] = { x: sx + i * sp, y: H / 2 + y };
  });

  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    <defs>
      <marker id="ah${idx}" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0,8 3,0 6" fill="${CC.additive}" opacity=".6"/>
      </marker>
      <filter id="glow${idx}"><feGaussianBlur stdDeviation="3" result="g"/>
        <feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>`;

  // Edges
  deps.forEach(d => {
    const a = pos[d.from_id], b = pos[d.to_id];
    if (!a || !b) return;
    const c = CC[d.constraint_type] || '#555';
    const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2 - Math.abs(b.x - a.x) * 0.18;
    s += `<path d="M${a.x},${a.y} Q${cx},${cy} ${b.x},${b.y}" stroke="${c}" stroke-width="2" fill="none" marker-end="url(#ah${idx})" opacity=".5"/>`;
    s += `<text x="${cx}" y="${cy - 8}" font-family="IBM Plex Mono" font-size="9" fill="#5a6173" text-anchor="middle">${d.constraint_type}${d.verified ? ' ✓' : ''}</text>`;
  });

  // Nodes
  facts.forEach(f => {
    const p = pos[f.id], c = RC[f.role] || '#6b7280';
    s += `<g class="node-g" data-fid="${f.id}" onclick="selectFact('${f.id}',${idx})">`;
    // Highlight ring (hidden by default)
    s += `<circle class="node-ring" cx="${p.x}" cy="${p.y}" r="30" fill="none" stroke="${c}" stroke-width="2" opacity="0"/>`;
    // Main node
    s += `<circle cx="${p.x}" cy="${p.y}" r="22" fill="${c}18" stroke="${c}" stroke-width="1.5"/>`;
    s += `<text x="${p.x}" y="${p.y - 1}" font-family="IBM Plex Mono" font-size="12" font-weight="600" fill="${c}" text-anchor="middle">${f.id}</text>`;
    s += `<text x="${p.x}" y="${p.y + 11}" font-family="IBM Plex Mono" font-size="8" fill="#5a6173" text-anchor="middle">${f.role}</text>`;
    // Value below
    const v = f.value || f.raw_text || '';
    const sv = v.length > 16 ? v.slice(0, 14) + '…' : v;
    s += `<text x="${p.x}" y="${p.y + 40}" font-family="IBM Plex Mono" font-size="9" fill="#5a6173" text-anchor="middle">${sv}</text>`;
    s += '</g>';
  });

  s += '</svg>';
  el.innerHTML = s;
}

// Initial render
setTimeout(() => drawGraph(0), 60);
</script>
</body>
</html>'''


def generate_linked_html(data: list[dict]) -> str:
    """Generate linked visualization HTML from TDG dict list."""
    return _TEMPLATE.replace('%DATA%', json.dumps(data))


def main():
    parser = argparse.ArgumentParser(description="Generate linked TDG visualization")
    parser.add_argument("input", nargs="?", help="Input JSON file")
    parser.add_argument("-o", "--output", default="tdg_linked.html")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tdg_pipeline.pipeline import TDGPipeline
        demos = [
            ("ww2_europe", "historical", "World War II in Europe",
             "World War II in Europe began on September 1, 1939, when Germany invaded Poland. "
             "The war lasted approximately 5 years and 8 months, ending on May 8, 1945 with "
             "Germany's unconditional surrender. The Battle of Stalingrad, which took place "
             "between August 23, 1942 and February 2, 1943, marked a crucial turning point."),
            ("service_agreement", "legal", "Service Agreement",
             "This Service Agreement takes effect on January 15, 2025 (the Effective Date). "
             "All invoices shall be due and payable within 30 days after the Effective Date. "
             "This Agreement shall commence on the Effective Date and continue for a period "
             "of 12 months, terminating on January 15, 2026. Either party may terminate "
             "upon 90 days written notice prior to the termination date."),
            ("patient_001", "medical", "Patient Treatment",
             "The patient was admitted on March 3, 2024 with acute respiratory symptoms. "
             "Treatment commenced on March 4, 2024 and continued for 14 days. "
             "The patient was discharged on March 17, 2024 with instructions to "
             "return for follow-up within 30 days."),
            ("acme_merger", "corporate", "Acme Corp Merger",
             "Acme Corp announced the merger on January 10, 2023. "
             "The regulatory review lasted 6 months, concluding on July 8, 2023. "
             "The merger was completed on August 1, 2023, and the integration "
             "period extended for 18 months until February 2025."),
            ("marie_curie", "biographical", "Marie Curie",
             "Marie Curie was born on November 7, 1867 in Warsaw. She moved to Paris "
             "in 1891 and began her studies at the Sorbonne. She was awarded the Nobel "
             "Prize in Physics on December 10, 1903. Marie Curie died on July 4, 1934, "
             "having lived for 66 years."),
        ]
        pipe = TDGPipeline()
        data = [pipe.process(t, document_id=did, document_type=dt, document_entity=ent).to_dict()
                for did, dt, ent, t in demos]
    elif args.input:
        with open(args.input) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    else:
        parser.error("Provide input JSON or use --demo")

    html = generate_linked_html(data)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Linked visualization -> {args.output} ({len(data)} documents)")


if __name__ == "__main__":
    main()
