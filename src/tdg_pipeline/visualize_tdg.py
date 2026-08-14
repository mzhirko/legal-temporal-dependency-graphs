#!/usr/bin/env python3
"""
Visualize a TDG JSON output as a proper knowledge graph.

Usage:
    python visualize_tdg.py result.json
    python visualize_tdg.py result.json -o graph.html
    python visualize_tdg.py result.json --no-open
"""

import argparse
import json
import os
import sys
import webbrowser

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TDG -- Knowledge Graph</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'DM Sans', sans-serif;
  background: #f7f7f5;
  color: #1a1a18;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

header {
  background: #fff;
  border-bottom: 1px solid #e8e6e0;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}
.doc-id { font-size: 14px; font-weight: 600; color: #1a1a18; }
.doc-meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #888; }
.type-badge {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px; padding: 3px 10px; border-radius: 20px;
  background: #e8e4f8; color: #5340b8;
}
.legend { margin-left: auto; display: flex; align-items: center; gap: 14px; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #666; }
.ls { width: 13px; height: 13px; border-radius: 3px; }

.main { flex: 1; display: flex; overflow: hidden; }

#canvas-wrap { flex: 1; position: relative; overflow: hidden; cursor: grab; }
#canvas-wrap:active { cursor: grabbing; }
canvas { position: absolute; top: 0; left: 0; }

.sidebar {
  width: 300px; background: #fff; border-left: 1px solid #e8e6e0;
  display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
}
.panel { border-bottom: 1px solid #e8e6e0; overflow-y: auto; flex-shrink: 0; }
.panel-head {
  padding: 9px 16px; font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: .7px; color: #999;
  background: #fafaf8; border-bottom: 1px solid #e8e6e0;
  display: flex; justify-content: space-between;
  position: sticky; top: 0;
}
.cnt { color: #5340b8; font-family: 'IBM Plex Mono', monospace; }

.fact-row {
  padding: 7px 16px; cursor: pointer; border-bottom: 1px solid #f0ede8;
  display: flex; gap: 8px; align-items: flex-start; transition: background .1s;
}
.fact-row:hover { background: #fafaf8; }
.fact-row.active { background: #f0edfb; }
.fid { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #bbb; padding-top: 2px; min-width: 22px; }
.fbody { flex: 1; min-width: 0; }
.fentity { font-size: 12px; font-weight: 600; color: #1a1a18; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fdesc { font-size: 11px; color: #888; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.role-pill {
  font-size: 9px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
  padding: 2px 7px; border-radius: 12px; margin-top: 2px; white-space: nowrap;
}

.dep-row {
  padding: 6px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
  border-bottom: 1px solid #f0ede8; display: flex; align-items: center; gap: 5px; color: #444;
}
.dep-type { margin-left: auto; font-size: 9px; padding: 1px 6px; border-radius: 10px; }
.dep-expr { font-size: 10px; color: #aaa; padding: 1px 16px 6px; font-family: 'IBM Plex Mono', monospace; }

#detail { flex: 1; overflow-y: auto; padding: 16px; }
#detail h3 { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #999; margin-bottom: 10px; }
.kv { display: flex; gap: 8px; margin-bottom: 6px; font-size: 12px; }
.k { color: #aaa; min-width: 72px; }
.v { color: #1a1a18; font-family: 'IBM Plex Mono', monospace; word-break: break-all; }
.sentence-box {
  margin-top: 12px; padding: 10px 14px; background: #f7f7f5;
  border-left: 3px solid #c7bff0; border-radius: 0 6px 6px 0;
  font-size: 12px; color: #444; line-height: 1.7; font-style: italic;
}
.empty { font-size: 12px; color: #bbb; padding: 8px 0; }

.zoom-controls {
  position: absolute; bottom: 16px; left: 16px;
  display: flex; flex-direction: column; gap: 4px; z-index: 10;
}
.zoom-btn {
  width: 32px; height: 32px; background: #fff; border: 1px solid #e0ddd8;
  border-radius: 6px; font-size: 16px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: #444; transition: background .1s; font-family: inherit;
}
.zoom-btn:hover { background: #f0ede8; }

.edge-legend {
  position: absolute; top: 12px; left: 12px; background: #fff;
  border: 1px solid #e0ddd8; border-radius: 8px; padding: 10px 14px;
  font-size: 11px; color: #666; z-index: 10;
}
.edge-legend-title {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .6px; color: #aaa; margin-bottom: 7px;
}
.edge-item { display: flex; align-items: center; gap: 7px; margin-bottom: 4px; }
.eline { width: 28px; height: 2px; border-radius: 1px; }
</style>
</head>
<body>

<header>
  <span class="doc-id" id="doc-id">--</span>
  <span class="doc-meta" id="doc-meta">--</span>
  <span class="type-badge" id="doc-type">--</span>
  <div class="legend">
    <div class="legend-item"><div class="ls" style="background:#dcf5ea;border:2px solid #34c47c"></div>START</div>
    <div class="legend-item"><div class="ls" style="background:#fde8e8;border:2px solid #e85555;border-radius:50%"></div>END</div>
    <div class="legend-item"><div class="ls" style="background:#ede8fb;border:2px solid #7c6ad4;transform:rotate(45deg)"></div>DURATION</div>
    <div class="legend-item"><div class="ls" style="background:#fef3dc;border:2px solid #d4940a"></div>CONTAINS</div>
    <div class="legend-item"><div class="ls" style="background:#efefef;border:2px solid #aaa"></div>UNKNOWN</div>
  </div>
</header>

<div class="main">
  <div id="canvas-wrap">
    <canvas id="c"></canvas>
    <div class="edge-legend">
      <div class="edge-legend-title">Edges</div>
      <div class="edge-item"><div class="eline" style="background:#5340b8"></div>additive</div>
      <div class="edge-item"><div class="eline" style="background:#d4940a"></div>ordering</div>
      <div class="edge-item"><div class="eline" style="background:#1a9e75"></div>interval</div>
    </div>
    <div class="zoom-controls">
      <button class="zoom-btn" onclick="zoom(1.2)">+</button>
      <button class="zoom-btn" onclick="zoom(0.83)">−</button>
      <button class="zoom-btn" onclick="resetView()" title="Reset" style="font-size:11px">⊡</button>
    </div>
  </div>

  <div class="sidebar">
    <div class="panel" style="max-height:44%">
      <div class="panel-head">Events <span class="cnt" id="fact-cnt">0</span></div>
      <div id="fact-list"></div>
    </div>
    <div class="panel" style="max-height:22%">
      <div class="panel-head">Dependencies <span class="cnt" id="dep-cnt">0</span></div>
      <div id="dep-list"></div>
    </div>
    <div id="detail"><div class="empty">Click a node to inspect</div></div>
  </div>
</div>

<script>
const DATA = %DATA%;

const RC = {
  START:    { fill:'#dcf5ea', stroke:'#34c47c', text:'#0d6e3f' },
  END:      { fill:'#fde8e8', stroke:'#e85555', text:'#8b1a1a' },
  DURATION: { fill:'#ede8fb', stroke:'#7c6ad4', text:'#3d2a8a' },
  CONTAINS: { fill:'#fef3dc', stroke:'#d4940a', text:'#7a5200' },
  UNKNOWN:  { fill:'#f0f0ee', stroke:'#aaaaaa', text:'#555555' },
};
const DC = { additive:'#5340b8', ordering:'#d4940a', interval:'#1a9e75', periodic:'#c084fc' };
const NW = 130, NH = 52;

document.getElementById('doc-id').textContent = DATA.document_id;
document.getElementById('doc-type').textContent = DATA.document_type;
document.getElementById('doc-meta').textContent =
  `${DATA.facts.length} events · ${DATA.dependencies.length} deps · ${DATA.edit_scenarios.length} scenarios`;
document.getElementById('fact-cnt').textContent = DATA.facts.length;
document.getElementById('dep-cnt').textContent = DATA.dependencies.length;

// Sidebar
const factList = document.getElementById('fact-list');
DATA.facts.forEach(f => {
  const c = RC[f.role] || RC.UNKNOWN;
  const div = document.createElement('div');
  div.className = 'fact-row'; div.dataset.id = f.id;
  div.innerHTML = `
    <span class="fid">${f.id}</span>
    <div class="fbody">
      <div class="fentity">${f.entity}</div>
      <div class="fdesc">${f.description || f.raw_text || f.value || ''}</div>
    </div>
    <span class="role-pill" style="background:${c.fill};color:${c.text};border:1px solid ${c.stroke}55">${f.role}</span>`;
  div.onclick = () => selectFact(f.id);
  factList.appendChild(div);
});

const depList = document.getElementById('dep-list');
if (!DATA.dependencies.length)
  depList.innerHTML = '<div class="empty" style="padding:10px 16px">None found</div>';
DATA.dependencies.forEach(d => {
  const c = DC[d.constraint_type] || '#888';
  depList.innerHTML += `
    <div class="dep-row">
      <b>${d.from_id}</b><span style="color:${c}">-></span><b>${d.to_id}</b>
      <span class="dep-type" style="background:${c}18;color:${c};border:1px solid ${c}40">${d.constraint_type}</span>
    </div>
    <div class="dep-expr">${d.constraint_expr}</div>`;
});

// Canvas
const wrap = document.getElementById('canvas-wrap');
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, nodes = [], edges = [], scale = 1, offsetX = 0, offsetY = 0;
let selectedId = null, draggingNode = null, dragOffset = {x:0,y:0}, panStart = null;

function resize() {
  W = wrap.clientWidth; H = wrap.clientHeight;
  canvas.width = W; canvas.height = H; draw();
}
window.addEventListener('resize', resize);

function initGraph() {
  const n = DATA.facts.length; if (!n) return;
  const colX = { START:160, DURATION:400, END:640, CONTAINS:400, UNKNOWN:280 };
  const counters = {};
  nodes = DATA.facts.map(f => {
    counters[f.role] = (counters[f.role] || 0) + 1;
    return { id:f.id, x:colX[f.role]||300, y:100+(counters[f.role]-1)*130, fact:f, r:RC[f.role]||RC.UNKNOWN };
  });
  edges = DATA.dependencies.map(d => ({
    from: nodes.find(n=>n.id===d.from_id),
    to:   nodes.find(n=>n.id===d.to_id),
    dep:d, color:DC[d.constraint_type]||'#888'
  })).filter(e=>e.from&&e.to);
  resetView();
}

function resetView() {
  if (!nodes.length) return;
  const xs=nodes.map(n=>n.x), ys=nodes.map(n=>n.y);
  const minX=Math.min(...xs)-100, maxX=Math.max(...xs)+100;
  const minY=Math.min(...ys)-80,  maxY=Math.max(...ys)+80;
  const gw=maxX-minX, gh=maxY-minY;
  scale = Math.min(W/gw, H/gh, 1.3)*0.88;
  offsetX = (W-gw*scale)/2 - minX*scale;
  offsetY = (H-gh*scale)/2 - minY*scale;
  draw();
}

function toScreen(x,y){ return {x:x*scale+offsetX, y:y*scale+offsetY}; }
function fromScreen(sx,sy){ return {x:(sx-offsetX)/scale, y:(sy-offsetY)/scale}; }

function rr(ctx,x,y,w,h,r){
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y);
  ctx.quadraticCurveTo(x+w,y,x+w,y+r); ctx.lineTo(x+w,y+h-r);
  ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h); ctx.lineTo(x+r,y+h);
  ctx.quadraticCurveTo(x,y+h,x,y+h-r); ctx.lineTo(x,y+r);
  ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}

function nodeClip(node, dx, dy) {
  const r=node.fact.role, hw=NW/2+2, hh=NH/2+2;
  const dist=Math.sqrt(dx*dx+dy*dy)||1, nx=dx/dist, ny=dy/dist;
  if(r==='END') return hw;
  if(r==='DURATION'){
    return Math.min(hw/Math.abs(nx||.001), hh/Math.abs(ny||.001));
  }
  return Math.min(hw/Math.abs(nx||.001), hh/Math.abs(ny||.001));
}

function draw() {
  ctx.clearRect(0,0,W,H);
  ctx.save();
  ctx.translate(offsetX,offsetY); ctx.scale(scale,scale);

  // Edges
  edges.forEach(({from:a,to:b,dep,color})=>{
    const dx=b.x-a.x, dy=b.y-a.y;
    const ta=nodeClip(a,dx,dy), tb=nodeClip(b,-dx,-dy);
    const dist=Math.sqrt(dx*dx+dy*dy)||1;
    const x1=a.x+(dx/dist)*ta, y1=a.y+(dy/dist)*ta;
    const x2=b.x-(dx/dist)*tb, y2=b.y-(dy/dist)*tb;
    const mx=(x1+x2)/2-(y2-y1)*0.18, my=(y1+y2)/2+(x2-x1)*0.18;

    ctx.save();
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.quadraticCurveTo(mx,my,x2,y2);
    ctx.strokeStyle=color+'cc'; ctx.lineWidth=1.8; ctx.stroke();

    // Arrow
    const t=0.88;
    const qx=(1-t)*(1-t)*x1+2*(1-t)*t*mx+t*t*x2;
    const qy=(1-t)*(1-t)*y1+2*(1-t)*t*my+t*t*y2;
    const ang=Math.atan2(y2-qy,x2-qx);
    ctx.beginPath();
    ctx.moveTo(x2,y2);
    ctx.lineTo(x2-Math.cos(ang-.42)*11, y2-Math.sin(ang-.42)*11);
    ctx.lineTo(x2-Math.cos(ang+.42)*11, y2-Math.sin(ang+.42)*11);
    ctx.closePath(); ctx.fillStyle=color; ctx.fill();

    // Label pill
    const lx=.25*x1+.5*mx+.25*x2, ly=.25*y1+.5*my+.25*y2;
    const label=dep.constraint_type;
    ctx.font='500 10px DM Sans,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    const lw=ctx.measureText(label).width+12;
    ctx.beginPath(); rr(ctx,lx-lw/2,ly-9,lw,18,5);
    ctx.fillStyle='#fff'; ctx.fill();
    ctx.strokeStyle=color+'55'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle=color; ctx.fillText(label,lx,ly+1);
    ctx.restore();
  });

  // Nodes
  nodes.forEach(n=>{
    const {x,y,fact:f,r:c} = n;
    const hw=NW/2, hh=NH/2;
    const sel=n.id===selectedId;

    ctx.save();
    if(sel){
      ctx.beginPath();
      if(f.role==='END') ctx.arc(x,y,hw+9,0,Math.PI*2);
      else if(f.role==='DURATION'){ ctx.moveTo(x,y-hh-10); ctx.lineTo(x+hw+10,y); ctx.lineTo(x,y+hh+10); ctx.lineTo(x-hw-10,y); ctx.closePath(); }
      else { ctx.beginPath(); rr(ctx,x-hw-7,y-hh-7,NW+14,NH+14,11); }
      ctx.strokeStyle=c.stroke+'55'; ctx.lineWidth=4; ctx.stroke();
    }

    ctx.beginPath();
    if(f.role==='END') ctx.arc(x,y,hw,0,Math.PI*2);
    else if(f.role==='DURATION'){ ctx.moveTo(x,y-hh); ctx.lineTo(x+hw,y); ctx.lineTo(x,y+hh); ctx.lineTo(x-hw,y); ctx.closePath(); }
    else rr(ctx,x-hw,y-hh,NW,NH,8);
    ctx.fillStyle=c.fill; ctx.fill();
    ctx.strokeStyle=sel?c.stroke:c.stroke+'aa'; ctx.lineWidth=sel?2:1.5; ctx.stroke();

    // Text
    ctx.fillStyle=c.text; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.font='600 11px IBM Plex Mono,monospace'; ctx.fillText(n.id,x,y-8);
    const val=(f.value||f.raw_text||'').slice(0,16);
    ctx.font='400 10px DM Sans,sans-serif'; ctx.fillStyle=c.text+'bb'; ctx.fillText(val,x,y+8);
    ctx.restore();
  });

  ctx.restore();
}

function selectFact(id){
  selectedId=id;
  document.querySelectorAll('.fact-row').forEach(r=>r.classList.toggle('active',r.dataset.id===id));
  const f=DATA.facts.find(f=>f.id===id);
  const detail=document.getElementById('detail');
  if(!f){detail.innerHTML='<div class="empty">Not found</div>';draw();return;}
  const c=RC[f.role]||RC.UNKNOWN;
  detail.innerHTML=`
    <h3 style="color:${c.stroke}">${f.role} -- ${f.id}</h3>
    <div class="kv"><span class="k">Entity</span><span class="v">${f.entity}</span></div>
    ${f.description?`<div class="kv"><span class="k">Event</span><span class="v">${f.description}</span></div>`:''}
    <div class="kv"><span class="k">Value</span><span class="v">${f.value||'--'}</span></div>
    ${f.date_parsed?`<div class="kv"><span class="k">Date</span><span class="v">${f.date_parsed}</span></div>`:''}
    ${f.duration_days!=null?`<div class="kv"><span class="k">Days</span><span class="v">${f.duration_days}</span></div>`:''}
    <div class="kv"><span class="k">Confidence</span><span class="v">${(f.confidence*100).toFixed(0)}%</span></div>
    ${f.sentence?`<div class="sentence-box">"${f.sentence.slice(0,300)}"</div>`:''}`;
  draw();
}

function hitTest(sx,sy){
  const {x,y}=fromScreen(sx,sy), hw=NW/2+6, hh=NH/2+6;
  return nodes.find(n=>{
    if(n.fact.role==='END') return Math.hypot(x-n.x,y-n.y)<hw;
    return Math.abs(x-n.x)<hw&&Math.abs(y-n.y)<hh;
  });
}

canvas.addEventListener('mousedown',e=>{
  const r=canvas.getBoundingClientRect(), sx=e.clientX-r.left, sy=e.clientY-r.top;
  const hit=hitTest(sx,sy);
  if(hit){draggingNode=hit;const g=fromScreen(sx,sy);dragOffset={x:g.x-hit.x,y:g.y-hit.y};}
  else panStart={x:e.clientX-offsetX,y:e.clientY-offsetY};
});
canvas.addEventListener('mousemove',e=>{
  const r=canvas.getBoundingClientRect(), sx=e.clientX-r.left, sy=e.clientY-r.top;
  if(draggingNode){const g=fromScreen(sx,sy);draggingNode.x=g.x-dragOffset.x;draggingNode.y=g.y-dragOffset.y;draw();}
  else if(panStart){offsetX=e.clientX-panStart.x;offsetY=e.clientY-panStart.y;draw();}
  canvas.style.cursor=hitTest(sx,sy)?'pointer':(panStart?'grabbing':'grab');
});
canvas.addEventListener('mouseup',e=>{
  if(draggingNode){
    const r=canvas.getBoundingClientRect();
    const sx=e.clientX-r.left, sy=e.clientY-r.top;
    const s=toScreen(draggingNode.x,draggingNode.y);
    if(Math.hypot(sx-s.x,sy-s.y)<6) selectFact(draggingNode.id);
    draggingNode=null;
  }
  panStart=null;
});
canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect();
  if(!hitTest(e.clientX-r.left,e.clientY-r.top)){
    selectedId=null;
    document.querySelectorAll('.fact-row').forEach(r=>r.classList.remove('active'));
    document.getElementById('detail').innerHTML='<div class="empty">Click a node to inspect</div>';
    draw();
  }
});
canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const d=e.deltaY>0?.85:1.18;
  offsetX=mx-(mx-offsetX)*d; offsetY=my-(my-offsetY)*d; scale*=d; draw();
},{passive:false});

function zoom(f){offsetX=W/2-(W/2-offsetX)*f;offsetY=H/2-(H/2-offsetY)*f;scale*=f;draw();}

resize();
initGraph();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize TDG JSON as a knowledge graph")
    parser.add_argument("input", help="TDG JSON file from demo_llm.py")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file (default: <input>.html)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    for fact in data.get("facts", []):
        if "description" not in fact:
            fact["description"] = fact.get("raw_text", "")

    output = args.output or os.path.splitext(args.input)[0] + ".html"
    html = HTML.replace("%DATA%", json.dumps(data))

    with open(output, "w") as f:
        f.write(html)

    print(f"Saved: {output}")
    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(output)}")


if __name__ == "__main__":
    main()
