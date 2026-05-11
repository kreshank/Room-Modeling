/**
 * Static viewer: scene_graph.json + optional predict JSON + optional extras.
 * Extension hook: applyExtrasLayer — see bottom.
 */

const RESERVED_EDGE_KEYS = new Set(["source", "target", "type"]);

const STATUS_RANK = { violated: 0, weak: 1, good: 2 };

const NODE_FILL = {
  object: "rgba(80, 120, 200, 0.35)",
  wall: "rgba(100, 100, 100, 0.4)",
  door: "rgba(200, 140, 60, 0.45)",
  window: "rgba(120, 200, 220, 0.4)",
  room: "rgba(180, 180, 220, 0.2)",
  zone: "rgba(160, 100, 200, 0.25)",
  unknown: "rgba(150, 150, 150, 0.3)",
};

const WORST_STATUS_COLOR = {
  violated: "rgba(220, 60, 60, 0.55)",
  weak: "rgba(230, 180, 40, 0.5)",
  good: "rgba(60, 180, 90, 0.4)",
};

let sceneGraph = null;
let predict = null;
let extras = null;

/** @type {Map<string, {x:number,y:number}>} */
let anchors = new Map();
let bounds = { minX: 0, maxX: 1, minY: 0, maxY: 1 };
let offsetX = 0;
let offsetY = 0;

let hoverId = null;
let selectedId = null;

/** edge type string -> enabled */
const edgeTypeEnabled = new Map();

function $(id) {
  return document.getElementById(id);
}

function anchorForNode(node) {
  const g = node.geometry || {};
  if (Array.isArray(g.centroid_xy) && g.centroid_xy.length >= 2) {
    return { x: g.centroid_xy[0], y: g.centroid_xy[1] };
  }
  if (Array.isArray(g.corners_xy) && g.corners_xy.length) {
    let sx = 0;
    let sy = 0;
    for (const c of g.corners_xy) {
      if (Array.isArray(c) && c.length >= 2) {
        sx += c[0];
        sy += c[1];
      }
    }
    const n = g.corners_xy.length;
    return { x: sx / n, y: sy / n };
  }
  if (typeof g.cx === "number" && typeof g.cy === "number") {
    return { x: g.cx, y: g.cy };
  }
  return null;
}

function rebuildAnchors() {
  anchors = new Map();
  if (!sceneGraph || !Array.isArray(sceneGraph.nodes)) return;
  for (const node of sceneGraph.nodes) {
    const p = anchorForNode(node);
    if (p) anchors.set(node.id, p);
  }
}

function extendBoundsFromXY(x, y, bb) {
  bb.minX = Math.min(bb.minX, x);
  bb.maxX = Math.max(bb.maxX, x);
  bb.minY = Math.min(bb.minY, y);
  bb.maxY = Math.max(bb.maxY, y);
}

function extendBoundsFromRing(ring, bb) {
  if (!Array.isArray(ring)) return;
  for (const pt of ring) {
    if (Array.isArray(pt) && pt.length >= 2) extendBoundsFromXY(pt[0], pt[1], bb);
  }
}

function computeBounds() {
  const bb = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  const room = sceneGraph && sceneGraph.room;
  if (room && Array.isArray(room.polygon_xy)) {
    for (const ring of room.polygon_xy) extendBoundsFromRing(ring, bb);
  }
  if (room && Array.isArray(room.walkable_polygon_xy)) {
    for (const ring of room.walkable_polygon_xy) extendBoundsFromRing(ring, bb);
  }
  for (const p of anchors.values()) extendBoundsFromXY(p.x, p.y, bb);
  if (!Number.isFinite(bb.minX)) {
    bb.minX = 0;
    bb.maxX = 1;
    bb.minY = 0;
    bb.maxY = 1;
  }
  const pad = 0.06 * Math.max(bb.maxX - bb.minX, bb.maxY - bb.minY, 1);
  bounds = {
    minX: bb.minX - pad,
    maxX: bb.maxX + pad,
    minY: bb.minY - pad,
    maxY: bb.maxY + pad,
  };
}

function worldToScreen(x, y, w, h) {
  const rw = bounds.maxX - bounds.minX;
  const rh = bounds.maxY - bounds.minY;
  const sx = (x - bounds.minX) / rw;
  const sy = (y - bounds.minY) / rh;
  return {
    x: offsetX + sx * w,
    y: offsetY + (1 - sy) * h,
  };
}

function updateTransform() {
  const cv = $("cv");
  const rect = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const wCss = Math.max(320, Math.floor(rect.width));
  const hCss = Math.max(240, Math.floor(rect.height));
  cv.width = Math.floor(wCss * dpr);
  cv.height = Math.floor(hCss * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = wCss;
  const h = hCss;
  offsetX = 8;
  offsetY = 8;
  const innerW = w - 16;
  const innerH = h - 16;
  return { ctx, w, h, innerW, innerH };
}

function collectNumericEdgeKeys() {
  const keys = new Set();
  if (!sceneGraph || !Array.isArray(sceneGraph.edges)) return keys;
  for (const e of sceneGraph.edges) {
    for (const k of Object.keys(e)) {
      if (RESERVED_EDGE_KEYS.has(k)) continue;
      const v = e[k];
      if (typeof v === "number" && Number.isFinite(v)) keys.add(k);
    }
  }
  return keys;
}

function populateWeightSelect() {
  const sel = $("sel-weight");
  const keys = [...collectNumericEdgeKeys()].sort();
  sel.innerHTML = '<option value="">(uniform)</option>';
  for (const k of keys) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k;
    sel.appendChild(o);
  }
}

function collectEdgeTypes() {
  const types = new Set();
  if (!sceneGraph || !Array.isArray(sceneGraph.edges)) return types;
  for (const e of sceneGraph.edges) {
    if (e && e.type) types.add(String(e.type));
  }
  return types;
}

function rebuildEdgeTypeFilters() {
  edgeTypeEnabled.clear();
  const types = [...collectEdgeTypes()].sort();
  const showPrinciple = $("chk-principle").checked;
  const box = $("edge-type-boxes");
  box.innerHTML = "";
  for (const t of types) {
    const on = showPrinciple || !t.startsWith("principle_");
    edgeTypeEnabled.set(t, on);
    const id = `et-${t.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = on;
    cb.dataset.edgeType = t;
    cb.addEventListener("change", () => {
      edgeTypeEnabled.set(t, cb.checked);
      draw();
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(` ${t}`));
    box.appendChild(lab);
  }
  if (!types.length) box.textContent = "No edges.";
}

function predictionsByTarget() {
  /** @type {Map<string, Array<{principle:string,status:string,score:number}>>} */
  const m = new Map();
  if (!predict || !Array.isArray(predict.principle_predictions)) return m;
  for (const row of predict.principle_predictions) {
    const tid = row.target;
    if (!m.has(tid)) m.set(tid, []);
    m.get(tid).push({
      principle: row.principle,
      status: row.status,
      score: row.score,
    });
  }
  return m;
}

function worstStatusForTarget(predMap, nodeId) {
  const rows = predMap.get(nodeId);
  if (!rows || !rows.length) return null;
  let worst = "good";
  let wr = STATUS_RANK.good;
  for (const r of rows) {
    const rk = STATUS_RANK[r.status] ?? 3;
    if (rk < wr) {
      wr = rk;
      worst = r.status;
    }
  }
  return worst;
}

function edgeWeightStats(key) {
  if (!key || !sceneGraph || !Array.isArray(sceneGraph.edges)) return { min: 0, max: 1 };
  let minV = Infinity;
  let maxV = -Infinity;
  for (const e of sceneGraph.edges) {
    if (!edgeTypeEnabled.get(String(e.type))) continue;
    if (String(e.type).startsWith("principle_") && !$("chk-principle").checked) continue;
    const v = e[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    minV = Math.min(minV, v);
    maxV = Math.max(maxV, v);
  }
  if (!Number.isFinite(minV)) return { min: 0, max: 1 };
  if (maxV - minV < 1e-9) return { min: minV, max: minV + 1e-9 };
  return { min: minV, max: maxV };
}

function strokeForEdge(e, weightKey) {
  if (!weightKey) return { width: 0.6, alpha: 0.35 };
  const st = edgeWeightStats(weightKey);
  const v = e[weightKey];
  if (typeof v !== "number" || !Number.isFinite(v)) return { width: 0.5, alpha: 0.2 };
  const t = (v - st.min) / (st.max - st.min);
  return { width: 0.5 + 2.5 * t, alpha: 0.2 + 0.55 * t };
}

function visibleEdges() {
  if (!sceneGraph || !Array.isArray(sceneGraph.edges)) return [];
  const showPrinciple = $("chk-principle").checked;
  const merge = $("chk-merge").checked;
  const list = sceneGraph.edges.filter((e) => {
    const typ = String(e.type || "");
    if (!showPrinciple && typ.startsWith("principle_")) return false;
    return edgeTypeEnabled.get(typ) !== false;
  });
  if (!merge) return list.map((e, i) => ({ ...e, _i: i }));
  const seen = new Set();
  const out = [];
  for (let i = 0; i < list.length; i++) {
    const e = list[i];
    const a = e.source;
    const b = e.target;
    const typ = String(e.type || "");
    const k = a < b ? `${a}|${b}|${typ}` : `${b}|${a}|${typ}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ ...e, _i: i });
  }
  return out;
}

function drawPolygonRings(ctx, rings, w, h, fillStyle, strokeStyle, lineWidth) {
  if (!Array.isArray(rings)) return;
  for (const ring of rings) {
    if (!Array.isArray(ring) || ring.length < 2) continue;
    ctx.beginPath();
    const p0 = worldToScreen(ring[0][0], ring[0][1], w, h);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < ring.length; i++) {
      const pi = worldToScreen(ring[i][0], ring[i][1], w, h);
      ctx.lineTo(pi.x, pi.y);
    }
    ctx.closePath();
    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (strokeStyle) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }
}

function drawFootprint(ctx, corners, w, h, fillStyle, strokeStyle) {
  if (!Array.isArray(corners) || corners.length < 2) return;
  ctx.beginPath();
  const p0 = worldToScreen(corners[0][0], corners[0][1], w, h);
  ctx.moveTo(p0.x, p0.y);
  for (let i = 1; i < corners.length; i++) {
    const pi = worldToScreen(corners[i][0], corners[i][1], w, h);
    ctx.lineTo(pi.x, pi.y);
  }
  ctx.closePath();
  ctx.fillStyle = fillStyle;
  ctx.fill();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function draw() {
  const { ctx, w, h, innerW, innerH } = updateTransform();
  ctx.clearRect(0, 0, w, h);
  if (!sceneGraph) return;

  const room = sceneGraph.room || {};
  const predMap = predictionsByTarget();
  const useWorstColor = $("chk-color").checked && predict;
  const weightKey = $("sel-weight").value;

  drawPolygonRings(
    ctx,
    room.walkable_polygon_xy,
    innerW,
    innerH,
    "rgba(200, 210, 220, 0.5)",
    "rgba(160, 170, 180, 0.35)",
    1
  );
  drawPolygonRings(
    ctx,
    room.polygon_xy,
    innerW,
    innerH,
    null,
    "#222",
    1.5
  );

  for (const node of sceneGraph.nodes || []) {
    const g = node.geometry || {};
    const corners = g.corners_xy;
    const ntype = node.type || "unknown";
    let fill = NODE_FILL[ntype] || NODE_FILL.unknown;
    if (useWorstColor) {
      const ws = worstStatusForTarget(predMap, node.id);
      if (ws && WORST_STATUS_COLOR[ws]) fill = WORST_STATUS_COLOR[ws];
    }
    if (Array.isArray(corners) && corners.length >= 3) {
      drawFootprint(ctx, corners, innerW, innerH, fill, "#333");
    }
  }

  const edges = visibleEdges();
  for (const e of edges) {
    const pa = anchors.get(e.source);
    const pb = anchors.get(e.target);
    if (!pa || !pb) continue;
    const sa = worldToScreen(pa.x, pa.y, innerW, innerH);
    const sb = worldToScreen(pb.x, pb.y, innerW, innerH);
    const sw = strokeForEdge(e, weightKey);
    ctx.beginPath();
    ctx.moveTo(sa.x, sa.y);
    ctx.lineTo(sb.x, sb.y);
    ctx.strokeStyle = `rgba(40, 40, 120, ${sw.alpha})`;
    ctx.lineWidth = sw.width;
    ctx.stroke();
  }

  const rHit = 6;
  for (const node of sceneGraph.nodes || []) {
    const p = anchors.get(node.id);
    if (!p) continue;
    const s = worldToScreen(p.x, p.y, innerW, innerH);
    const hi = node.id === hoverId || node.id === selectedId;
    ctx.beginPath();
    ctx.arc(s.x, s.y, hi ? rHit + 2 : rHit, 0, Math.PI * 2);
    ctx.fillStyle = hi ? "#fff" : "rgba(255,255,255,0.85)";
    ctx.fill();
    ctx.strokeStyle = hi ? "#06c" : "#555";
    ctx.lineWidth = hi ? 2 : 1;
    ctx.stroke();
  }
}

function formatNodeDetail(nodeId) {
  const node = (sceneGraph.nodes || []).find((n) => n.id === nodeId);
  const lines = [];
  if (node) {
    lines.push(`${node.id}  (${node.type} / ${node.kind})  ${node.label || ""}`);
    lines.push(JSON.stringify(node.geometry || {}).slice(0, 400));
  }
  const predMap = predictionsByTarget();
  const rows = predMap.get(nodeId);
  if (rows && rows.length) {
    lines.push("— predictions —");
    for (const r of rows) {
      lines.push(`  ${r.principle}: ${r.status}  (${r.score})`);
    }
  }
  const ex = applyExtrasLayer(extras, nodeId);
  if (ex) lines.push(ex);
  return lines.join("\n");
}

function updateSummary() {
  const el = $("summary");
  if (!sceneGraph) {
    el.textContent = "No scene graph loaded.";
    el.classList.add("muted");
    return;
  }
  const s = sceneGraph.summary || {};
  const parts = [
    `Nodes: ${s.n_nodes ?? (sceneGraph.nodes || []).length}`,
    `Edges: ${s.n_edges ?? (sceneGraph.edges || []).length}`,
    `Source: ${sceneGraph.source_scene_json || "—"}`,
  ];
  if (predict) {
    if (predict.graph_score != null) parts.push(`graph_score: ${predict.graph_score}`);
    parts.push(`predictions: ${(predict.principle_predictions || []).length} cells`);
  }
  el.textContent = parts.join(" · ");
  el.classList.remove("muted");
}

function loadSceneGraph(data) {
  sceneGraph = data;
  rebuildAnchors();
  computeBounds();
  populateWeightSelect();
  rebuildEdgeTypeFilters();
  updateSummary();
  draw();
}

function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => {
      try {
        resolve(JSON.parse(fr.result));
      } catch (e) {
        reject(e);
      }
    };
    fr.onerror = () => reject(fr.error);
    fr.readAsText(file, "utf-8");
  });
}

function canvasHitNode(clientX, clientY) {
  const cv = $("cv");
  const rect = cv.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const wCss = Math.max(320, Math.floor(rect.width));
  const hCss = Math.max(240, Math.floor(rect.height));
  const innerW = wCss - 16;
  const innerH = hCss - 16;
  let best = null;
  let bestD = 14;
  for (const node of sceneGraph.nodes || []) {
    const p = anchors.get(node.id);
    if (!p) continue;
    const s = worldToScreen(p.x, p.y, innerW, innerH);
    const dx = s.x - x;
    const dy = s.y - y;
    const d = Math.hypot(dx, dy);
    if (d < bestD) {
      bestD = d;
      best = node.id;
    }
  }
  return best;
}

function bindEvents() {
  $("file-scene").addEventListener("change", async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    try {
      const data = await readJsonFile(f);
      loadSceneGraph(data);
      $("detail").textContent = "";
    } catch (e) {
      alert("Invalid JSON: " + e);
    }
  });

  $("file-predict").addEventListener("change", async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) {
      predict = null;
      updateSummary();
      draw();
      return;
    }
    try {
      predict = await readJsonFile(f);
      updateSummary();
      draw();
    } catch (e) {
      alert("Invalid predict JSON: " + e);
    }
  });

  $("file-extras").addEventListener("change", async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) {
      extras = null;
      _renderExtrasPanels(null);
      return;
    }
    try {
      extras = await readJsonFile(f);
      applyExtrasLayer(extras, null);
      draw();
    } catch (e) {
      alert("Invalid extras JSON: " + e);
    }
  });

  $("chk-principle").addEventListener("change", () => {
    rebuildEdgeTypeFilters();
    draw();
  });
  $("chk-merge").addEventListener("change", () => draw());
  $("chk-color").addEventListener("change", () => draw());
  $("sel-weight").addEventListener("change", () => draw());

  const cv = $("cv");
  cv.addEventListener("mousemove", (ev) => {
    if (!sceneGraph) return;
    const id = canvasHitNode(ev.clientX, ev.clientY);
    if (id !== hoverId) {
      hoverId = id;
      draw();
      $("detail").textContent = id ? formatNodeDetail(id) : selectedId ? formatNodeDetail(selectedId) : "";
    }
  });
  cv.addEventListener("mouseleave", () => {
    hoverId = null;
    draw();
  });
  cv.addEventListener("click", (ev) => {
    if (!sceneGraph) return;
    const id = canvasHitNode(ev.clientX, ev.clientY);
    selectedId = id;
    if (id) $("detail").textContent = formatNodeDetail(id);
  });

  window.addEventListener("resize", () => {
    if (sceneGraph) draw();
  });
}

/**
 * Hook for LLM explanations or edge attention JSON.
 * Called on file load (nodeId=null) to populate sidebar panels, and on
 * hover/click (nodeId set) to surface per-node text in the detail panel.
 * @param {object|null} data parsed extras file
 * @param {string|null} nodeId selected/hover node, or null when file loads
 * @returns {string|null} extra text lines for the detail panel (hover/click)
 */
function applyExtrasLayer(data, nodeId) {
  if (!data || typeof data !== "object") {
    _renderExtrasPanels(null);
    return null;
  }

  // On initial load (nodeId === null) render the sidebar panels.
  if (nodeId === null) {
    _renderExtrasPanels(data);
  }

  // Per-node detail text (hover / click)
  if (Array.isArray(data.explanations)) {
    const rows = data.explanations.filter(
      (row) => !nodeId || row.target === nodeId || row.node_id === nodeId
    );
    if (!rows.length) return null;
    return rows
      .map((row) => {
        const p = row.principle || row.topic || "";
        const txt = row.text || row.summary || "";
        return `[LLM] ${p}: ${txt}`;
      })
      .join("\n");
  }

  if (Array.isArray(data.edge_attention) && !nodeId) {
    return `[extras] ${data.edge_attention.length} attention edges loaded`;
  }

  return null;
}

function _renderExtrasPanels(data) {
  const panel = $("extras-panel");
  if (!data) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "";

  // Score row
  const scoreEl = $("extras-score-row");
  const gs = data.graph_score != null ? ` · score ${data.graph_score}` : "";
  const lbl = data.overall_score_label ? ` — ${data.overall_score_label}` : "";
  scoreEl.textContent = `Room${gs}${lbl}`;

  // Summary
  const summaryEl = $("extras-summary");
  summaryEl.textContent = data.summary || "";

  // Recommendations
  const recsEl = $("extras-recs");
  recsEl.innerHTML = "";
  const recs = Array.isArray(data.recommendations) ? data.recommendations : [];
  if (recs.length) {
    const h = document.createElement("strong");
    h.textContent = "Recommendations";
    recsEl.appendChild(h);
    const ul = document.createElement("ul");
    for (const rec of recs) {
      const li = document.createElement("li");
      li.textContent = rec;
      ul.appendChild(li);
    }
    recsEl.appendChild(ul);
  }

  // Ranked violations table — clicking a row highlights that node
  const violEl = $("extras-violations");
  violEl.innerHTML = "";
  const violations = Array.isArray(data.ranked_violations) ? data.ranked_violations : [];
  if (violations.length) {
    const h = document.createElement("strong");
    h.textContent = "Ranked violations";
    violEl.appendChild(h);
    const table = document.createElement("table");
    const thead = table.createTHead();
    const hr = thead.insertRow();
    for (const col of ["Principle", "Target", "Impact", "Score"]) {
      const th = document.createElement("th");
      th.textContent = col;
      hr.appendChild(th);
    }
    const tbody = table.createTBody();
    for (const v of violations) {
      const tr = tbody.insertRow();
      const impactCls = `impact-${(v.impact || "medium").toLowerCase()}`;
      const statusCls = `status-${(v.status || "violated").toLowerCase()}`;
      const cells = [
        { text: (v.principle || "").replace(/_/g, " "), cls: statusCls },
        { text: v.target || "", cls: "" },
        { text: v.impact || "", cls: impactCls },
        { text: v.score != null ? Number(v.score).toFixed(2) : "", cls: "" },
      ];
      for (const { text, cls } of cells) {
        const td = tr.insertCell();
        td.textContent = text;
        if (cls) td.className = cls;
      }
      // Clicking a row selects that node in the canvas
      tr.addEventListener("click", () => {
        const tid = v.target;
        if (tid && sceneGraph) {
          selectedId = tid;
          $("detail").textContent = formatNodeDetail(tid);
          draw();
        }
      });
    }
    violEl.appendChild(table);
  }
}

bindEvents();
