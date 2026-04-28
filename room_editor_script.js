    const LABEL_PRESETS = {
      bed: { width: 2.00, depth: 1.50, height: 0.70 },
      sofa: { width: 2.00, depth: 0.90, height: 0.85 },
      chair: { width: 0.55, depth: 0.55, height: 0.95 },
      table: { width: 1.20, depth: 0.75, height: 0.75 },
      desk: { width: 1.35, depth: 0.70, height: 0.75 },
      work_desk: { width: 1.50, depth: 0.75, height: 0.75 },
      kitchen_counter: { width: 2.00, depth: 0.65, height: 0.95 },
      stove: { width: 0.75, depth: 0.65, height: 0.95 },
      drawer: { width: 0.85, depth: 0.50, height: 0.95 },
      cabinet_or_shelf: { width: 0.90, depth: 0.45, height: 1.80 },
      shelf_or_bookcase: { width: 0.90, depth: 0.35, height: 1.90 },
      floor_lamp: { width: 0.25, depth: 0.25, height: 1.60 },
      floor_lamp_or_tall_thin_object: { width: 0.25, depth: 0.25, height: 1.60 },
      box: { width: 0.50, depth: 0.40, height: 0.35 },
      container: { width: 0.45, depth: 0.35, height: 0.35 },
      door: { width: 0.90, depth: 0.08, height: 2.05 },
      window: { width: 1.20, depth: 0.08, height: 1.20 },
      wall: { width: 1.00, depth: 0.10, height: 2.50 },
      unknown_object: { width: 0.75, depth: 0.75, height: 0.75 }
    };
    const LABELS = Object.keys(LABEL_PRESETS);
    const ARCH_LABELS = new Set(['door', 'window', 'wall']);

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const fileInput = document.getElementById('fileInput');
    const exportBtn = document.getElementById('exportBtn');
    const resetBtn = document.getElementById('resetBtn');
    const rotateLeftBtn = document.getElementById('rotateLeftBtn');
    const rotateRightBtn = document.getElementById('rotateRightBtn');
    const showLabelsCheckbox = document.getElementById('showLabels');
    const showGridCheckbox = document.getElementById('showGrid');
    const snapModeCheckbox = document.getElementById('snapMode');
    const objectList = document.getElementById('objectList');
    const selectedInfo = document.getElementById('selectedInfo');
    const editPanel = document.getElementById('editPanel');
    const addLabel = document.getElementById('addLabel');
    const editLabel = document.getElementById('editLabel');
    const customLabel = document.getElementById('customLabel');
    const editX = document.getElementById('editX');
    const editY = document.getElementById('editY');
    const editW = document.getElementById('editW');
    const editD = document.getElementById('editD');
    const editTheta = document.getElementById('editTheta');
    const editHeight = document.getElementById('editHeight');
    const applyEditBtn = document.getElementById('applyEditBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const duplicateBtn = document.getElementById('duplicateBtn');
    const bringFrontBtn = document.getElementById('bringFrontBtn');
    const snap90Btn = document.getElementById('snap90Btn');
    const swapSizeBtn = document.getElementById('swapSizeBtn');
    const addObjectBtn = document.getElementById('addObjectBtn');
    const addAtViewBtn = document.getElementById('addAtViewBtn');

    for (const label of LABELS) {
      for (const select of [addLabel, editLabel]) {
        const opt = document.createElement('option');
        opt.value = label;
        opt.textContent = label;
        select.appendChild(opt);
      }
    }

    let scene = null;
    let originalScene = null;
    let selectedId = null;
    let draggingObject = false;
    let panning = false;
    let dragOffset = { x: 0, y: 0 };
    let lastMouse = { x: 0, y: 0 };
    let viewport = { scale: 70, offsetX: 0, offsetY: 0 };
    let spaceHeld = false;

    function deepCopy(obj) { return JSON.parse(JSON.stringify(obj)); }
    function deg(rad) { return rad * 180 / Math.PI; }
    function rad(degValue) { return degValue * Math.PI / 180; }
    function clampNumber(v, fallback) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }

    function normalizeObject(obj) {
      obj.id = obj.id ?? nextObjectId();
      obj.label = obj.label || 'unknown_object';
      obj.raw_label = obj.raw_label || obj.label;
      obj.cx = clampNumber(obj.cx, 0);
      obj.cy = clampNumber(obj.cy, 0);
      obj.width = clampNumber(obj.width ?? obj.w, 0.75);
      obj.depth = clampNumber(obj.depth ?? obj.d, 0.75);
      obj.theta = clampNumber(obj.theta, 0);
      obj.z_min = clampNumber(obj.z_min, 0);
      obj.z_max = clampNumber(obj.z_max, obj.z_min + 0.75);
      obj.height = clampNumber(obj.height, obj.z_max - obj.z_min);
      obj.footprint = objectCorners(obj).map(p => [p.x, p.y]);
      return obj;
    }

    function normalizeScene() {
      if (!scene) return;
      scene.units = scene.units || 'meters';
      scene.objects = (scene.objects || []).map(normalizeObject);
      ensureRoomBBox();
    }

    function ensureRoomBBox() {
      if (!scene) return;
      let pts = [];
      if (scene.room_polygon?.length) pts.push(...scene.room_polygon.map(([x, y]) => ({ x, y })));
      for (const obj of scene.objects || []) pts.push(...objectCorners(obj));
      if (!pts.length) pts = [{x:0,y:0},{x:4,y:3}];
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const min_x = Math.min(...xs), max_x = Math.max(...xs), min_y = Math.min(...ys), max_y = Math.max(...ys);
      scene.room_bbox = { min_x, min_y, max_x, max_y, width: max_x - min_x, height: max_y - min_y };
    }

    function nextObjectId() {
      const ids = scene?.objects?.map(o => Number(o.id)).filter(Number.isFinite) || [];
      return ids.length ? Math.max(...ids) + 1 : 1;
    }

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function worldToScreen(x, y) {
      return { x: x * viewport.scale + viewport.offsetX, y: canvas.clientHeight - (y * viewport.scale + viewport.offsetY) };
    }
    function screenToWorld(x, y) {
      return { x: (x - viewport.offsetX) / viewport.scale, y: ((canvas.clientHeight - y) - viewport.offsetY) / viewport.scale };
    }

    function fitScene() {
      if (!scene) return;
      ensureRoomBBox();
      const bbox = scene.room_bbox;
      const w = Math.max(bbox.width, 0.1), h = Math.max(bbox.height, 0.1);
      const pad = 40;
      const scaleX = (canvas.clientWidth - pad * 2) / w;
      const scaleY = (canvas.clientHeight - pad * 2) / h;
      viewport.scale = Math.max(10, Math.min(scaleX, scaleY));
      viewport.offsetX = pad - bbox.min_x * viewport.scale + ((canvas.clientWidth - pad * 2) - w * viewport.scale) / 2;
      viewport.offsetY = pad - bbox.min_y * viewport.scale + ((canvas.clientHeight - pad * 2) - h * viewport.scale) / 2;
      draw();
    }

    function roomCenter() {
      if (scene?.room_polygon?.length) {
        const pts = scene.room_polygon;
        return { x: pts.reduce((s, p) => s + p[0], 0) / pts.length, y: pts.reduce((s, p) => s + p[1], 0) / pts.length };
      }
      ensureRoomBBox();
      const b = scene.room_bbox;
      return { x: (b.min_x + b.max_x) / 2, y: (b.min_y + b.max_y) / 2 };
    }

    function viewCenter() { return screenToWorld(canvas.clientWidth / 2, canvas.clientHeight / 2); }

    function objectCorners(obj) {
      const hw = Math.max(obj.width, 0.01) / 2;
      const hd = Math.max(obj.depth, 0.01) / 2;
      const c = Math.cos(obj.theta || 0), s = Math.sin(obj.theta || 0);
      const local = [[-hw,-hd],[hw,-hd],[hw,hd],[-hw,hd]];
      return local.map(([x, y]) => ({ x: obj.cx + x * c - y * s, y: obj.cy + x * s + y * c }));
    }

    function pointInPolygon(point, polygon) {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;
        const intersect = ((yi > point.y) !== (yj > point.y)) &&
          (point.x < (xj - xi) * (point.y - yi) / ((yj - yi) || 1e-9) + xi);
        if (intersect) inside = !inside;
      }
      return inside;
    }

    function drawGrid() {
      if (!showGridCheckbox.checked) return;
      const spacing = 0.5;
      ctx.save();
      ctx.strokeStyle = '#ececec';
      ctx.lineWidth = 1;
      const bbox = scene?.room_bbox;
      if (!bbox) { ctx.restore(); return; }
      const startX = Math.floor(bbox.min_x / spacing) * spacing - spacing;
      const endX = Math.ceil(bbox.max_x / spacing) * spacing + spacing;
      const startY = Math.floor(bbox.min_y / spacing) * spacing - spacing;
      const endY = Math.ceil(bbox.max_y / spacing) * spacing + spacing;
      for (let x = startX; x <= endX; x += spacing) {
        const a = worldToScreen(x, startY), b = worldToScreen(x, endY);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let y = startY; y <= endY; y += spacing) {
        const a = worldToScreen(startX, y), b = worldToScreen(endX, y);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.restore();
    }

    function colorForLabel(label, selected) {
      if (selected) return { fill: 'rgba(114,196,255,0.25)', stroke: '#0079d6' };
      if (label === 'door') return { fill: 'rgba(80,180,90,0.18)', stroke: '#2b8a3e' };
      if (label === 'window') return { fill: 'rgba(50,140,255,0.16)', stroke: '#1971c2' };
      if (label === 'wall') return { fill: 'rgba(0,0,0,0.18)', stroke: '#111' };
      return { fill: 'rgba(80,80,80,0.10)', stroke: '#333' };
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      if (!scene) return;
      drawGrid();
      const room = scene.room_polygon || [];
      if (room.length >= 3) {
        ctx.save();
        ctx.beginPath();
        room.forEach(([x, y], i) => {
          const p = worldToScreen(x, y);
          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(0,0,0,0.04)';
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 2;
        ctx.fill(); ctx.stroke();
        ctx.restore();
      }
      for (const obj of scene.objects || []) {
        const selected = obj.id === selectedId;
        const corners = objectCorners(obj).map(p => worldToScreen(p.x, p.y));
        const colors = colorForLabel(obj.label, selected);
        ctx.save();
        ctx.beginPath();
        corners.forEach((p, i) => { if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
        ctx.closePath();
        ctx.fillStyle = colors.fill; ctx.strokeStyle = colors.stroke; ctx.lineWidth = selected ? 2.4 : 1.4;
        ctx.fill(); ctx.stroke();
        const center = worldToScreen(obj.cx, obj.cy);
        const heading = worldToScreen(obj.cx + (obj.width / 2) * Math.cos(obj.theta), obj.cy + (obj.width / 2) * Math.sin(obj.theta));
        ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(heading.x, heading.y); ctx.strokeStyle = colors.stroke; ctx.lineWidth = 1.2; ctx.stroke();
        if (showLabelsCheckbox.checked) {
          ctx.fillStyle = '#111'; ctx.font = '12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(obj.label, center.x, center.y);
        }
        ctx.restore();
      }
    }

    function getSelected() { return scene?.objects?.find(o => o.id === selectedId) || null; }

    function renderObjectList() {
      objectList.innerHTML = '';
      if (!scene?.objects) return;
      for (const obj of scene.objects) {
        const div = document.createElement('div');
        div.className = 'object-item' + (obj.id === selectedId ? ' selected' : '');
        div.innerHTML = `<strong>${obj.label}</strong><br><span class="muted">id ${obj.id}</span><div class="tag">${obj.width.toFixed(2)} m × ${obj.depth.toFixed(2)} m · ${deg(obj.theta).toFixed(0)}°</div>`;
        div.onclick = () => selectObject(obj.id);
        objectList.appendChild(div);
      }
    }

    function syncEditFields(obj) {
      if (!obj) { editPanel.style.display = 'none'; return; }
      editPanel.style.display = 'grid';
      editLabel.value = LABELS.includes(obj.label) ? obj.label : 'unknown_object';
      customLabel.value = LABELS.includes(obj.label) ? '' : obj.label;
      editX.value = obj.cx.toFixed(2);
      editY.value = obj.cy.toFixed(2);
      editW.value = obj.width.toFixed(2);
      editD.value = obj.depth.toFixed(2);
      editTheta.value = deg(obj.theta).toFixed(1);
      editHeight.value = Math.max(obj.height || (obj.z_max - obj.z_min), 0).toFixed(2);
    }

    function updateSelectedInfo() {
      const obj = getSelected();
      if (!obj) { selectedInfo.textContent = 'None'; editPanel.style.display = 'none'; return; }
      selectedInfo.innerHTML = `<strong>${obj.label}</strong><br>center: (${obj.cx.toFixed(2)}, ${obj.cy.toFixed(2)}) m<br>size: ${obj.width.toFixed(2)} × ${obj.depth.toFixed(2)} m<br>rotation: ${deg(obj.theta).toFixed(1)}°<br>z-range: ${obj.z_min.toFixed(2)} to ${obj.z_max.toFixed(2)} m`;
      syncEditFields(obj);
    }

    function selectObject(id) { selectedId = id; updateSelectedInfo(); renderObjectList(); draw(); }

    function pickObject(worldPoint) {
      if (!scene?.objects) return null;
      for (let i = scene.objects.length - 1; i >= 0; i--) {
        const obj = scene.objects[i];
        if (pointInPolygon(worldPoint, objectCorners(obj))) return obj;
      }
      return null;
    }

    function refreshAll() { ensureRoomBBox(); updateSelectedInfo(); renderObjectList(); draw(); }

    function nudgeRotation(deltaDeg) {
      const obj = getSelected(); if (!obj) return;
      obj.theta += rad(deltaDeg); obj.footprint = objectCorners(obj).map(p => [p.x, p.y]); refreshAll();
    }

    function applySelectedEdits() {
      const obj = getSelected(); if (!obj) return;
      const label = customLabel.value.trim() || editLabel.value;
      obj.label = label;
      obj.raw_label = obj.raw_label || label;
      obj.cx = clampNumber(editX.value, obj.cx);
      obj.cy = clampNumber(editY.value, obj.cy);
      obj.width = Math.max(0.01, clampNumber(editW.value, obj.width));
      obj.depth = Math.max(0.01, clampNumber(editD.value, obj.depth));
      obj.theta = rad(clampNumber(editTheta.value, deg(obj.theta)));
      obj.height = Math.max(0, clampNumber(editHeight.value, obj.height || obj.z_max - obj.z_min));
      obj.z_min = clampNumber(obj.z_min, 0);
      obj.z_max = obj.z_min + obj.height;
      obj.source = obj.source || 'manual_or_corrected';
      obj.footprint = objectCorners(obj).map(p => [p.x, p.y]);
      refreshAll();
    }

    function makeObject(label, center) {
      const p = LABEL_PRESETS[label] || LABEL_PRESETS.unknown_object;
      const obj = {
        id: nextObjectId(), object_id: nextObjectId(), label, raw_label: label,
        cx: center.x, cy: center.y, width: p.width, depth: p.depth, theta: 0,
        z_min: 0, z_max: p.height, height: p.height, point_count: 0,
        source: 'manual_added', confidence: 1.0, footprint: [], bbox3d_center: [], bbox3d_size: []
      };
      obj.footprint = objectCorners(obj).map(pt => [pt.x, pt.y]);
      obj.bbox3d_center = [obj.cx, obj.cy, p.height / 2];
      obj.bbox3d_size = [obj.width, obj.depth, p.height];
      return obj;
    }

    function addObjectAt(center) {
      if (!scene) {
        scene = { scene_id: 'manual_scene', units: 'meters', room_polygon: [[0,0],[4,0],[4,3],[0,3]], objects: [] };
        originalScene = deepCopy(scene);
      }
      const obj = makeObject(addLabel.value, center);
      scene.objects.push(obj);
      selectObject(obj.id);
    }

    function deleteSelected() {
      if (!scene || selectedId == null) return;
      scene.objects = scene.objects.filter(o => o.id !== selectedId);
      selectedId = scene.objects[0]?.id ?? null;
      refreshAll();
    }

    function duplicateSelected() {
      const obj = getSelected(); if (!obj) return;
      const dup = deepCopy(obj);
      dup.id = nextObjectId(); dup.object_id = dup.id; dup.cx += 0.15; dup.cy += 0.15; dup.source = 'manual_duplicated';
      dup.footprint = objectCorners(dup).map(p => [p.x, p.y]);
      scene.objects.push(dup); selectObject(dup.id);
    }

    function bringSelectedFront() {
      const obj = getSelected(); if (!obj) return;
      scene.objects = scene.objects.filter(o => o.id !== selectedId);
      scene.objects.push(obj); refreshAll();
    }

    function snapSelected90() {
      const obj = getSelected(); if (!obj) return;
      obj.theta = rad(Math.round(deg(obj.theta) / 90) * 90);
      obj.footprint = objectCorners(obj).map(p => [p.x, p.y]); refreshAll();
    }

    function swapSelectedSize() {
      const obj = getSelected(); if (!obj) return;
      [obj.width, obj.depth] = [obj.depth, obj.width];
      obj.theta += Math.PI / 2;
      obj.footprint = objectCorners(obj).map(p => [p.x, p.y]); refreshAll();
    }

    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0]; if (!file) return;
      const text = await file.text();
      scene = JSON.parse(text); normalizeScene(); originalScene = deepCopy(scene);
      selectedId = scene.objects?.[0]?.id ?? null;
      fitScene(); refreshAll();
    });

    exportBtn.addEventListener('click', () => {
      if (!scene) return;
      for (const obj of scene.objects || []) { obj.footprint = objectCorners(obj).map(p => [p.x, p.y]); }
      const blob = new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `${scene.scene_id || 'scene'}_edited_layout.json`; a.click();
      URL.revokeObjectURL(a.href);
    });

    resetBtn.addEventListener('click', () => { if (!originalScene) return; scene = deepCopy(originalScene); normalizeScene(); selectedId = scene.objects?.[0]?.id ?? null; fitScene(); refreshAll(); });
    rotateLeftBtn.addEventListener('click', () => nudgeRotation(-5));
    rotateRightBtn.addEventListener('click', () => nudgeRotation(5));
    applyEditBtn.addEventListener('click', applySelectedEdits);
    deleteBtn.addEventListener('click', deleteSelected);
    duplicateBtn.addEventListener('click', duplicateSelected);
    bringFrontBtn.addEventListener('click', bringSelectedFront);
    snap90Btn.addEventListener('click', snapSelected90);
    swapSizeBtn.addEventListener('click', swapSelectedSize);
    addObjectBtn.addEventListener('click', () => addObjectAt(roomCenter()));
    addAtViewBtn.addEventListener('click', () => addObjectAt(viewCenter()));
    showLabelsCheckbox.addEventListener('change', draw);
    showGridCheckbox.addEventListener('change', draw);

    canvas.addEventListener('mousedown', (e) => {
      if (!scene) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const world = screenToWorld(x, y); lastMouse = { x, y };
      if (spaceHeld) { panning = true; canvas.classList.add('dragging'); return; }
      const obj = pickObject(world);
      if (obj) { selectedId = obj.id; draggingObject = true; dragOffset = { x: obj.cx - world.x, y: obj.cy - world.y }; }
      else { selectedId = null; }
      refreshAll();
    });

    window.addEventListener('mousemove', (e) => {
      if (!scene) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const dx = x - lastMouse.x, dy = y - lastMouse.y; lastMouse = { x, y };
      if (panning) { viewport.offsetX += dx; viewport.offsetY -= dy; draw(); return; }
      if (!draggingObject || selectedId == null) return;
      const obj = getSelected(); if (!obj) return;
      const world = screenToWorld(x, y);
      let nx = world.x + dragOffset.x, ny = world.y + dragOffset.y;
      if (snapModeCheckbox.checked) { nx = Math.round(nx / 0.05) * 0.05; ny = Math.round(ny / 0.05) * 0.05; }
      obj.cx = nx; obj.cy = ny; obj.footprint = objectCorners(obj).map(p => [p.x, p.y]);
      updateSelectedInfo(); renderObjectList(); draw();
    });

    window.addEventListener('mouseup', () => { draggingObject = false; panning = false; canvas.classList.remove('dragging'); });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
      const before = screenToWorld(mouseX, mouseY);
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
      viewport.scale = Math.max(5, Math.min(1000, viewport.scale * factor));
      const after = screenToWorld(mouseX, mouseY);
      viewport.offsetX += (after.x - before.x) * viewport.scale;
      viewport.offsetY += (after.y - before.y) * viewport.scale;
      draw();
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { spaceHeld = true; e.preventDefault(); }
      if (e.key === 'q' || e.key === 'Q') nudgeRotation(-5);
      if (e.key === 'e' || e.key === 'E') nudgeRotation(5);
      if (e.key === 'Delete' || e.key === 'Backspace') { if (getSelected()) { e.preventDefault(); deleteSelected(); } }
    });
    window.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceHeld = false; });
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
