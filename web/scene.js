/* MARVIN scene. Renders the recorded mission from window.DATA.
   Uses three.js (WebGL) where available (e.g. hosted / local browser); falls back to an animated
   Canvas-2D top-down map where WebGL is blocked (e.g. sandboxed artifact iframes). */
(function () {
  "use strict";
  const app = document.getElementById("app");
  const D = window.DATA;
  const status = document.getElementById("status");

  function fail(msg) {
    if (app) app.innerHTML = '<pre style="color:#fa4d56;font:12px/1.5 ui-monospace,monospace;' +
      'padding:16px;white-space:pre-wrap;margin:0">' + msg + '</pre>';
  }
  if (!app || !D) { fail("scene: missing #app or window.DATA"); return; }

  function webglOK() {
    try {
      const c = document.createElement("canvas");
      return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl")));
    } catch (e) { return false; }
  }

  // ---------------------------------------------------------------- shared helpers
  const N = D.grid, EXT = D.extent;
  let tmin = 1e9, tmax = -1e9;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const v = D.terrain[r][c]; if (v < tmin) tmin = v; if (v > tmax) tmax = v; }
  function norm(v) { return (v - tmin) / (tmax - tmin + 1e-9); }

  // ---------------------------------------------------------------- dispatch (helpers ready)
  let used3D = false;
  if (typeof THREE !== "undefined" && webglOK()) {
    try { run3D(); used3D = true; }
    catch (e) { if (window.console) console.warn("WebGL failed, 2D fallback:", e); app.innerHTML = ""; }
  }
  if (!used3D) {
    try { run2D(); } catch (e) { fail("2D scene error:\n" + ((e && e.stack) || e)); }
  }

  // ---------------------------------------------------------------- 2D fallback
  function run2D() {
    const cv = document.createElement("canvas");
    cv.style.cssText = "width:100%;height:100%;display:block";
    app.appendChild(cv);
    const ctx = cv.getContext("2d");

    // pre-render the terrain once at native grid resolution, then scale up smoothly
    const off = document.createElement("canvas"); off.width = N; off.height = N;
    const octx = off.getContext("2d"); const img = octx.createImageData(N, N);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const t = norm(D.terrain[r][c]);
      const rr = N - 1 - r;                         // flip so north is up
      const i = (rr * N + c) * 4;
      img.data[i] = 26 + 205 * t; img.data[i + 1] = 16 + 118 * t; img.data[i + 2] = 12 + 44 * t; img.data[i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);

    const traj = D.trajectory;
    const toPx = (x, y, w, h) => [(x + EXT) / (2 * EXT) * w, (EXT - y) / (2 * EXT) * h];
    let tt = 0;

    function rrect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }
    function rover(x, y, ang) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
      ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.beginPath(); ctx.ellipse(0, 1, 12, 8, 0, 0, 7); ctx.fill();
      ctx.fillStyle = "#e7ecf3"; ctx.strokeStyle = "#0b0d12"; ctx.lineWidth = 1;
      rrect(-9, -6, 18, 12, 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#23262e";[[-7, -7.5], [7, -7.5], [-7, 4.5], [7, 4.5]].forEach(w => ctx.fillRect(w[0] - 2.5, w[1], 5, 3));
      ctx.fillStyle = "#ff832b"; ctx.beginPath(); ctx.moveTo(9, -4); ctx.lineTo(15, 0); ctx.lineTo(9, 4); ctx.closePath(); ctx.fill();
      ctx.restore();
    }

    function frame() {
      requestAnimationFrame(frame);
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const w = app.clientWidth || 640, h = app.clientHeight || 420;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, N, N, 0, 0, w, h);

      ctx.strokeStyle = "rgba(150,170,200,0.09)"; ctx.lineWidth = 1;
      for (let g = -4; g <= 4; g += 2) {
        let px = toPx(g, 0, w, h)[0]; ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
        let py = toPx(0, g, w, h)[1]; ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
      }

      tt += 0.0016; if (tt > 1.18) tt = 0;
      const p = Math.min(tt, 1) * (traj.length - 1);
      const i0 = Math.floor(p), i1 = Math.min(i0 + 1, traj.length - 1), f = p - i0;
      const rx = traj[i0][0] * (1 - f) + traj[i1][0] * f, ry = traj[i0][1] * (1 - f) + traj[i1][1] * f;

      ctx.strokeStyle = "#ff9d5c"; ctx.lineWidth = 2.4; ctx.lineJoin = "round"; ctx.beginPath();
      for (let k = 0; k <= i0; k++) { const q = toPx(traj[k][0], traj[k][1], w, h); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
      const rp = toPx(rx, ry, w, h); ctx.lineTo(rp[0], rp[1]); ctx.stroke();

      let cached = 0;
      D.targets.forEach(t => {
        const done = p >= t.collectAt; if (done) cached++;
        const q = toPx(t.x, t.y, w, h);
        ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, 7);
        ctx.fillStyle = done ? "#5fdd8a" : "#ff832b"; ctx.fill();
        ctx.lineWidth = 1.4; ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.stroke();
      });

      const dx = traj[i1][0] - traj[i0][0], dy = traj[i1][1] - traj[i0][1];
      rover(rp[0], rp[1], Math.atan2(-dy, dx));
      if (status) status.textContent = cached + " / " + D.targets.length + " samples cached";
    }
    frame();
  }

  // ---------------------------------------------------------------- 3D (WebGL)
  function run3D() {
    const W = () => app.clientWidth || window.innerWidth;
    const H = () => app.clientHeight || window.innerHeight;
    const HEIGHT_EXAG = 7.0;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x161616);
    scene.fog = new THREE.FogExp2(0x161616, 0.014);
    const camera = new THREE.PerspectiveCamera(50, W() / H(), 0.1, 400);
    camera.position.set(8.5, 6.5, 9.5);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W(), H()); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    app.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
    controls.minDistance = 3; controls.maxDistance = 40; controls.target.set(0, 0.3, 0);
    if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) controls.autoRotate = false;

    scene.add(new THREE.AmbientLight(0x8fa2c0, 0.55));
    const sun = new THREE.DirectionalLight(0xfff1de, 1.25); sun.position.set(7, 12, 5); scene.add(sun);
    const rim = new THREE.DirectionalLight(0x4466aa, 0.5); rim.position.set(-6, 4, -7); scene.add(rim);

    function heightAt(x, z) {
      const c = Math.max(0, Math.min(N - 1, Math.round((x + EXT) / (2 * EXT) * (N - 1))));
      const r = Math.max(0, Math.min(N - 1, Math.round((z + EXT) / (2 * EXT) * (N - 1))));
      return D.terrain[r][c] * HEIGHT_EXAG;
    }
    const geo = new THREE.PlaneGeometry(2 * EXT, 2 * EXT, N - 1, N - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position, colors = [];
    for (let k = 0; k < pos.count; k++) {
      const x = pos.getX(k), z = pos.getZ(k), h = heightAt(x, z); pos.setY(k, h);
      const col = new THREE.Color().setHSL(0.055, 0.62, 0.12 + 0.42 * norm(h / HEIGHT_EXAG));
      colors.push(col.r, col.g, col.b);
    }
    geo.computeVertexNormals();
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96, metalness: 0.02 })));

    const pts = D.trajectory.map(p => new THREE.Vector3(p[0], heightAt(p[0], p[1]) + 0.04, p[1]));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xff9d5c })));

    const targetMeshes = D.targets.map(t => {
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.16),
        new THREE.MeshStandardMaterial({ color: 0xff832b, emissive: 0x5a3a00, emissiveIntensity: 0.6, roughness: 0.4 }));
      m.position.set(t.x, heightAt(t.x, t.y) + 0.22, t.y); scene.add(m);
      return { mesh: m, collectAt: t.collectAt, done: false };
    });

    const rover = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xdfe6ee, metalness: 0.35, roughness: 0.45 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x23262e, metalness: 0.2, roughness: 0.7 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.16, 0.36), bodyMat); body.position.y = 0.15; rover.add(body);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.22, 8), darkMat);
    mast.position.set(0.16, 0.32, 0); rover.add(mast);
    const wheels = [];
    [[0.19, 0.17], [0.19, -0.17], [-0.19, 0.17], [-0.19, -0.17]].forEach(w => {
      const axle = new THREE.Group(); axle.rotation.x = Math.PI / 2; axle.position.set(w[0], 0.07, w[1]);
      const wh = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.05, 18), darkMat);
      axle.add(wh); rover.add(axle); wheels.push(wh);
    });
    scene.add(rover);

    const traj = D.trajectory; let tt = 0;
    (function animate() {
      requestAnimationFrame(animate);
      tt += 0.0016; if (tt > 1.18) tt = 0;
      const p = Math.min(tt, 1) * (traj.length - 1);
      const i0 = Math.floor(p), i1 = Math.min(i0 + 1, traj.length - 1), f = p - i0;
      const x = traj[i0][0] * (1 - f) + traj[i1][0] * f, y = traj[i0][1] * (1 - f) + traj[i1][1] * f;
      rover.position.set(x, heightAt(x, y) + 0.02, y);
      const dx = traj[i1][0] - traj[i0][0], dy = traj[i1][1] - traj[i0][1];
      if (Math.abs(dx) + Math.abs(dy) > 1e-5) rover.rotation.y = Math.atan2(-dy, dx);
      wheels.forEach(w => (w.rotation.y += 0.15 + Math.hypot(dx, dy) * 40));
      let cached = 0;
      targetMeshes.forEach(t => {
        if (p >= t.collectAt) { cached++; if (!t.done) { t.done = true; t.mesh.material.color.set(0x5fdd8a); t.mesh.material.emissive.set(0x1b6b3a); } }
        t.mesh.rotation.y += 0.02;
      });
      if (status) status.textContent = cached + " / " + targetMeshes.length + " samples cached";
      controls.update(); renderer.render(scene, camera);
    })();

    new ResizeObserver(() => { camera.aspect = W() / H(); camera.updateProjectionMatrix(); renderer.setSize(W(), H()); }).observe(app);
    renderer.domElement.addEventListener("pointerdown", () => (controls.autoRotate = false));
  }
})();
