/* MARVIN 3D showcase — real Jezero terrain, an animated rover with spinning wheels driving the
   recorded mission path, and targets that light up as they're cached. Reads window.DATA. */
(function () {
  "use strict";
  const D = window.DATA;
  const N = D.grid, EXT = D.extent, HEIGHT_EXAG = 7.0;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05060b);
  scene.fog = new THREE.FogExp2(0x05060b, 0.012);

  const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 400);
  camera.position.set(8.5, 6.5, 9.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  document.getElementById("app").appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.minDistance = 3;
  controls.maxDistance = 40;
  controls.target.set(0, 0.3, 0);
  if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) controls.autoRotate = false;

  scene.add(new THREE.AmbientLight(0x8fa2c0, 0.55));
  const sun = new THREE.DirectionalLight(0xfff1de, 1.25);
  sun.position.set(7, 12, 5);
  scene.add(sun);
  const rim = new THREE.DirectionalLight(0x4466aa, 0.5);
  rim.position.set(-6, 4, -7);
  scene.add(rim);

  // starfield
  const sg = new THREE.BufferGeometry(), sp = [];
  for (let i = 0; i < 1600; i++) sp.push((Math.random() - 0.5) * 260, (Math.random() - 0.5) * 260, (Math.random() - 0.5) * 260);
  sg.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
  scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0x9fb2d4, size: 0.18, sizeAttenuation: true })));

  // terrain height lookup (shared by mesh + rover so they always agree)
  let tmin = 1e9, tmax = -1e9;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const v = D.terrain[r][c]; if (v < tmin) tmin = v; if (v > tmax) tmax = v; }
  function heightAt(x, z) {
    const c = Math.max(0, Math.min(N - 1, Math.round((x + EXT) / (2 * EXT) * (N - 1))));
    const r = Math.max(0, Math.min(N - 1, Math.round((z + EXT) / (2 * EXT) * (N - 1))));
    return D.terrain[r][c] * HEIGHT_EXAG;
  }

  // terrain mesh (real Jezero DEM), colored dark-basin -> bright-rim
  const geo = new THREE.PlaneGeometry(2 * EXT, 2 * EXT, N - 1, N - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position, colors = [];
  for (let k = 0; k < pos.count; k++) {
    const x = pos.getX(k), z = pos.getZ(k), h = heightAt(x, z);
    pos.setY(k, h);
    const t = (h / HEIGHT_EXAG - tmin) / (tmax - tmin + 1e-9);
    const col = new THREE.Color().setHSL(0.055, 0.62, 0.10 + 0.42 * t);
    colors.push(col.r, col.g, col.b);
  }
  geo.computeVertexNormals();
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96, metalness: 0.02 })));

  // driven path as a glowing polyline
  const pts = D.trajectory.map(p => new THREE.Vector3(p[0], heightAt(p[0], p[1]) + 0.04, p[1]));
  const pathGeo = new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0x00e5ff })));

  // targets
  const targetMeshes = D.targets.map(t => {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.16),
      new THREE.MeshStandardMaterial({ color: 0xffae42, emissive: 0x5a3a00, emissiveIntensity: 0.6, roughness: 0.4 }));
    m.position.set(t.x, heightAt(t.x, t.y) + 0.22, t.y);
    scene.add(m);
    return { mesh: m, collectAt: t.collectAt, done: false };
  });

  // rover with spinning wheels
  const rover = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xdfe6ee, metalness: 0.35, roughness: 0.45 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x23262e, metalness: 0.2, roughness: 0.7 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.16, 0.36), bodyMat); body.position.y = 0.15; rover.add(body);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.22, 8), darkMat); mast.position.set(0.16, 0.32, 0); rover.add(mast);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.05, 0.09), darkMat); head.position.set(0.18, 0.43, 0); rover.add(head);
  const wheels = [];
  [[0.19, 0.17], [0.19, -0.17], [-0.19, 0.17], [-0.19, -0.17]].forEach(([wx, wz]) => {
    const axle = new THREE.Group(); axle.rotation.x = Math.PI / 2; axle.position.set(wx, 0.07, wz);
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.05, 18), darkMat);
    axle.add(w); rover.add(axle); wheels.push(w);
  });
  scene.add(rover);

  const traj = D.trajectory;
  let tt = 0;
  const totalStatus = document.getElementById("status");
  function animate() {
    requestAnimationFrame(animate);
    tt += 0.0016; if (tt > 1.18) tt = 0;              // loop with a pause at the end
    const p = Math.min(tt, 1) * (traj.length - 1);
    const i0 = Math.floor(p), i1 = Math.min(i0 + 1, traj.length - 1), f = p - i0;
    const x = traj[i0][0] * (1 - f) + traj[i1][0] * f;
    const y = traj[i0][1] * (1 - f) + traj[i1][1] * f;
    rover.position.set(x, heightAt(x, y) + 0.02, y);
    const dx = traj[i1][0] - traj[i0][0], dy = traj[i1][1] - traj[i0][1];
    if (Math.abs(dx) + Math.abs(dy) > 1e-5) rover.rotation.y = Math.atan2(-dy, dx);
    const speed = Math.hypot(dx, dy) * 40;
    wheels.forEach(w => (w.rotation.y += 0.15 + speed));
    let cached = 0;
    targetMeshes.forEach(t => {
      if (p >= t.collectAt) {
        cached++;
        if (!t.done) { t.done = true; t.mesh.material.color.set(0x5fdd8a); t.mesh.material.emissive.set(0x1b6b3a); }
      }
      t.mesh.rotation.y += 0.02;
    });
    if (totalStatus) totalStatus.textContent = cached + " / " + targetMeshes.length + " samples cached";
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  // stop auto-rotate once the user grabs the scene
  renderer.domElement.addEventListener("pointerdown", () => (controls.autoRotate = false));
})();
