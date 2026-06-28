const $ = (id) => document.getElementById(id);

const imgOriginal = $("img-original");
const imgYolo = $("img-yolo");
const imgOrb = $("img-orb");
const statusEl = $("status");
const btnPlay = $("btn-play");
const speedLabel = $("speed-label");
const speedSlider = $("speed-slider");
const scrubber = $("scrubber");
const timeLabel = $("time-label");
const loopCheck = $("loop-check");
const cameraSelect = $("camera-select");
const mapPos = $("map-pos");
const mapStats = $("map-stats");
const mapCanvas = $("map-canvas");

let scrubbing = false;
let speedOptions = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
let activeTab = "live";
let map3d = null;
let lastPointcloud = null;

function setImg(img, b64) {
  if (b64) img.src = `data:image/jpeg;base64,${b64}`;
}

function formatTime(sec) {
  sec = Math.max(0, sec || 0);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "POST" };
  await fetch(path, opts);
}

// --- Tab switching ---
document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    activeTab = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("view-live").classList.toggle("active", activeTab === "live");
    $("view-map").classList.toggle("active", activeTab === "map");
    if (activeTab === "map") {
      initMap3D();
      if (lastPointcloud) updateMap3D(lastPointcloud);
      onMapResize();
    }
  };
});

// --- Three.js 3D point cloud ---
function initMap3D() {
  if (map3d) return;

  const container = $("map-3d-container");
  const w = container.clientWidth;
  const h = container.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(60, w / h, 0.05, 500);
  camera.position.set(3, 2, 5);

  const renderer = new THREE.WebGLRenderer({ canvas: mapCanvas, antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // Grid on ground plane (Y-up in Three.js; our world Y is down, so swap)
  const grid = new THREE.GridHelper(20, 20, 0xcccccc, 0xe8e8e8);
  scene.add(grid);

  const axes = new THREE.AxesHelper(2);
  scene.add(axes);

  const cloudGeo = new THREE.BufferGeometry();
  const cloudMat = new THREE.PointsMaterial({ size: 0.06, vertexColors: true, sizeAttenuation: true });
  const cloudMesh = new THREE.Points(cloudGeo, cloudMat);
  scene.add(cloudMesh);

  const trajGeo = new THREE.BufferGeometry();
  const trajMat = new THREE.LineBasicMaterial({ color: 0x0099cc, linewidth: 2 });
  const trajLine = new THREE.Line(trajGeo, trajMat);
  scene.add(trajLine);

  const camGeo = new THREE.SphereGeometry(0.15, 16, 16);
  const camMat = new THREE.MeshBasicMaterial({ color: 0xff6600 });
  const camMarker = new THREE.Mesh(camGeo, camMat);
  scene.add(camMarker);

  const dirGeo = new THREE.ConeGeometry(0.08, 0.35, 8);
  const dirMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
  const dirCone = new THREE.Mesh(dirGeo, dirMat);
  dirCone.rotation.x = Math.PI / 2;
  camMarker.add(dirCone);
  dirCone.position.z = 0.3;

  map3d = { scene, camera, renderer, controls, cloudMesh, cloudGeo, trajLine, trajGeo, camMarker, autoCentered: false };

  function animate() {
    requestAnimationFrame(animate);
    if (activeTab === "map") {
      controls.update();
      renderer.render(scene, camera);
    }
  }
  animate();
}

function worldToThree(x, y, z) {
  // World: X right, Y down, Z forward → Three.js: X right, Y up, Z toward viewer
  return new THREE.Vector3(x, -y, z);
}

function updateMap3D(pc) {
  lastPointcloud = pc;
  if (!map3d || activeTab !== "map") return;

  const { cloudMesh, cloudGeo, trajLine, trajGeo, camMarker, camera, controls } = map3d;

  if (pc.points && pc.points.length > 0) {
    const verts = new Float32Array(pc.points.length * 3);
    const cols = new Float32Array(pc.points.length * 3);
    for (let i = 0; i < pc.points.length; i++) {
      const [x, y, z] = pc.points[i];
      const v = worldToThree(x, y, z);
      verts[i * 3] = v.x;
      verts[i * 3 + 1] = v.y;
      verts[i * 3 + 2] = v.z;
      const c = pc.colors[i] || [100, 100, 100];
      cols[i * 3] = c[0] / 255;
      cols[i * 3 + 1] = c[1] / 255;
      cols[i * 3 + 2] = c[2] / 255;
    }
    cloudGeo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    cloudGeo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    cloudGeo.computeBoundingSphere();
    cloudMesh.visible = true;
  } else {
    cloudMesh.visible = false;
  }

  if (pc.trajectory && pc.trajectory.length >= 2) {
    const tverts = new Float32Array(pc.trajectory.length * 3);
    for (let i = 0; i < pc.trajectory.length; i++) {
      const v = worldToThree(pc.trajectory[i][0], pc.trajectory[i][1], pc.trajectory[i][2]);
      tverts[i * 3] = v.x;
      tverts[i * 3 + 1] = v.y;
      tverts[i * 3 + 2] = v.z;
    }
    trajGeo.setAttribute("position", new THREE.BufferAttribute(tverts, 3));
    trajLine.visible = true;
  } else {
    trajLine.visible = false;
  }

  if (pc.position) {
    const p = worldToThree(pc.position[0], pc.position[1], pc.position[2]);
    camMarker.position.copy(p);
    const yaw = (pc.yaw_deg || 0) * Math.PI / 180;
    camMarker.rotation.y = -yaw;

    if (!map3d.autoCentered && pc.count > 50) {
      controls.target.copy(p);
      camera.position.set(p.x + 3, p.y + 2, p.z + 4);
      map3d.autoCentered = true;
    } else if (pc.count > 0) {
      controls.target.lerp(p, 0.05);
    }
  }

  const pos = pc.position || [0, 0, 0];
  mapPos.textContent = `Position  X:${pos[0].toFixed(2)}m  Y:${pos[1].toFixed(2)}m  Z:${pos[2].toFixed(2)}m  ·  Heading ${(pc.yaw_deg || 0).toFixed(1)}°`;
  mapStats.textContent = `Points: ${pc.count || 0}  ·  Status: ${pc.status || "INIT"}  ·  Path: ${(pc.distance_m || 0).toFixed(2)}m`;
}

function onMapResize() {
  if (!map3d) return;
  const container = $("map-3d-container");
  const w = container.clientWidth;
  const h = container.clientHeight;
  map3d.camera.aspect = w / h;
  map3d.camera.updateProjectionMatrix();
  map3d.renderer.setSize(w, h);
}

window.addEventListener("resize", onMapResize);

// --- WebSocket ---
async function loadCameras() {
  const res = await fetch("/api/cameras");
  const data = await res.json();
  cameraSelect.innerHTML = "";
  for (const cam of data.cameras) {
    const opt = document.createElement("option");
    opt.value = cam.index;
    opt.textContent = cam.is_iphone ? `${cam.name} (iPhone)` : cam.name;
    cameraSelect.appendChild(opt);
  }
  if (data.iphone_index !== null) {
    cameraSelect.value = String(data.iphone_index);
  }
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (ev) => {
    const state = JSON.parse(ev.data);
    if (state.speed_options) speedOptions = state.speed_options;

    const p = state.panels;
    setImg(imgOriginal, p.original?.image);
    setImg(imgYolo, p.yolo?.image);
    setImg(imgOrb, p.orb?.image);

    if (state.pointcloud) {
      updateMap3D(state.pointcloud);
    }

    const pb = state.playback;
    statusEl.textContent = pb.error || `Source: ${pb.source || "—"}`;
    btnPlay.textContent = pb.playing ? "Pause" : "Play";
    speedLabel.textContent = `Speed: ${speedOptions[pb.speed_index ?? 3].toFixed(2)}x`;
    timeLabel.textContent = `${formatTime(pb.current_sec)} / ${formatTime(pb.total_sec)}`;

    const isFile = pb.total_sec > 0;
    scrubber.disabled = !isFile;
    if (!scrubbing && isFile) {
      scrubber.value = Math.round((pb.current_sec / pb.total_sec) * 1000);
    }
  };

  ws.onclose = () => setTimeout(connectWs, 1000);
}

$("btn-upload").onclick = () => $("file-input").click();

$("file-input").onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  statusEl.textContent = "Uploading...";
  try {
    const res = await fetch("/api/source/file", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    await res.json();
    if (map3d) map3d.autoCentered = false;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
  e.target.value = "";
};

$("btn-webcam").onclick = () => { api("/api/source/webcam", { index: 0 }); if (map3d) map3d.autoCentered = false; };
$("btn-continuity").onclick = () => { api("/api/source/continuity", { index: Number(cameraSelect.value) }); if (map3d) map3d.autoCentered = false; };
cameraSelect.onchange = () => api("/api/source/continuity", { index: Number(cameraSelect.value) });

btnPlay.onclick = () => api("/api/playback/toggle");

speedSlider.oninput = async () => {
  const idx = Number(speedSlider.value);
  speedLabel.textContent = `Speed: ${speedOptions[idx].toFixed(2)}x`;
  await api("/api/playback/speed", { index: idx });
};

scrubber.onmousedown = () => { scrubbing = true; };
scrubber.onmouseup = async () => {
  scrubbing = false;
  await api("/api/playback/seek", { fraction: Number(scrubber.value) / 1000 });
  if (map3d) map3d.autoCentered = false;
};

loopCheck.onchange = () => api("/api/playback/loop", { enabled: loopCheck.checked });

document.addEventListener("keydown", (e) => {
  if (e.key === "y" || e.key === "Y") api("/api/toggle/yolo");
  if (e.key === "o" || e.key === "O") api("/api/toggle/orb");
});

loadCameras();
connectWs();
