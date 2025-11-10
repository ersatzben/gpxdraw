import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const CONFIG = {
  horizontalScale: 0.02,
  verticalExaggeration: 6.0,
  tubeRadius: 32,
  tubeSegments: 1200,
  tubeRadialSegments: 14,
  curtainSegments: 700,
  syntheticElevationAmplitudeMeters: 45,
};

const overlayStatus = document.getElementById("elevation-status");
const setStatus = (message, isError = false) => {
  overlayStatus.textContent = message;
  overlayStatus.style.color = isError ? "#ff8080" : "#9aa8c1";
};

if (window.location.protocol === "file:") {
  setStatus(
    "Serve this folder over HTTP so the GPX can be fetched (e.g. `python3 -m http.server`).",
    true
  );
  throw new Error("GPX fetch is blocked when running from file://");
}
const scene = new THREE.Scene();
scene.background = new THREE.Color("#05060b");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputEncoding = THREE.sRGBEncoding;
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(
  48,
  window.innerWidth / window.innerHeight,
  0.1,
  1_000_000
);
camera.position.set(0, 520, 1120);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.screenSpacePanning = false;
controls.minDistance = 60;
controls.maxDistance = 9000;
controls.maxPolarAngle = Math.PI * 0.49;

const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(420, 520, 180);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x6db3ff, 0.45);
rimLight.position.set(-360, 280, -480);
scene.add(rimLight);

const hemi = new THREE.HemisphereLight(0x4b6da1, 0x07090d, 0.22);
scene.add(hemi);

let animationFrame = 0;

init().catch((error) => {
  console.error("Failed to initialise visualisation", error);
  setStatus("Failed to load GPX (see console).", true);
});

async function init() {
  const rawPoints = await loadGPX("./route19668100.gpx");
  const processed = preparePoints(rawPoints);
  const statusDetails =
    processed.elevationMode === "gpx"
      ? "Elevation from GPX."
      : "GPX lacks <ele>; synthetic elevation.";
  setStatus(`${processed.sampledPoints.length} samples loaded. ${statusDetails}`);

  const group = buildRouteGroup(processed);
  scene.add(group);

  addGroundPlane(processed.bounds, processed.baseY);
  addEnvironmentFog();

  fitCameraToObject(camera, controls, group, 1.32);
  animate();
}

function animate() {
  animationFrame = requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function stop() {
  cancelAnimationFrame(animationFrame);
}

async function loadGPX(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `GPX fetch failed: ${response.status} ${response.statusText}`
    );
  }

  const text = await response.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(text, "application/xml");

  const parserError = doc.querySelector("parsererror");
  if (parserError) {
    throw new Error(parserError.textContent ?? "Unable to parse GPX file.");
  }

  const points = Array.from(doc.querySelectorAll("trkpt")).map((node) => {
    const lat = parseFloat(node.getAttribute("lat") ?? "NaN");
    const lon = parseFloat(node.getAttribute("lon") ?? "NaN");
    const eleNode = node.querySelector("ele");
    const ele = eleNode ? parseFloat(eleNode.textContent ?? "NaN") : null;
    return { lat, lon, ele };
  });

  if (!points.length) {
    throw new Error("GPX contains no track points.");
  }

  return points.filter(
    (pt) =>
      Number.isFinite(pt.lat) &&
      Number.isFinite(pt.lon) &&
      Math.abs(pt.lat) <= 90 &&
      Math.abs(pt.lon) <= 180
  );
}

function preparePoints(trackPoints) {
  const filtered = dedupeSequential(trackPoints);

  const lat0 = filtered[0].lat;
  const lon0 = filtered[0].lon;
  const metersPerDegreeLat = 111_320;
  const metersPerDegreeLon = metersPerDegreeLat * Math.cos(THREE.MathUtils.degToRad(lat0));

  const rawElevations = filtered.map((pt) =>
    Number.isFinite(pt.ele) ? pt.ele : null
  );
  const hasElevation = rawElevations.some((ele) => ele !== null);

  const planarPoints = filtered.map((pt) => {
    const x = (pt.lon - lon0) * metersPerDegreeLon;
    const z = -(pt.lat - lat0) * metersPerDegreeLat;
    return new THREE.Vector3(x, 0, z);
  });

  const elevations = hasElevation
    ? rawElevations.map((ele) => ele ?? 0)
    : synthesiseElevation(planarPoints);

  const elevationMode = hasElevation ? "gpx" : "synthetic";

  const minElevation = Math.min(...elevations);
  const maxElevation = Math.max(...elevations);
  const elevationRange = Math.max(maxElevation - minElevation, 1);

  const scaledPoints = planarPoints.map((point, index) => {
    const normalizedElevation =
      (elevations[index] - minElevation) / elevationRange;
    const y = normalizedElevation * CONFIG.verticalExaggeration * elevationRange;

    return new THREE.Vector3(
      point.x * CONFIG.horizontalScale,
      y,
      point.z * CONFIG.horizontalScale
    );
  });

  const curve = new THREE.CatmullRomCurve3(
    scaledPoints,
    false,
    "centripetal",
    0.55
  );

  const sampled = sampleCurve(curve, CONFIG.tubeSegments);
  const bounds = computeBounds(sampled);

  return {
    curve,
    sampledPoints: sampled,
    bounds,
    baseY: bounds.minY - Math.max(bounds.size.y * 0.12, 40),
    elevationMode,
    elevationRange: { min: minElevation, max: maxElevation },
  };
}

function dedupeSequential(points, epsilon = 1e-9) {
  if (points.length <= 1) {
    return points.slice();
  }

  const result = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const prev = result[result.length - 1];
    const current = points[i];
    if (
      Math.abs(prev.lat - current.lat) > epsilon ||
      Math.abs(prev.lon - current.lon) > epsilon
    ) {
      result.push(current);
    }
  }

  return result;
}

function synthesiseElevation(points) {
  const distances = [0];
  for (let i = 1; i < points.length; i += 1) {
    const d = points[i].distanceTo(points[i - 1]);
    distances[i] = distances[i - 1] + d;
  }

  const total = distances[distances.length - 1] || 1;
  const amplitude = Math.max(
    CONFIG.syntheticElevationAmplitudeMeters,
    total * 0.015
  );

  const curvature = computeCurvature(points);

  return distances.map((distance, index) => {
    const t = distance / total;
    const base = Math.sin(t * Math.PI * 2) * amplitude * 0.4;
    const detail = Math.sin(t * Math.PI * 8) * amplitude * 0.18;
    const bend = curvature[index] * amplitude * 1.35;
    return base + detail + bend;
  });
}

function computeCurvature(points) {
  const result = new Array(points.length).fill(0);
  for (let i = 1; i < points.length - 1; i += 1) {
    const prev = points[i - 1];
    const current = points[i];
    const next = points[i + 1];

    const v1 = prev.clone().sub(current).normalize();
    const v2 = next.clone().sub(current).normalize();
    const angle = v1.angleTo(v2);
    result[i] = angle;
  }
  return smoothArray(result, 8);
}

function smoothArray(values, windowSize) {
  const half = Math.max(1, Math.floor(windowSize / 2));
  const smoothed = new Array(values.length).fill(0);

  for (let i = 0; i < values.length; i += 1) {
    let acc = 0;
    let count = 0;
    for (let offset = -half; offset <= half; offset += 1) {
      const idx = i + offset;
      if (idx >= 0 && idx < values.length) {
        acc += values[idx];
        count += 1;
      }
    }
    smoothed[i] = acc / count;
  }

  return smoothed;
}

function sampleCurve(curve, segments) {
  const points = [];
  for (let i = 0; i <= segments; i += 1) {
    points.push(curve.getPoint(i / segments));
  }
  return points;
}

function computeBounds(points) {
  const min = new THREE.Vector3(+Infinity, +Infinity, +Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

  points.forEach((pt) => {
    min.min(pt);
    max.max(pt);
  });

  const size = new THREE.Vector3().subVectors(max, min);
  const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);

  return {
    min,
    max,
    size,
    center,
    minY: min.y,
    maxY: max.y,
  };
}

function buildRouteGroup(processed) {
  const group = new THREE.Group();
  group.name = "RouteGroup";

  const { curve, sampledPoints, bounds } = processed;
  const [lowColor, highColor] = [new THREE.Color("#49c6ff"), new THREE.Color("#ff6b6b")];

  const tubeGeometry = new THREE.TubeGeometry(
    curve,
    CONFIG.tubeSegments,
    CONFIG.tubeRadius,
    CONFIG.tubeRadialSegments,
    false
  );

  applyElevationVertexColors(
    tubeGeometry,
    bounds.minY,
    bounds.maxY,
    lowColor,
    highColor
  );

  const tubeMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.05,
    roughness: 0.28,
    envMapIntensity: 0.3,
  });

  const tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
  tubeMesh.castShadow = true;
  tubeMesh.receiveShadow = true;
  tubeMesh.name = "RouteRibbon";
  group.add(tubeMesh);

  const glowLine = buildGlowLine(sampledPoints, lowColor, highColor, bounds);
  group.add(glowLine);

  const curtain = buildCurtainMesh(
    curve,
    bounds.minY,
    processed.baseY,
    lowColor,
    highColor
  );
  group.add(curtain);

  group.position.sub(bounds.center);

  return group;
}

function applyElevationVertexColors(geometry, minY, maxY, lowColor, highColor) {
  const positions = geometry.attributes.position.array;
  const colors = new Float32Array(positions.length);
  const range = Math.max(maxY - minY, 1e-5);

  for (let i = 0; i < positions.length; i += 3) {
    const y = positions[i + 1];
    const t = THREE.MathUtils.clamp((y - minY) / range, 0, 1);
    const color = lowColor.clone().lerp(highColor, t);
    colors[i] = color.r;
    colors[i + 1] = color.g;
    colors[i + 2] = color.b;
  }

  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

function buildGlowLine(points, lowColor, highColor, bounds) {
  const lineGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  const range = Math.max(bounds.maxY - bounds.minY, 1e-5);

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    positions[i * 3] = point.x;
    positions[i * 3 + 1] = point.y;
    positions[i * 3 + 2] = point.z;

    const t = THREE.MathUtils.clamp((point.y - bounds.minY) / range, 0, 1);
    const color = lowColor.clone().lerp(highColor, t);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  lineGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  lineGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const lineMaterial = new THREE.LineBasicMaterial({
    vertexColors: true,
    linewidth: 2,
    transparent: true,
    opacity: 0.95,
  });

  const line = new THREE.Line(lineGeometry, lineMaterial);
  line.name = "RouteGlow";
  return line;
}

function buildCurtainMesh(curve, minY, baseY, lowColor, highColor) {
  const segments = CONFIG.curtainSegments;
  const positions = new Float32Array((segments + 1) * 2 * 3);
  const colors = new Float32Array((segments + 1) * 2 * 3);
  const indices = [];
  const elevationRange = Math.max(minY - baseY, 1);

  for (let i = 0; i <= segments; i += 1) {
    const t = i / segments;
    const top = curve.getPoint(t);
    const bottom = top.clone();
    bottom.y = baseY;

    const colorT = THREE.MathUtils.clamp(
      (top.y - baseY) / elevationRange,
      0,
      1
    );
    const topColor = lowColor.clone().lerp(highColor, colorT);
    const bottomColor = topColor.clone().lerp(new THREE.Color("#0a0d12"), 0.85);

    const topIndex = i * 6;
    const bottomIndex = topIndex + 3;

    positions[topIndex] = top.x;
    positions[topIndex + 1] = top.y;
    positions[topIndex + 2] = top.z;
    colors[topIndex] = topColor.r;
    colors[topIndex + 1] = topColor.g;
    colors[topIndex + 2] = topColor.b;

    positions[bottomIndex] = bottom.x;
    positions[bottomIndex + 1] = bottom.y;
    positions[bottomIndex + 2] = bottom.z;
    colors[bottomIndex] = bottomColor.r;
    colors[bottomIndex + 1] = bottomColor.g;
    colors[bottomIndex + 2] = bottomColor.b;
  }

  for (let i = 0; i < segments; i += 1) {
    const a = i * 2;
    const b = a + 1;
    const c = a + 2;
    const d = a + 3;
    indices.push(a, b, c);
    indices.push(b, d, c);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "ElevationCurtain";
  return mesh;
}

function addGroundPlane(bounds, baseY) {
  const size = Math.max(bounds.size.x, bounds.size.z) * 1.6;
  const geometry = new THREE.CircleGeometry(size, 128);
  const material = new THREE.MeshStandardMaterial({
    color: 0x07090d,
    roughness: 1,
    metalness: 0,
    opacity: 0.55,
    transparent: true,
  });

  const circle = new THREE.Mesh(geometry, material);
  circle.rotation.x = -Math.PI / 2;
  circle.position.y = baseY;
  circle.receiveShadow = true;
  circle.name = "GroundPlane";
  scene.add(circle);
}

function addEnvironmentFog() {
  scene.fog = new THREE.FogExp2(new THREE.Color("#05060b"), 0.00045);
}

function fitCameraToObject(camera, controls, object, padding = 1.25) {
  const boundingBox = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  boundingBox.getSize(size);
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  const fitHeightDistance =
    maxDim / (2 * Math.tan((Math.PI * camera.fov) / 360));
  const fitWidthDistance = fitHeightDistance / camera.aspect;
  const distance = padding * Math.max(fitHeightDistance, fitWidthDistance);

  const direction = controls.target
    .clone()
    .sub(camera.position)
    .normalize()
    .negate();
  camera.position.copy(direction.multiplyScalar(distance).add(center));

  controls.target.copy(center);
  controls.update();
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

window.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stop();
  } else {
    animate();
  }
});

