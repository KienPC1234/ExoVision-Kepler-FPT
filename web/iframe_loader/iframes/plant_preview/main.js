import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// capability check
if (typeof THREE === 'undefined' || typeof OrbitControls === 'undefined') {
  const capWarning = document.getElementById('capWarning');
  capWarning.style.display = 'block';
  capWarning.textContent = 'Three.js or OrbitControls chưa được tải đúng.';
  throw new Error('Missing THREE or OrbitControls import');
}

const capWarning = document.getElementById('capWarning');
function showError(msg) { capWarning.style.display = 'block'; capWarning.textContent = msg; }
function hideError() { capWarning.style.display = 'none'; capWarning.textContent = ''; }

let isRealistic = false;
const REAL_AU_UNITS = 100;
const REAL_R_SUN_AU = 0.00465;
const REAL_R_J_AU = 0.000467;
const REAL_PLANET_MAG = 100;
const REAL_STAR_MAG = 100;
let currentOrbitalScale = 5;
let currentOrbitalRadius = 5;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function tempToHSL(eqTemp) {
  const t = clamp((eqTemp - 50) / (2000 - 50), 0, 1);
  const hue = (1 - t) * 210;
  const sat = 20 + Math.abs(eqTemp - 300) / 300 * 70;
  const light = clamp(40 + t * 20, 30, 70);
  return `hsl(${Math.round(hue)}, ${Math.round(sat)}%, ${Math.round(light)}%)`;
}
function hslToThreeColor(hsl) { return new THREE.Color(hsl); }

function blackBodyColor(temp) {
  let r, g, b;
  temp = temp / 100;
  if (temp <= 66) {
    r = 255;
  } else {
    r = temp - 60;
    r = 329.698727446 * Math.pow(r, -0.1332047592);
  }
  r = clamp(r, 0, 255);
  if (temp <= 66) {
    g = temp;
    g = 99.4708025861 * Math.log(g) - 161.1195681661;
  } else {
    g = temp - 60;
    g = 288.1221695283 * Math.pow(g, -0.0755148498);
  }
  g = clamp(g, 0, 255);
  if (temp >= 66) {
    b = 255;
  } else {
    if (temp <= 19) {
      b = 0;
    } else {
      b = temp - 10;
      b = 138.5177312231 * Math.log(b) - 305.0447927307;
    }
  }
  b = clamp(b, 0, 255);
  return new THREE.Color(r / 255, g / 255, b / 255);
}

function estimateMassFromRadius(r) {
  if (r < 1.5) return Math.pow(r, 3.7);
  if (r < 4.0) return 2.69 * Math.pow(r, 0.93);
  return 0.486 * Math.pow(r, 1.89);
}

function calculateProxies(params) {
  if ('pl_radj' in params) {
    const rad_j = params.pl_radj || 0;
    if (rad_j > 0) {
      const rad_earth = rad_j * 11.209;
      const mass_earth = estimateMassFromRadius(rad_earth);
      const density_rel = mass_earth / Math.pow(rad_earth, 3);
      params.mass = mass_earth;
      params.density = density_rel * 5.514;
      params.density_proxy = 1 / Math.pow(rad_j, 3);
    } else {
      params.mass = 0;
      params.density = 0;
      params.density_proxy = 0;
    }
  }
  if ('pl_orbper' in params && 'st_teff' in params && 'pl_insol' in params) {
    const per = params.pl_orbper || 0, teff = params.st_teff || 0, insol = params.pl_insol || 1;
    params.habitability_proxy = (per * 0.7) / (teff + 1e-12);
  }
  if ('depth' in params && 'pl_trandur' in params) {
    const dep = params.depth || 0, dur = params.pl_trandur || 0;
    params.transit_shape_proxy = dep / (dur + 1e-12);
  }
}

function getPlanetClassification(params) {
  const density = params.density || 0;
  const eqt = params.pl_eqt || 300;
  const rad_j = params.pl_radj || 1;
  const rad_earth = rad_j * 11.209;
  if (rad_earth < 0.5) return 'Sub-Earth';
  if (eqt > 400) return 'Hot Lava World';
  if (eqt < 200) return 'Frozen World';
  if (density > 3 && eqt > 200 && eqt < 400) return 'Terrestrial';
  if (density < 2 && rad_earth > 20) return 'Gas Giant';
  if (density < 3 && rad_earth > 10 && rad_earth <= 20) return 'Ice Giant';
  return 'Unknown';
}

// Scene + renderer
const container = document.getElementById('sceneContainer');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth || 800, container.clientHeight || 600);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const cubeTextureLoader = new THREE.CubeTextureLoader();
try {
  const spaceBackground = cubeTextureLoader.load([
    'dark-s_px.jpg', 'dark-s_nx.jpg', 'dark-s_py.jpg', 'dark-s_ny.jpg', 'dark-s_pz.jpg', 'dark-s_nz.jpg'
  ]);
  scene.background = spaceBackground;
} catch (e) { console.warn('Background not loaded', e); }

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
camera.position.set(10, 10, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 0.25));

const textureLoader = new THREE.TextureLoader();

// --- Star: MeshBasic + glow sprite + bright PointLight (no shadows) ---
const starTex = textureLoader.load('2k_sun.jpg');
const starGeom = new THREE.SphereGeometry(0.5, 32, 32);
const starMaterial = new THREE.MeshBasicMaterial({ map: starTex, toneMapped: false });
const starSphere = new THREE.Mesh(starGeom, starMaterial);
starSphere.position.set(0, 0, 0);
scene.add(starSphere);

// glow sprite (additive)
let glowMap = textureLoader.load('glow.png', undefined, undefined, undefined);
if (!glowMap || !glowMap.image) {
  const cvs = document.createElement('canvas'); cvs.width = cvs.height = 128;
  const ctx = cvs.getContext('2d');
  const grd = ctx.createRadialGradient(64, 64, 2, 64, 64, 64);
  grd.addColorStop(0, 'rgba(255,255,220,1)');
  grd.addColorStop(0.2, 'rgba(255,200,120,0.6)');
  grd.addColorStop(1, 'rgba(255,200,120,0)');
  ctx.fillStyle = grd; ctx.fillRect(0, 0, 128, 128);
  glowMap = new THREE.CanvasTexture(cvs);
}
const spriteMat = new THREE.SpriteMaterial({
  map: glowMap, color: 0xffffff, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, toneMapped: false
});
spriteMat.opacity = 0.9;
const starGlow = new THREE.Sprite(spriteMat);
starGlow.scale.set(4, 4, 1); starGlow.position.copy(starSphere.position);
scene.add(starGlow);

// stronger point light with decay so directionality is visible
const starLight = new THREE.PointLight(0xffffff, 4.0, 500, 2); // distance large, decay 2
starLight.position.set(0, 0, 0);
starLight.castShadow = false;
scene.add(starLight);

// gentle rim/backlight
const rim = new THREE.DirectionalLight(0xffffff, 0.12);
rim.position.set(-10, 6, -8);
rim.castShadow = false;
scene.add(rim);

// grid
let grid = new THREE.GridHelper(40, 40, 0x12313a, 0x091217);
grid.position.y = -3.5; grid.material.opacity = 0.06; grid.material.transparent = true;
scene.add(grid);

// planet group
let planetGroup = new THREE.Group(); scene.add(planetGroup);

// connector group
let connectorGroup = null;

// raycaster + tooltip
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

function getUserInfoFromObject(obj) {
  let cur = obj;
  while (cur) {
    if (cur.userData && cur.userData.info) return cur.userData.info;
    cur = cur.parent;
  }
  return null;
}

function onMouseMove(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children, true);
  tooltip.style.display = 'none';

  // default hide inline sprite text
  if (visObjects && visObjects.textSprite) visObjects.textSprite.visible = false;

  if (intersects.length > 0) {
    for (const inter of intersects) {
      const info = getUserInfoFromObject(inter.object);
      if (info) {
        // show DOM tooltip
        tooltip.innerHTML = info;
        const x = Math.min(window.innerWidth - 310, e.clientX + 10);
        const y = Math.min(window.innerHeight - 100, e.clientY + 10);
        tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`; tooltip.style.display = 'block';

        // if hovered object is connector, show inline sprite at midpoint as well
        if (visObjects && visObjects.connectorMesh) {
          let found = false;
          let cur = inter.object;
          while (cur) {
            if (cur === visObjects.connectorMesh) { found = true; break; }
            cur = cur.parent;
          }
          if (found && visObjects.textSprite) {
            // update sprite text if necessary and show
            const distAU = (currentParams && currentParams.pl_orbper) ? visObjects.currentDistanceAU : null;
            if (distAU !== null) {
              updateTextSprite(visObjects.textSprite, `Distance: ${distAU.toFixed(2)} AU`);
            }
            visObjects.textSprite.visible = true;
          }
        }
        break;
      }
    }
  }
}
renderer.domElement.addEventListener('mousemove', onMouseMove);

// resize
function resize() {
  const rect = container.getBoundingClientRect();
  const w = Math.max(rect.width, 200), h = Math.max(rect.height, 200);
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  const panel = container.querySelector('#controlHelp');
  if (panel) { panel.style.right = '12px'; panel.style.bottom = '12px'; }
}
window.addEventListener('resize', resize);
if (window.ResizeObserver) { const ro = new ResizeObserver(resize); ro.observe(container); }

// text sprite function (returns sprite)
function makeTextSprite(message, parameters) {
  if (parameters === undefined) parameters = {};
  const fontface = parameters.fontface || "Arial";
  const fontsize = parameters.fontsize || 18;
  const borderThickness = parameters.borderThickness || 4;
  const borderColor = parameters.borderColor || { r: 0, g: 0, b: 0, a: 1.0 };
  const backgroundColor = parameters.backgroundColor || { r: 255, g: 255, b: 255, a: 0.0 };
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 128;
  const context = canvas.getContext('2d');
  context.font = "Bold " + fontsize + "px " + fontface;
  const metrics = context.measureText(message);
  const textWidth = metrics.width;
  context.clearRect(0, 0, canvas.width, canvas.height);
  // background rounded rect
  context.fillStyle = `rgba(${backgroundColor.r},${backgroundColor.g},${backgroundColor.b},${backgroundColor.a})`;
  context.strokeStyle = `rgba(${borderColor.r},${borderColor.g},${borderColor.b},${borderColor.a})`;
  context.lineWidth = borderThickness;
  // center text
  context.textBaseline = 'middle';
  context.fillStyle = "rgba(255,255,255,1.0)";
  context.fillText(message, 10, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(3, 0.7, 1);
  sprite.renderOrder = 999; // always on top when visible
  sprite.material.depthTest = false;
  return sprite;
}

// helper to update existing sprite's texture (recreate)
function updateTextSprite(sprite, message) {
  if (!sprite) return;
  const parent = sprite.parent;
  const params = { fontsize: 28 };
  const newSprite = makeTextSprite(message, params);
  newSprite.position.copy(sprite.position);
  parent.add(newSprite);
  parent.remove(sprite);
  // replace in visObjects if needed
  visObjects.textSprite = newSprite;
}

// build planet: atmosphere lighter, add axial tilt axis
function buildPlanet(params) {
  while (planetGroup.children.length) planetGroup.remove(planetGroup.children[0]);
  const disposition = params.disposition || 0;
  if (disposition !== 1) return null;

  const insol = params.pl_insol || 1, rStar = params.st_rad || 1, tStar = params.st_teff || 5778, tSun = 5772;
  const aAU = Math.sqrt(rStar * rStar * Math.pow(tStar / tSun, 4) / insol);
  let orbitalRadius;
  if (isRealistic) {
    orbitalRadius = clamp(aAU * REAL_AU_UNITS, 1, 200);
  } else {
    orbitalRadius = clamp(aAU * currentOrbitalScale, 2, 50);
  }

  // orbit ring
  const orbitGeo = new THREE.RingGeometry(orbitalRadius - 0.05, orbitalRadius + 0.05, 128);
  const orbitMat = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, side: THREE.DoubleSide, transparent: true, opacity: 0.5 });
  const orbit = new THREE.Mesh(orbitGeo, orbitMat);
  orbit.rotation.x = Math.PI / 2;
  orbit.userData.info = `<h3>Orbit</h3><p>Radius: ${orbitalRadius.toFixed(2)} units</p><p>Period: ${params.pl_orbper || 'N/A'} days</p>`;
  planetGroup.add(orbit);
  orbit.rotation.z = (1 - (params.koi_impact || 0)) * Math.PI / 2;

  let visualRadius;
  if (isRealistic) {
    visualRadius = clamp((params.pl_radj || 1) * REAL_R_J_AU * REAL_AU_UNITS * REAL_PLANET_MAG, 0.01, 5);
  } else {
    visualRadius = clamp((params.pl_radj || 1) * 0.45, 0.2, 3.0);
  }
  const color = hslToThreeColor(tempToHSL(params.pl_eqt || 300));

  // choose texture
  const classification = getPlanetClassification(params);
  let planetTextureUrl;
  switch (classification) {
    case 'Terrestrial': planetTextureUrl = '2k_earth_daymap.jpg'; break;
    case 'Gas Giant': planetTextureUrl = '2k_jupiter.jpg'; break;
    case 'Ice Giant': planetTextureUrl = '2k_neptune.jpg'; break;
    case 'Hot Lava World': planetTextureUrl = '2k_venus_surface.jpg'; break;
    case 'Frozen World': planetTextureUrl = '2k_moon.jpg'; break;
    case 'Sub-Earth': planetTextureUrl = '2k_mars.jpg'; break;
    default: planetTextureUrl = '2k_jupiter.jpg';
  }
  const planetTexture = textureLoader.load(planetTextureUrl);

  // planet material: slightly shinier to show star highlights better
  const planetMat = new THREE.MeshStandardMaterial({
    map: planetTexture, color: 0xffffff, metalness: 0.03, roughness: 0.45, emissive: 0x222222, emissiveIntensity: 0.15
  });

  const geom = new THREE.SphereGeometry(visualRadius, 64, 64);
  const planet = new THREE.Mesh(geom, planetMat);
  planet.castShadow = true; planet.receiveShadow = true;
  planet.userData.info = `<h3>Planet</h3>
    <p>Classification: ${classification}</p>
    <p>Radius: ${params.pl_radj || 'N/A'} R_J</p>
    <p>Eq Temp: ${params.pl_eqt || 'N/A'} K</p>
    <p>Insolation: ${params.pl_insol || 'N/A'} F_Earth</p>
    <p>Mass: ${params.mass?.toFixed(2) || 'N/A'} M⊕</p>
    <p>Density: ${params.density?.toFixed(2) || 'N/A'} g/cm³</p>
    <p>Density Proxy: ${params.density_proxy?.toFixed(2) || 'N/A'}</p>
    <p>Habitability Proxy: ${params.habitability_proxy?.toFixed(2) || 'N/A'}</p>
    <p>Orbital Period: ${params.pl_orbper || 'N/A'} days</p>
    <p>Transit Duration: ${params.pl_trandur || 'N/A'} hours</p>
    <p>Depth: ${params.depth || 'N/A'} (fraction)</p>
    <p>Impact Parameter: ${params.koi_impact || 'N/A'}</p>
    <p>Disposition: ${params.disposition || 'N/A'}</p>
    <p>Transit Shape Proxy: ${params.transit_shape_proxy?.toFixed(2) || 'N/A'}</p>`;
  planet.userData.orbitalRadius = orbitalRadius;
  planet.renderOrder = 1; // draw planet after connector (helps visibility)

  // clouds
  const cloudGeom = new THREE.SphereGeometry(visualRadius * 1.02, 48, 48);
  const cloudMat = new THREE.MeshLambertMaterial({
    color: 0xffffff, transparent: true, opacity: 0.05,
    depthWrite: false
  });
  const clouds = new THREE.Mesh(cloudGeom, cloudMat);
  clouds.userData.info = planet.userData.info;

  // atmosphere
  const atmScale = 1 + (params.transit_shape_proxy || 0) * 0.5;
  const atmGeom = new THREE.SphereGeometry(visualRadius * 1.07 * atmScale, 48, 48);
  const atmosphereColor = color.clone().lerp(new THREE.Color(0xffffff), 0.85);
  const atmMat = new THREE.MeshBasicMaterial({
    color: atmosphereColor,
    transparent: true,
    opacity: 0.05,
    blending: THREE.AdditiveBlending,
    side: THREE.FrontSide,
    depthWrite: false
  });
  const atmosphere = new THREE.Mesh(atmGeom, atmMat);
  atmosphere.userData.info = planet.userData.info;

  // tilt & spin groups
  const tiltGroup = new THREE.Group();
  tiltGroup.position.set(orbitalRadius, 0, 0);

  const spinGroup = new THREE.Group();
  spinGroup.add(planet);
  spinGroup.add(clouds);
  spinGroup.add(atmosphere);

  // rings (optional)
  let rings = null;
  if ((params.depth && params.depth > 0.02) || (params.koi_impact && params.koi_impact > 0.9)) {
    const ringInner = visualRadius * 1.2;
    const ringOuter = visualRadius * (1.6 + Math.min((params.depth || 0) * 40, 1));
    const ringGeo = new THREE.RingGeometry(ringInner, ringOuter, 128);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xaaaaaa, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
    rings = new THREE.Mesh(ringGeo, ringMat);
    rings.rotation.x = Math.PI / 2; rings.userData.info = planet.userData.info;
    spinGroup.add(rings);
  }

  // AXIS indicators
  const axialTiltDeg = (typeof params.axial_tilt === 'number') ? params.axial_tilt : ((params.koi_impact || 0) * 90);
  const tiltRad = THREE.MathUtils.degToRad(axialTiltDeg || 0);
  const axisGroup = new THREE.Group();
  const axisHalfLength = visualRadius * 1.5;
  const cylGeom = new THREE.CylinderGeometry(0.01 * visualRadius, 0.01 * visualRadius, axisHalfLength, 8);
  const axisMat = new THREE.MeshBasicMaterial({ color: 0xffff66, toneMapped: false, depthTest: false, transparent: true, opacity: 0.95 });

  const northAxis = new THREE.Mesh(cylGeom, axisMat);
  northAxis.position.set(0, visualRadius + axisHalfLength / 2, 0);
  northAxis.renderOrder = 999;
  axisGroup.add(northAxis);

  const southAxis = new THREE.Mesh(cylGeom, axisMat);
  southAxis.position.set(0, -visualRadius - axisHalfLength / 2, 0);
  southAxis.renderOrder = 999;
  axisGroup.add(southAxis);

  const coneGeom = new THREE.ConeGeometry(0.04 * visualRadius, 0.12 * visualRadius, 8);
  const coneMat = new THREE.MeshBasicMaterial({ color: 0xffcc44, depthTest: false, toneMapped: false });
  const coneNorth = new THREE.Mesh(coneGeom, coneMat);
  coneNorth.position.set(0, visualRadius + axisHalfLength + 0.06 * visualRadius, 0);
  axisGroup.add(coneNorth);
  const coneSouth = new THREE.Mesh(coneGeom, coneMat);
  coneSouth.position.set(0, -visualRadius - axisHalfLength - 0.06 * visualRadius, 0);
  coneSouth.rotation.x = Math.PI;
  axisGroup.add(coneSouth);

  spinGroup.add(axisGroup);
  axisGroup.userData = { info: `<h3>Axial Tilt</h3><p>Tilt: ${axialTiltDeg.toFixed(1)}°</p>` };
  northAxis.userData.info = axisGroup.userData.info;
  southAxis.userData.info = axisGroup.userData.info;
  coneNorth.userData.info = axisGroup.userData.info;
  coneSouth.userData.info = axisGroup.userData.info;

  tiltGroup.rotation.z = tiltRad;
  tiltGroup.add(spinGroup);

  clouds.userData.info = planet.userData.info;
  atmosphere.userData.info = planet.userData.info;
  if (rings) rings.userData.info = planet.userData.info;

  planetGroup.add(tiltGroup);

  // Connector group - create a cylinder mesh (thicker, pickable) and arrow cones
  if (connectorGroup) scene.remove(connectorGroup);
  connectorGroup = new THREE.Group();
  scene.add(connectorGroup);

  // unit cylinder geometry (height 1) - we will scale Y to length
  const connGeom = new THREE.CylinderGeometry(1, 1, 1, 16, 1, true);
  // material: emissive based on star color, semi-transparent so doesn't fully block planet texture
  const connMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: new THREE.Color(0x8888ff),
    emissiveIntensity: 0.8,
    transparent: true,
    opacity: 0.75,
    roughness: 0.35,
    metalness: 0.1,
    depthWrite: false // avoid writing depth to reduce occlusion of planet texture
  });
  const connectorMesh = new THREE.Mesh(connGeom, connMat);
  connectorMesh.userData = { info: `<h3>Distance</h3><p>${aAU.toFixed(2)} AU</p>` };
  // make it selectable with raycaster
  connectorMesh.name = 'connectorMesh';
  connectorMesh.renderOrder = 0;
  connectorGroup.add(connectorMesh);

  // small cones at both ends as visual arrowheads
  const coneSmall = new THREE.ConeGeometry(0.3, 0.6, 12);
  const coneMatSmall = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6, transparent: true, opacity: 0.9 });
  const coneStart = new THREE.Mesh(coneSmall, coneMatSmall);
  const coneEnd = new THREE.Mesh(coneSmall, coneMatSmall);
  connectorGroup.add(coneStart, coneEnd);

  // text sprite (hidden by default) - updated on hover
  const textSprite = makeTextSprite(`Distance: ${aAU.toFixed(2)} AU`, { fontsize: 24 });
  textSprite.visible = false; // only show on hover
  connectorGroup.add(textSprite);

  // store some helper refs
  const result = { planet, clouds, atmosphere, rings, orbit, axisGroup, tiltGroup, spinGroup, connectorMesh, coneStart, coneEnd, textSprite };
  result.currentDistanceAU = aAU;
  result.orbitalRadius = orbitalRadius;
  return result;
}

// UI & table (unchanged)
const tableBody = document.querySelector('#infoTable tbody');
const unitsMap = {
  koi_kepmag: 'mag',
  pl_radj: 'R_J',
  koi_impact: '',
  pl_trandur: 'hours',
  depth: '',
  pl_orbper: 'days',
  st_teff: 'K',
  st_logg: 'dex',
  st_rad: 'R_Sun',
  pl_insol: 'F_Earth',
  pl_eqt: 'K',
  st_dist: 'pc',
  disposition: '',
  habitability_proxy: 'days/K',
  transit_shape_proxy: 'fraction/hr',
  mass: 'M⊕',
  density: 'g/cm³',
  density_proxy: ''
};

// --- Thay thế populateTable (bảng có nhãn dễ đọc + header vertical) ---
const friendlyLabels = {
  koi_kepmag: 'KepMag',
  pl_radj: 'Planet radius (R_J)',
  koi_impact: 'Impact parameter',
  pl_trandur: 'Transit duration (hours)',
  depth: 'Depth (fraction)',
  pl_orbper: 'Orbital period (days)',
  st_teff: 'Stellar Teff (K)',
  st_logg: 'Stellar logg',
  st_rad: 'Stellar radius (R_Sun)',
  pl_insol: 'Insolation (F_Earth)',
  pl_eqt: 'Equilibrium temp (K)',
  st_dist: 'Distance (pc)',
  disposition: 'Disposition',
  habitability_proxy: 'Habitability',
  transit_shape_proxy: 'Transit shape',
  mass: 'Mass (M⊕)',
  density: 'Density (g/cm³)',
  density_proxy: 'Density proxy',
  axial_tilt: 'Axial tilt (°)'
};

function populateTable(params) {
  const table = document.getElementById('infoTable');
  // tạo/ghi lại header (vertical Field)
  let thead = table.querySelector('thead');
  if (thead) thead.remove();
  thead = document.createElement('thead');
  const htr = document.createElement('tr');
  const thField = document.createElement('th');
  thField.innerHTML = 'Field';
  thField.style.textAlign = 'center';
  thField.style.padding = '6px';
  thField.style.minWidth = '28px';
  const thValue = document.createElement('th');
  thValue.textContent = 'Value';
  thValue.style.textAlign = 'left';
  thValue.style.padding = '6px';
  htr.appendChild(thField);
  htr.appendChild(thValue);
  thead.appendChild(htr);
  table.appendChild(thead);

  // xóa body trước đó
  tableBody.innerHTML = '';
  const keys = Object.keys(params).sort();
  for (const k of keys) {
    const unit = unitsMap[k] || '';
    const label = friendlyLabels[k] || k;
    const tr = document.createElement('tr');

    const td1 = document.createElement('td');
    td1.textContent = label;
    td1.title = k; // tooltip hiện tên biến gốc nếu cần
    td1.style.fontWeight = '600';
    td1.style.whiteSpace = 'nowrap';
    td1.style.padding = '6px 10px';

    const td2 = document.createElement('td');
    td2.textContent = `${params[k] ?? 'undefined'}${unit ? ' ' + unit : ''}`;
    td2.style.padding = '6px 10px';

    tr.appendChild(td1); tr.appendChild(td2);
    tableBody.appendChild(tr);
  }

  // một chút style gọn cho table (đảm bảo tồn tại)
  table.style.borderCollapse = 'collapse';
  table.querySelectorAll('td, th').forEach(el => {
    el.style.borderBottom = '1px solid rgba(255,255,255,0.06)';
  });
}




// DOM refs
const jsonInput = document.getElementById('jsonInput');
const btnUpdate = document.getElementById('btnUpdate');
const btnReset = document.getElementById('btnReset');
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
speedSlider.addEventListener('input', () => { speedValue.textContent = speedSlider.value; });

// Realistic toggle
let realisticToggle = document.getElementById('realisticToggle');
if (!realisticToggle) {
  const realisticDiv = document.createElement('div');
  realisticDiv.style.marginTop = '10px';
  realisticDiv.innerHTML = '<label for="realisticToggle" style="color: #fff; font-size: 14px;"><input type="checkbox" id="realisticToggle"> Realistic Scale</label>';
  const parentControl = speedSlider.parentNode;
  parentControl.appendChild(realisticDiv);
  realisticToggle = document.getElementById('realisticToggle');
}
realisticToggle.addEventListener('change', () => {
  isRealistic = realisticToggle.checked;
  if (currentParams) updateFromParams(currentParams);
});

let currentParams = null;
let visObjects = null;

function safeParseJSON(text) { try { return JSON.parse(text); } catch (e) { alert('JSON không hợp lệ: ' + e.message); return null; } }

function updateFromParams(params) {
  if (!params) return;
  hideError();
  calculateProxies(params);
  currentParams = params;
  populateTable(params);

  let orbitalScale = 5;
  let starScaleVal = (params.st_rad || 1) * 3;
  if (isRealistic) {
    orbitalScale = REAL_AU_UNITS;
    starScaleVal = (params.st_rad || 1) * REAL_R_SUN_AU * REAL_AU_UNITS * REAL_STAR_MAG;
  }
  currentOrbitalScale = orbitalScale;

  // star scale & glow
  starSphere.scale.setScalar(starScaleVal);
  starGlow.scale.setScalar(3 * starScaleVal);

  // star light intensity/color tweaks
  const magIntensity = clamp(2 - (params.koi_kepmag || 12.5) / 10, 0.5, 5);
  starLight.intensity = magIntensity * 3.5 * (1 + (params.st_teff || 5778) / 9000);
  const starColor = blackBodyColor(params.st_teff || 5778);
  starMaterial.color.copy(starColor);
  spriteMat.color.copy(starColor);
  spriteMat.opacity = 0.55;
  starLight.color.copy(starColor);
  starLight.decay = 2;
  starLight.distance = 1000;

  starSphere.userData.info = `<h3>Host Star</h3>
    <p>Radius: ${params.st_rad || 'N/A'} R_Sun</p>
    <p>Teff: ${params.st_teff || 'N/A'} K</p>
    <p>Logg: ${params.st_logg || 'N/A'} dex</p>
    <p>Kepmag: ${params.koi_kepmag || 'N/A'} mag</p>
    <p>Distance: ${params.st_dist || 'N/A'} pc</p>`;
  starGlow.userData.info = starSphere.userData.info;

  visObjects = buildPlanet(params);
  if (visObjects) {
    currentOrbitalRadius = visObjects.orbitalRadius;

    // update grid
    scene.remove(grid);
    const gridSize = Math.max(40, currentOrbitalRadius * 2);
    const gridDivisions = Math.max(20, Math.round(gridSize / 2));
    grid = new THREE.GridHelper(gridSize, gridDivisions, 0x12313a, 0x091217);
    grid.position.y = -Math.max(3.5, starScaleVal * 0.5);
    grid.material.opacity = 0.06;
    grid.material.transparent = true;
    scene.add(grid);

    // update camera position
    camera.position.set(currentOrbitalRadius * 1.2, currentOrbitalRadius * 0.8, currentOrbitalRadius * 1.5);
    controls.target.set(0, 0, 0);
    controls.update();
  }
}

btnUpdate.addEventListener('click', () => { const parsed = safeParseJSON(jsonInput.value); if (parsed) updateFromParams(parsed); });
btnReset.addEventListener('click', () => {
  const defaultStr = `{
"koi_kepmag":12.5,"pl_radj":1.0,"koi_impact":0.5,"pl_trandur":10.5,"depth":0.01,"pl_orbper":365.25,
"st_teff":5778,"st_logg":4.44,"st_rad":1.0,"pl_insol":1.0,"pl_eqt":288,"st_dist":100,"disposition":1
}`;
  jsonInput.value = defaultStr; btnUpdate.click();
});

// animation
let last = performance.now();
function animate(t) {
  requestAnimationFrame(animate);
  const dt = (t - last) * 0.001 || 0.016;
  last = t;

  const speed = parseFloat(speedSlider.value) || 1;
  if (visObjects && visObjects.tiltGroup && currentParams) {
    // orbital motion (avoid divide by zero)
    const orbitalPeriod = (currentParams.pl_orbper || 365.25);
    const orbitalPeriodSec = Math.max(0.000001, orbitalPeriod / speed);
    const angle = (t / 1000) / orbitalPeriodSec * 2 * Math.PI;
    const orbRad = visObjects.planet.userData.orbitalRadius;
    visObjects.tiltGroup.position.x = Math.cos(angle) * orbRad;
    visObjects.tiltGroup.position.z = Math.sin(angle) * orbRad;

    // rotate around tilted axis scaled by speed so spin up with orbital speed
    const baseSpin = 0.2; // base spin rate
    visObjects.spinGroup.rotation.y += baseSpin * dt * speed;
    visObjects.clouds.rotation.y += 0.15 * dt * speed;
    if (visObjects.rings) visObjects.rings.rotation.z += 0.05 * dt;

    // Update connector (cylinder) to connect star (0,0,0) to planet
    if (connectorGroup && visObjects.connectorMesh) {
      const starPos = new THREE.Vector3(0, 0, 0);
      const planetPos = visObjects.tiltGroup.position.clone();
      const dir = planetPos.clone().sub(starPos);
      const length = dir.length();
      const dirNorm = dir.clone().normalize();

      // --- nhỏ hơn: radius tỷ lệ với khoảng cách nhưng rất nhỏ ---
      const radius = clamp(length * 0.005, 0.005, 0.06); // <-- giảm tỉ lệ xuống 0.005, cap max 0.06
      const connector = visObjects.connectorMesh;
      // connector geometry dùng CylinderGeometry(1,1,1) => scale Y thành length, X/Z thành radius
      connector.scale.set(radius, length, radius);
      const midpoint = starPos.clone().add(planetPos).multiplyScalar(0.5);
      connector.position.copy(midpoint);
      connector.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirNorm);

      // cones thu nhỏ tương ứng
      visObjects.coneStart.position.copy(starPos);
      visObjects.coneStart.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirNorm);
      visObjects.coneStart.position.add(dirNorm.clone().multiplyScalar(radius * 1.2));
      visObjects.coneStart.scale.setScalar(0.6 * radius * 3);

      visObjects.coneEnd.position.copy(planetPos);
      visObjects.coneEnd.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirNorm);
      visObjects.coneEnd.position.add(dirNorm.clone().multiplyScalar(-radius * 1.2));
      visObjects.coneEnd.scale.setScalar(0.6 * radius * 3);

      // update text sprite vị trí
      visObjects.textSprite.position.copy(midpoint.clone().add(new THREE.Vector3(0, Math.max(0.3, radius * 6), 0)));
      visObjects.textSprite.lookAt(camera.position);

      // cập nhật stored distance
      const distAU = planetPos.length() / currentOrbitalScale;
      visObjects.currentDistanceAU = distAU;
      connector.userData.info = `<h3>Distance</h3><p>${distAU.toFixed(2)} AU</p>`;

      // nhẹ tint emissive theo màu sao (nếu muốn)
      if (connector.material && starMaterial.color) {
        connector.material.emissive.copy(starMaterial.color).multiplyScalar(0.35);
      }
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

// create help panel inside render container (unchanged)
function createHelpPanelInsideContainer() {
  const old = document.getElementById('controlHelp');
  if (old) old.remove();
  
  const panel = document.createElement('div');
  panel.id = 'controlHelp';
  panel.style.position = 'absolute';
  panel.style.right = '12px';
  panel.style.bottom = '12px';
  panel.style.background = 'rgba(0,0,0,0.6)';
  panel.style.color = '#fff';
  panel.style.padding = '10px';
  panel.style.borderRadius = '8px';
  panel.style.zIndex = 50;
  panel.style.pointerEvents = 'auto';
  panel.style.transition = 'width 0.3s ease, opacity 0.3s ease';
  
  // Tính toán width động dựa trên diện tích container
  const containerWidth = container.offsetWidth || window.innerWidth;
  const dynamicWidth = Math.min(300, containerWidth * 0.8);
  
  panel.style.width = dynamicWidth + 'px';
  panel.style.minWidth = '60px'; // Chiều rộng tối thiểu cho nút khi ẩn
  
  panel.innerHTML = `
    <div id="helpHeader" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <strong style="display:none;">Controls & Hover</strong>
      <button id="toggleHelp" style="background:rgba(255,255,255,0.12);color:#fff;border:0;padding:4px 8px;border-radius:6px;cursor:pointer;min-width:40px;">Hide</button>
    </div>
    <div id="helpBody" style="padding-left:18px;margin:8px 0;display:block;">
      <ul style="padding-left:18px;margin:8px 0;">
        <li>Left click: rotate / drag</li>
        <li>Right click + drag: pan</li>
        <li>Scroll: zoom</li>
        <li>Speed slider: adjust speed</li>
      </ul>
      <div style="font-size:12px;opacity:0.95">
        If you hover over a planet and don't see any info — it might be obscured by clouds/atmosphere; hover will climb the parent chain to find info. Try hiding the panel if it's blocking the view.
      </div>
    </div>
  `;

  
  container.style.position = container.style.position || 'relative';
  container.appendChild(panel);
  
  // Xử lý toggle
  const toggleHelp = document.getElementById('toggleHelp');
  const helpHeader = document.getElementById('helpHeader');
  const helpBody = document.getElementById('helpBody');
  const title = helpHeader.querySelector('strong');
  
  toggleHelp.addEventListener('click', (e) => {
    const btn = e.target;
    if (btn.textContent === 'Ẩn') {
      btn.textContent = 'Hiện';
      helpBody.style.display = 'none';
      title.style.display = 'none';
      panel.style.width = '60px';
      panel.style.opacity = '0.9';
      panel.style.pointerEvents = 'auto'; // Vẫn cho phép click nút
    } else {
      btn.textContent = 'Ẩn';
      helpBody.style.display = 'block';
      title.style.display = 'block';
      panel.style.width = dynamicWidth + 'px';
      panel.style.opacity = '1';
      panel.style.pointerEvents = 'auto';
    }
  });
  
  // Tùy chọn: Scale lại khi resize window
  window.addEventListener('resize', () => {
    if (document.getElementById('controlHelp') && helpBody.style.display !== 'none') {
      const newContainerWidth = container.offsetWidth || window.innerWidth;
      const newDynamicWidth = Math.min(300, newContainerWidth * 0.8);
      panel.style.width = newDynamicWidth + 'px';
    }
  });
}

// init
function init() {
  resize(); createHelpPanelInsideContainer();
  document.getElementById('btnReset').click();
  animate(performance.now());
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();