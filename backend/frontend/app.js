// ── Helpers ───────────────────────────────────────────────────────────────
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function formatLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ── Progress bar (animación simulada) ─────────────────────────────────────
let progressInterval = null;

function startProgress() {
  const fill = document.getElementById('progress-fill');
  fill.style.width = '0%';
  document.getElementById('progress-wrap').style.display = 'block';
  let pct = 0;
  progressInterval = setInterval(() => {
    pct += Math.random() * 9;
    if (pct > 88) pct = 88;
    fill.style.width = pct + '%';
  }, 300);
}

function finishProgress(cb) {
  clearInterval(progressInterval);
  progressInterval = null;
  const fill = document.getElementById('progress-fill');
  fill.style.width = '100%';
  setTimeout(() => {
    document.getElementById('progress-wrap').style.display = 'none';
    fill.style.width = '0%';
    if (cb) cb();
  }, 500);
}

// ── Estado ────────────────────────────────────────────────────────────────
let selectedModel = null;

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {

  // Cargar modelos y crear botones
  const container = document.getElementById('disease-buttons');
  try {
    const r = await fetch('/models');
    const j = await r.json();
    container.innerHTML = '';
    if (j.models && j.models.length) {
      j.models.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'disease-btn';
        btn.textContent = formatLabel(m);
        btn.dataset.model = m;
        btn.addEventListener('click', () => selectDisease(m));
        container.appendChild(btn);
      });
    } else {
      container.innerHTML = '<span class="error-msg">No hay modelos disponibles.</span>';
    }
  } catch {
    container.innerHTML = '<span class="error-msg">Error al cargar modelos.</span>';
  }

  // Preview de imagen al seleccionar archivo
  document.getElementById('file').addEventListener('change', (e) => {
    if (!e.target.files.length) return;
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = ev => {
      const img = document.getElementById('preview-img');
      img.src = ev.target.result;
      document.getElementById('preview-wrap').style.display = 'block';
      document.getElementById('upload-text').textContent = file.name;
    };
    reader.readAsDataURL(file);
    document.getElementById('btn-predict').disabled = false;
  });

  // Botón volver desde pantalla de análisis
  document.getElementById('btn-back-analyze').addEventListener('click', () => {
    resetAnalyze();
    show('screen-preload');
  });

  // Botón "Nueva prueba" desde resultados
  document.getElementById('btn-new').addEventListener('click', () => {
    resetAnalyze();
    show('screen-preload');
  });

  // Submit del formulario → predecir
  document.getElementById('form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const fileInput = document.getElementById('file');
    if (!fileInput.files.length) return;

    document.getElementById('form').style.display = 'none';
    startProgress();

    const fd = new FormData();
    fd.append('model_key', selectedModel);
    fd.append('file', fileInput.files[0]);

    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      const json = await res.json();
      finishProgress(() => showResults(json));
    } catch {
      finishProgress(() => {
        document.getElementById('form').style.display = 'flex';
        document.getElementById('btn-predict').disabled = false;
        alert('Error al procesar la imagen. Inténtalo de nuevo.');
      });
    }
  });
});

// ── Selección de enfermedad ───────────────────────────────────────────────
function selectDisease(modelKey) {
  selectedModel = modelKey;
  document.getElementById('analyze-title').textContent = formatLabel(modelKey);
  show('screen-analyze');
}

// ── Mostrar resultados ────────────────────────────────────────────────────
function showResults(json) {
  document.getElementById('result-tag').textContent = formatLabel(selectedModel);

  const info = document.getElementById('info');
  if (json.type === 'multiclass') {
    info.innerHTML = `
      <div class="result-row">
        <span class="result-key">Predicción</span>
        <span class="result-val">${json.label}</span>
      </div>
      <div class="result-row">
        <span class="result-key">Confianza</span>
        <span class="result-val">${json.confidence}</span>
      </div>`;
  } else {
    info.innerHTML = `
      <div class="result-row">
        <span class="result-key">Score</span>
        <span class="result-val">${json.score}</span>
      </div>`;
  }

  const g = document.getElementById('gradcam');
  if (json.gradcam_url) {
    g.src = json.gradcam_url + '?t=' + Date.now();
    g.style.display = 'block';
  } else {
    g.style.display = 'none';
  }

  show('screen-results');
}

// ── Resetear pantalla de análisis ─────────────────────────────────────────
function resetAnalyze() {
  selectedModel = null;
  clearInterval(progressInterval);
  progressInterval = null;
  document.getElementById('form').style.display = 'flex';
  document.getElementById('file').value = '';
  document.getElementById('preview-wrap').style.display = 'none';
  document.getElementById('preview-img').src = '';
  document.getElementById('upload-text').textContent = 'Haz clic o arrastra una imagen aquí';
  document.getElementById('btn-predict').disabled = true;
  document.getElementById('progress-wrap').style.display = 'none';
  const fill = document.getElementById('progress-fill');
  if (fill) fill.style.width = '0%';
}
