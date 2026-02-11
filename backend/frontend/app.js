async function loadModels() {
  const sel = document.getElementById('modelSelect');
  sel.innerHTML = '<option>Cargando...</option>';
  try {
    const r = await fetch('/models');
    const j = await r.json();
    sel.innerHTML = '';
    if (j.models && j.models.length) {
      j.models.forEach(m => {
        const o = document.createElement('option'); o.value = m; o.textContent = m; sel.appendChild(o);
      });
    } else {
      const o = document.createElement('option'); o.textContent = 'No hay modelos'; sel.appendChild(o);
    }
  } catch (e) {
    sel.innerHTML = '<option>Error</option>';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  const form = document.getElementById('form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const sel = document.getElementById('modelSelect');
    const fileInput = document.getElementById('file');
    if (!fileInput.files.length) return alert('Selecciona una imagen');
    const fd = new FormData();
    fd.append('model_key', sel.value);
    fd.append('file', fileInput.files[0]);
    const btn = form.querySelector('button'); btn.disabled = true; btn.textContent = 'Procesando...';
    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      const json = await res.json();
      document.getElementById('result').style.display = 'block';
      const info = document.getElementById('info');
      if (json.type === 'multiclass') {
        info.innerHTML = `<strong>Predicción:</strong> ${json.label} <br/><strong>Confianza:</strong> ${json.confidence}`;
      } else {
        info.innerHTML = `<strong>Score:</strong> ${json.score}`;
      }
      const g = document.getElementById('gradcam');
      if (json.gradcam_url) {
        g.src = json.gradcam_url + '?t=' + Date.now(); g.style.display = 'block';
      } else {
        g.style.display = 'none';
      }
    } catch (e) {
      alert('Error al predecir');
    } finally {
      btn.disabled = false; btn.textContent = 'Predecir';
    }
  });
});
