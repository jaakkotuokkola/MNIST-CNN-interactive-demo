
// Handles drawing on the canvas and sending image to server for prediction

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');
const resultEl = document.getElementById('result');
const probsEl = document.getElementById('probs');

let drawing = false;

function resizeCanvasForDpi() {
  const dpi = window.devicePixelRatio || 1;
  const w = canvas.width;
  const h = canvas.height;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width = w * dpi;
  canvas.height = h * dpi;
  ctx.scale(dpi, dpi);
}

resizeCanvasForDpi();

ctx.fillStyle = '#ffffff';
ctx.fillRect(0,0,canvas.width,canvas.height);
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.strokeStyle = '#000000';

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches.length) {
    return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
  } else {
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }
}

canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
});
canvas.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});
canvas.addEventListener('pointerup', () => { drawing = false; });
canvas.addEventListener('pointerleave', () => { drawing = false; });

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  resultEl.textContent = '—';
  probsEl.textContent = '';
});

predictBtn.addEventListener('click', async () => {
  // get a PNG data URL at displayed size
  const dataUrl = canvas.toDataURL('image/png');
  resultEl.textContent = '…';
  probsEl.textContent = '';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Prediction failed');
    resultEl.textContent = data.guess;
    probsEl.innerHTML = data.probs.map((p,i) => `<div>${i}: ${ (p*100).toFixed(2)}%</div>`).join('');
  } catch (err) {
    resultEl.textContent = 'Error';
    probsEl.textContent = err.message;
  }
});
