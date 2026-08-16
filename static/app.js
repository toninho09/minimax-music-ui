// MiniMax Music3 UI — no frameworks, just fetch + polling.

const $ = (id) => document.getElementById(id);

const els = {
  badge: $("model-badge"),
  lyrics: $("lyrics"),
  instructions: $("instructions"),
  duration: $("duration"),
  durationOut: $("duration-out"),
  steps: $("steps"),
  seed: $("seed"),
  generate: $("generate"),
  resultCard: $("result-card"),
  statusLine: $("status-line"),
  errorBox: $("error-box"),
  audioBox: $("audio-box"),
  player: $("player"),
  download: $("download"),
  meta: $("meta"),
};

let elapsedTimer = null;

// ---------------------------------------------------------------- model ---

async function pollModelStatus() {
  try {
    const res = await fetch("/api/status");
    const st = await res.json();
    updateBadge(st.model, st.offload);
    if (st.model === "ready") {
      els.generate.disabled = false;
      return;
    }
    if (st.model === "error") {
      els.badge.title = st.model_error || "";
      els.badge.textContent = "model load error";
      els.badge.className = "badge error";
    }
    setTimeout(pollModelStatus, 4000);
  } catch {
    setTimeout(pollModelStatus, 4000);
  }
}

function updateBadge(status, offload) {
  const mode = offload ? { low: "low vram", auto: "offload", none: "full gpu" }[offload] || offload : "";
  if (status === "loading") {
    els.badge.textContent = `loading model…${mode ? ` (${mode})` : ""}`;
    els.badge.className = "badge loading";
  } else if (status === "ready") {
    els.badge.textContent = `model ready${mode ? ` · ${mode}` : ""}`;
    els.badge.className = "badge ready";
  }
}

// ---------------------------------------------------- structured caption ---

// Builder fields (model card: Global Metadata / Vocal Details / Arrangement).
// Key = label in the final text; value = input id.
const CAPTION_FIELDS = {
  "Genre": "cap-genre",
  "Subgenre": "cap-subgenre",
  "BPM": "cap-bpm",
  "Key": "cap-key",
  "Mood": "cap-mood",
  "Listening scenario": "cap-scenario",
  "Production": "cap-production",
  "Vocals": "cap-vox-gender",
  "Vocal timbre": "cap-vox-timbre",
  "Vocal performance": "cap-vox-style",
  "Primary instruments": "cap-inst-main",
  "Secondary instruments": "cap-inst-sec",
  "Textures": "cap-texture",
};

function buildCaption() {
  const groups = {
    Global: ["Genre", "Subgenre", "BPM", "Key", "Mood", "Listening scenario", "Production"],
    Vocals: ["Vocals", "Vocal timbre", "Vocal performance"],
    Arrangement: ["Primary instruments", "Secondary instruments", "Textures"],
  };
  const parts = [];
  for (const [group, labels] of Object.entries(groups)) {
    const fields = labels
      .map((label) => {
        const el = document.getElementById(CAPTION_FIELDS[label]);
        const val = el ? el.value.trim() : "";
        return val ? `${label}: ${val}` : null;
      })
      .filter(Boolean);
    if (fields.length) parts.push(`${group}: ${fields.join(", ")}.`);
  }
  return parts.join(" ");
}

// ------------------------------------------------------------------ form ---

els.duration.addEventListener("input", () => {
  els.durationOut.textContent = `${els.duration.value}s`;
});

document.querySelectorAll("#tag-buttons button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const ta = els.lyrics;
    const tag = `[${btn.dataset.tag}]`;
    const pos = ta.selectionStart;
    const before = ta.value.slice(0, pos);
    const after = ta.value.slice(ta.selectionEnd);
    const prefix = before && !before.endsWith("\n") ? "\n" : "";
    ta.value = `${before}${prefix}${tag}\n${after}`;
    ta.focus();
    const cursor = (before + prefix + tag + "\n").length;
    ta.setSelectionRange(cursor, cursor);
  });
});

// ------------------------------------------------------------------ jobs ---

els.generate.addEventListener("click", async () => {
  const caption = buildCaption();
  const freeText = els.instructions.value.trim();
  const instructions = [caption, freeText].filter(Boolean).join(" ");

  const payload = {
    instructions,
    lyrics: els.lyrics.value,
    duration: Number(els.duration.value),
    num_inference_steps: Number(els.steps.value) || 30,
    seed: els.seed.value === "" ? null : Number(els.seed.value),
  };
  if (!payload.instructions) {
    showError("The music description is required (free text or structured caption).");
    return;
  }

  els.generate.disabled = true;
  hideResult();
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const { job_id } = await res.json();
    showResult();
    pollJob(job_id);
  } catch (e) {
    showError(e.message);
    els.generate.disabled = false;
  }
});

async function pollJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) {
    showError(`Failed to fetch job: HTTP ${res.status}`);
    els.generate.disabled = false;
    return;
  }
  const job = await res.json();
  render(job);

  if (job.status === "queued" || job.status === "running") {
    setTimeout(() => pollJob(jobId), 2000);
  } else {
    els.generate.disabled = false;
  }
}

function render(job) {
  const parts = [];
  if (job.status === "queued") {
    parts.push(`In queue — position ${job.queue_position + 1}`);
  } else if (job.status === "running") {
    parts.push("Generating music…");
    startElapsed(job.started_at);
  } else if (job.status === "done") {
    stopElapsed();
    parts.push("Music ready!");
    showAudio(job);
  } else if (job.status === "error") {
    stopElapsed();
    showError(job.error || "Unknown error.");
  }
  els.statusLine.className = `status ${job.status}`;
  els.statusLine.innerHTML = `<span class="dot"></span>${parts.join(" · ")} <span id="elapsed" class="hint"></span>`;
}

function startElapsed(startedAt) {
  const el = () => document.getElementById("elapsed");
  stopElapsed();
  const tick = () => {
    if (el()) el().textContent = ` ${Math.floor((Date.now() / 1000 - startedAt))}s elapsed`;
  };
  tick();
  elapsedTimer = setInterval(tick, 1000);
}

function stopElapsed() {
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = null;
}

// ---------------------------------------------------------------- result ---

function showResult() {
  els.resultCard.hidden = false;
  els.errorBox.hidden = true;
  els.audioBox.hidden = true;
  els.player.removeAttribute("src");
}

function hideResult() {
  els.resultCard.hidden = true;
}

function showAudio(job) {
  const url = job.audio_url;
  els.audioBox.hidden = false;
  els.player.src = url;
  els.download.href = url;
  els.download.download = `music-${job.id}.wav`;
  els.meta.textContent = `seed ${job.params.seed} · ${job.params.duration}s target · ${job.params.num_inference_steps} steps · job ${job.id}`;
}

function showError(message) {
  els.resultCard.hidden = false;
  els.audioBox.hidden = true;
  els.errorBox.hidden = false;
  els.errorBox.textContent = message;
}

// ------------------------------------------------------------------ init ---

pollModelStatus();
