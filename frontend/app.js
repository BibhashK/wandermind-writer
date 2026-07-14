/**
 * Wandermind Writer — frontend logic.
 *
 * HOW THE WEB WORKS (the short version):
 *   1. JavaScript runs in the browser.
 *   2. fetch() sends an HTTP request to the server.
 *   3. The server runs a Python function and sends data back.
 *   4. JavaScript updates the page with that data.
 *
 * Everything below is a variation on those four steps.
 */

// ---------------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const show = (id) => ($(id).hidden = false);
const hide = (id) => ($(id).hidden = true);

// ---------------------------------------------------------------------------
// STATE
// ---------------------------------------------------------------------------
// Which model tier to use. "fast" leads with Flash-Lite; "quality" with 3.5 Flash.
let speedMode = 'fast';

// SEO metadata from the current run — needed when we publish.
let seoData = null;

// ---------------------------------------------------------------------------
// CREDENTIALS — the browser only. Never persisted server-side.
// ---------------------------------------------------------------------------
// sessionStorage lives in this browser tab and dies when the tab closes. Keys
// travel to the server only as part of a request the user initiates, are held
// in memory for that one request, and are never written to a database.

function saveConfig() {
  const config = {
    google_api_key: $('google-key').value.trim(),
    tavily_api_key: $('tavily-key').value.trim(),
    wp_url: $('wp-url').value.trim(),
    wp_username: $('wp-username').value.trim(),
    wp_app_password: $('wp-password').value.trim(),
  };
  sessionStorage.setItem('config', JSON.stringify(config));
  return config;
}

function getConfig() {
  const stored = sessionStorage.getItem('config');
  return stored ? JSON.parse(stored) : null;
}

function loadConfigIntoForm() {
  const config = getConfig();
  if (!config) return;
  $('google-key').value = config.google_api_key || '';
  $('tavily-key').value = config.tavily_api_key || '';
  $('wp-url').value = config.wp_url || '';
  $('wp-username').value = config.wp_username || '';
  $('wp-password').value = config.wp_app_password || '';
}

// ---------------------------------------------------------------------------
// THE PIPELINE — the signature visual.
// ---------------------------------------------------------------------------
// Light travels along the wire as each stage runs. Completed stages lock in
// their own colour, so a finished run leaves a spectrum across the page.

const STAGES = ['seo', 'research', 'links', 'write', 'score', 'linkedin'];

function paintPipeline(activeIndex, allDone = false) {
  STAGES.forEach((s, i) => {
    const el = $(`stage-${s}`);
    el.dataset.stage = i + 1;
    if (allDone) el.className = 'stage done';
    else if (i < activeIndex) el.className = 'stage done';
    else if (i === activeIndex) el.className = 'stage active';
    else el.className = 'stage';
  });

  for (let i = 1; i <= 5; i++) {
    const wire = $(`wire-${i}`);
    wire.dataset.wire = i;
    if (allDone) wire.className = 'wire done';
    else if (i < activeIndex) wire.className = 'wire done';
    else if (i === activeIndex) wire.className = 'wire active';
    else wire.className = 'wire';
  }
}

const resetPipeline = () => paintPipeline(-1);
const setStage = (name) => paintPipeline(STAGES.indexOf(name));
const completePipeline = () => paintPipeline(STAGES.length, true);

// ---------------------------------------------------------------------------
// VERIFY — check the keys work before spending time or quota.
// ---------------------------------------------------------------------------
async function verify() {
  const config = saveConfig();
  const status = $('verify-status');
  status.className = '';
  status.textContent = 'Checking…';

  try {
    // fetch() = send an HTTP request. This is THE core web API.
    const response = await fetch('/api/verify', {
      method: 'POST',                                   // POST = sending data
      headers: { 'Content-Type': 'application/json' },  // "the body is JSON"
      body: JSON.stringify(config),                     // JS object → JSON text
    });

    const data = await response.json();                 // JSON text → JS object

    if (response.ok) {
      status.className = 'ok';
      status.textContent = `Connected as ${data.wordpress_user}`;
    } else {
      status.className = 'bad';
      status.textContent = data.detail;
    }
  } catch (err) {
    status.className = 'bad';
    status.textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// SCOUT — five ranked ideas, tuned to whatever the user's blog covers.
// ---------------------------------------------------------------------------
async function scout() {
  const config = getConfig();
  if (!config) {
    show('settings-panel');
    return;
  }

  const niche = $('niche-input').value.trim();
  if (!niche) {
    $('niche-input').focus();
    return;
  }

  hide('scout-prompt');
  hide('scout-results');
  resetPipeline();
  show('progress');
  $('progress-message').textContent = `Scanning ${niche}…`;

  try {
    const response = await fetch('/api/scout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config, niche, mode: speedMode }),
    });

    const data = await response.json();

    if (!response.ok) {
      $('progress-message').textContent = data.detail;
      return;
    }

    hide('progress');

    const list = $('scout-list');
    list.innerHTML = '';                     // clear any previous run

    data.topics.forEach((t) => {
      const card = document.createElement('div');
      card.className = 'topic-card';
      card.innerHTML = `
        <h3>${t.topic}</h3>
        <p><strong>Why</strong>${t.why}</p>
        <p><strong>Hook</strong>${t.hook}</p>
        <button>Write this</button>
      `;
      card.querySelector('button').onclick = () => {
        $('topic-input').value = t.topic;
        $('category-input').value = t.category || 'AI';
        hide('scout-results');
        generate();
      };
      list.appendChild(card);
    });

    show('scout-results');
  } catch (err) {
    $('progress-message').textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// GENERATE — the main event. This one STREAMS.
// ---------------------------------------------------------------------------
// Generating takes 20-50 seconds. Rather than freeze on a spinner, the server
// pushes an update after each step and we render it the moment it lands.
async function generate() {
  const config = getConfig();
  if (!config) {
    show('settings-panel');
    return;
  }

  const topic = $('topic-input').value.trim();
  if (!topic) {
    $('topic-input').focus();
    return;
  }

  hide('output');
  hide('research-section');
  hide('scout-prompt');
  resetPipeline();
  show('progress');
  $('progress-message').textContent = 'Starting…';

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config,
        topic,
        category: $('category-input').value.trim() || 'AI',
        mode: speedMode,
      }),
    });

    // --- Reading a stream ---------------------------------------------------
    // Instead of waiting for the whole response, read it in chunks as the
    // server sends them. Each chunk is a "data: {...}" line.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');    // events separated by a blank line
      buffer = parts.pop();                  // keep any incomplete chunk

      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        handleEvent(JSON.parse(part.slice(6)));   // strip the "data: " prefix
      }
    }
  } catch (err) {
    $('progress-message').textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// Called once for each event the server streams back.
function handleEvent(e) {
  switch (e.event) {
    case 'progress':
      $('progress-message').textContent = e.message;
      setStage(e.step);
      break;

    case 'seo':
      seoData = e;                           // stash it — needed at publish time
      break;

    case 'research': {
      const sources = $('sources-list');
      sources.innerHTML = '';
      e.sources.forEach((s) => {
        const item = document.createElement('div');
        item.innerHTML = `<a href="${s.url}" target="_blank" rel="noopener">${s.title}</a>`;
        sources.appendChild(item);
      });
      show('research-section');
      break;
    }

    case 'article':
      $('article-title').value = e.title;
      $('article-body').value = e.body;
      show('output');
      break;

    case 'score': {
      $('seo-score').textContent = `${e.score}/100 · ${e.grade} — ${e.verdict}`;
      const checks = $('seo-checks');
      checks.innerHTML = '';
      e.checks.forEach((c) => {
        const li = document.createElement('li');
        li.className = c.passed ? 'pass' : 'fail';
        li.textContent = `${c.label} — ${c.detail}`;
        checks.appendChild(li);
      });
      break;
    }

    case 'linkedin':
      $('linkedin-body').value = e.post;
      break;

    case 'done':
      completePipeline();
      $('progress-message').textContent = 'Ready. Review it, then push to WordPress.';
      setTimeout(() => hide('progress'), 2200);   // let the spectrum land
      break;

    case 'error':
      $('progress-message').textContent = e.message;
      break;
  }
}

// ---------------------------------------------------------------------------
// PUBLISH — push the (possibly edited) article to WordPress as a draft.
// ---------------------------------------------------------------------------
async function publish() {
  const config = getConfig();
  const status = $('publish-status');
  status.className = '';
  status.textContent = 'Publishing…';

  try {
    const response = await fetch('/api/publish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config,
        title: $('article-title').value,
        body_markdown: $('article-body').value,     // the user's edits win
        meta_description: seoData?.meta_description || '',
        category: $('category-input').value.trim() || 'AI',
      }),
    });

    const data = await response.json();

    if (response.ok) {
      status.className = 'ok';
      status.innerHTML =
        `Draft created. <a href="${data.edit_url}" target="_blank" rel="noopener">Open in WordPress</a>`;
    } else {
      status.className = 'bad';
      status.textContent = data.detail;
    }
  } catch (err) {
    status.className = 'bad';
    status.textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// WIRE UP THE CONTROLS
// ---------------------------------------------------------------------------

// Settings panel.
$('settings-btn').onclick = () => {
  const panel = $('settings-panel');
  panel.hidden = !panel.hidden;
};
$('verify-btn').onclick = verify;

// The speed/quality toggle. Clicking one deactivates the other and sets the
// mode that every subsequent request will carry.
document.querySelectorAll('button.mode').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('button.mode').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    speedMode = btn.dataset.mode;
    $('mode-explainer').textContent =
      speedMode === 'fast'
        ? 'Fast: about 20 seconds. Good for most topics.'
        : 'Quality: slower, stronger model. Better with facts, numbers, and nuance.';
  };
});

// Scout: the button opens the niche prompt; the prompt runs the scout.
$('scout-btn').onclick = () => {
  const panel = $('scout-prompt');
  panel.hidden = !panel.hidden;
  if (!panel.hidden) $('niche-input').focus();
};
$('scout-go-btn').onclick = scout;
$('niche-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') scout();
});

// Generate.
$('generate-btn').onclick = generate;
$('topic-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') generate();
});

// Publish.
$('publish-btn').onclick = publish;

// Copy the LinkedIn post.
$('copy-linkedin-btn').onclick = async () => {
  await navigator.clipboard.writeText($('linkedin-body').value);
  const btn = $('copy-linkedin-btn');
  btn.textContent = 'Copied';
  setTimeout(() => (btn.textContent = 'Copy'), 2000);
};

// On load: restore saved keys, and open Settings if there are none.
loadConfigIntoForm();
if (!getConfig()) show('settings-panel');