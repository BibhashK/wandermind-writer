/**
 * Wandermind Writer — frontend logic.
 *
 * HOW THE WEB WORKS (the short version):
 *   1. JavaScript runs in the browser.
 *   2. fetch() sends an HTTP request to your server.
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

function resetPipeline() {
  STAGES.forEach((s, i) => {
    const el = $(`stage-${s}`);
    el.className = 'stage';
    el.dataset.stage = i + 1;
  });
  for (let i = 1; i <= 5; i++) {
    const wire = $(`wire-${i}`);
    wire.className = 'wire';
    wire.dataset.wire = i;
  }
}

function setStage(name) {
  const idx = STAGES.indexOf(name);
  if (idx === -1) return;

  STAGES.forEach((s, i) => {
    const el = $(`stage-${s}`);
    el.className = i < idx ? 'stage done' : i === idx ? 'stage active' : 'stage';
    el.dataset.stage = i + 1;
  });

  for (let i = 1; i <= 5; i++) {
    const wire = $(`wire-${i}`);
    wire.className = i < idx ? 'wire done' : i === idx ? 'wire active' : 'wire';
    wire.dataset.wire = i;
  }
}

function completePipeline() {
  STAGES.forEach((s, i) => {
    const el = $(`stage-${s}`);
    el.className = 'stage done';
    el.dataset.stage = i + 1;
  });
  for (let i = 1; i <= 5; i++) {
    const wire = $(`wire-${i}`);
    wire.className = 'wire done';
    wire.dataset.wire = i;
  }
}

// ---------------------------------------------------------------------------
// VERIFY — check the keys work before spending time or quota.
// ---------------------------------------------------------------------------
async function verify() {
  const config = saveConfig();
  const status = $('verify-status');
  status.textContent = 'Checking…';

  try {
    // fetch() = send an HTTP request. This is THE core web API.
    const response = await fetch('/api/verify', {
      method: 'POST',                                   // POST = sending data
      headers: { 'Content-Type': 'application/json' },  // "the body is JSON"
      body: JSON.stringify(config),                     // JS object → JSON text
    });

    const data = await response.json();                 // JSON text → JS object

    status.textContent = response.ok
      ? `Connected as ${data.wordpress_user}`
      : data.detail;
  } catch (err) {
    status.textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// SCOUT — five ranked topic ideas.
// ---------------------------------------------------------------------------
async function scout() {
  const config = getConfig();
  if (!config) {
    show('settings-panel');
    return;
  }

  hide('scout-results');
  resetPipeline();
  show('progress');
  $('progress-message').textContent = 'Scanning the news…';

  try {
    const response = await fetch('/api/scout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),      // note: nested under "config"
    });

    const data = await response.json();
    hide('progress');

    if (!response.ok) {
      $('progress-message').textContent = data.detail;
      show('progress');
      return;
    }

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
    hide('progress');
    console.error(err);
  }
}

// ---------------------------------------------------------------------------
// GENERATE — the main event. This one STREAMS.
// ---------------------------------------------------------------------------
// Generating takes 30-60 seconds. Rather than freeze on a spinner, the server
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
    hide('progress');
    console.error(err);
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
      window.seoData = e;                    // stash it — needed at publish time
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
  status.textContent = 'Publishing…';

  try {
    const response = await fetch('/api/publish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config,
        title: $('article-title').value,
        body_markdown: $('article-body').value,     // the user's edits win
        meta_description: window.seoData?.meta_description || '',
        category: $('category-input').value.trim() || 'AI',
      }),
    });

    const data = await response.json();

    if (response.ok) {
      status.innerHTML =
        `Draft created. <a href="${data.edit_url}" target="_blank" rel="noopener">Open in WordPress</a>`;
    } else {
      status.textContent = data.detail;
    }
  } catch (err) {
    status.textContent = `Couldn't reach the server. ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// WIRE UP
// ---------------------------------------------------------------------------
$('settings-btn').onclick = () => {
  const panel = $('settings-panel');
  panel.hidden = !panel.hidden;
};

$('verify-btn').onclick = verify;
$('scout-btn').onclick = scout;
$('generate-btn').onclick = generate;
$('publish-btn').onclick = publish;

$('copy-linkedin-btn').onclick = async () => {
  await navigator.clipboard.writeText($('linkedin-body').value);
  const btn = $('copy-linkedin-btn');
  btn.textContent = 'Copied';
  setTimeout(() => (btn.textContent = 'Copy'), 2000);
};

// Enter in the topic field starts a run.
$('topic-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') generate();
});

// On load: restore saved keys, and open Settings if there are none.
loadConfigIntoForm();
if (!getConfig()) show('settings-panel');