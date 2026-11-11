// Streaming chat (milestone 9): POST the message, then show the answer as Server-Sent Events arrive.
const box = document.getElementById('messages');
const form = document.getElementById('composer');
if (box) box.scrollTop = box.scrollHeight;

function addMessage(role, text) {
  const article = document.createElement('article');
  article.className = 'message ' + role;
  const label = document.createElement('div');
  label.className = 'role';
  label.textContent = role;
  const content = document.createElement('div');
  content.className = 'content';
  content.textContent = text; // textContent, never innerHTML: the model's text is untrusted
  article.append(label, content);
  box.append(article);
  box.scrollTop = box.scrollHeight;
  return { article, content };
}

function addSources(article, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement('details');
  details.className = 'sources';
  const summary = document.createElement('summary');
  summary.textContent = sources.length + ' source(s)';
  const list = document.createElement('ul');
  for (const s of sources) {
    const item = document.createElement('li');
    const name = document.createElement('b');
    name.textContent = s.source;
    item.append(name);
    if (s.score !== null && s.score !== undefined) {
      const score = document.createElement('span');
      score.className = 'muted';
      score.textContent = ' (' + s.score.toFixed(2) + ')';
      item.append(score);
    }
    if (s.snippet) {
      const snippet = document.createElement('div');
      snippet.className = 'snippet';
      snippet.textContent = s.snippet;
      item.append(snippet);
    }
    list.append(item);
  }
  details.append(summary, list);
  article.append(details);
}

// Split the byte stream into events: blocks separated by a blank line, "event:" and "data:" lines.
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let end;
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7);
        else if (line.startsWith('data: ')) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
    if (done) return;
  }
}

async function send(text) {
  addMessage('user', text);
  const reply = addMessage('assistant', '…');
  let started = false;
  try {
    const response = await fetch(form.dataset.streamUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text }),
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    for await (const { event, data } of readEvents(response)) {
      if (event === 'token') {
        reply.content.textContent = (started ? reply.content.textContent : '') + data.text;
        started = true;
      } else if (event === 'restart') {
        reply.content.textContent = '…';
        started = false;
      } else if (event === 'done') {
        reply.content.textContent = data.message.content; // the saved text is the source of truth
        addSources(reply.article, data.message.sources);
      }
      box.scrollTop = box.scrollHeight;
    }
  } catch (err) {
    reply.content.textContent = 'Could not get an answer (' + err.message + '). Please try again.';
  }
}

if (form) {
  const button = form.querySelector('button');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const textarea = form.querySelector('textarea');
    const text = textarea.value.trim();
    if (!text || button.disabled) return;
    button.disabled = true;
    textarea.value = '';
    try {
      await send(text);
    } finally {
      button.disabled = false;
      textarea.focus();
    }
  });
}
