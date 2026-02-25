/* ── PronosticoSport — app.js ── */

const DATA_URL = 'data/predictions.json';

const SPORT_META = {
  nba:        { label: 'NBA',             icon: '🏀', color: '#c9a227' },
  nhl:        { label: 'NHL',             icon: '🏒', color: '#0057b7' },
  mlb:        { label: 'MLB',             icon: '⚾', color: '#e63946' },
  epl:        { label: 'Premier League',  icon: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', color: '#7c3aed' },
  laliga:     { label: 'La Liga',         icon: '🇪🇸', color: '#e63946' },
  bundesliga: { label: 'Bundesliga',      icon: '🇩🇪', color: '#e63946' },
  ligue1:     { label: 'Ligue 1',         icon: '🇫🇷', color: '#2563eb' },
  primeira:   { label: 'Primeira Liga',   icon: '🇵🇹', color: '#16a34a' },
  ucl:        { label: 'Champions',       icon: '🏆', color: '#6d28d9' },
};

const SOCCER_LEAGUES = ['epl', 'laliga', 'bundesliga', 'ligue1', 'primeira'];
const PARLAY_TARGETS = [3, 4, 5, 6, 10, 20];

let data = null;
let activeSportTab = 'nba';
let activeSoccerLeague = 'epl';
let activeParlayTab = 'por_deporte';

/* ── Boot ── */
document.addEventListener('DOMContentLoaded', async () => {
  renderLoading();
  try {
    const res = await fetch(DATA_URL + '?t=' + Date.now());
    if (!res.ok) throw new Error('HTTP ' + res.status);
    data = await res.json();
  } catch (e) {
    document.getElementById('content').innerHTML =
      `<div class="empty-state"><div class="icon">⚠️</div>
       <p>No se pudieron cargar las predicciones.</p>
       <p style="font-size:0.78rem;margin-top:0.5rem;opacity:0.6">${e.message}</p>
       </div>`;
    return;
  }
  buildNav();
  buildParlayNav();
  renderTab(activeSportTab);
  updateHeaderMeta();
});

/* ── Header meta ── */
function updateHeaderMeta() {
  const el = document.getElementById('generated-at');
  if (!el || !data) return;
  const d = new Date(data.generated_at);
  el.textContent = 'Actualizado: ' + d.toLocaleString('es-ES', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }) + ' UTC';
}

/* ── Build top navigation ── */
function buildNav() {
  const nav = document.querySelector('.tab-nav');
  nav.innerHTML = '';

  const tabs = [
    { key: 'nba',    label: '🏀 NBA'   },
    { key: 'nhl',    label: '🏒 NHL'   },
    { key: 'mlb',    label: '⚾ MLB'   },
    { key: 'soccer', label: '⚽ Fútbol' },
    { key: 'ucl',    label: '🏆 UCL'   },
    { key: 'parlays',label: '🎰 Parlays' },
  ];

  tabs.forEach(({ key, label }) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (key === activeSportTab ? ' active' : '');
    btn.dataset.tab = key;
    btn.textContent = label;
    btn.onclick = () => renderTab(key);
    nav.appendChild(btn);
  });
}

function setActiveTabBtn(key) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === key);
  });
}

/* ── Build parlay sub-nav ── */
function buildParlayNav() {}

/* ── Route tabs ── */
function renderTab(key) {
  activeSportTab = key;
  setActiveTabBtn(key);
  const content = document.getElementById('content');

  if (key === 'soccer') {
    renderSoccerSection(content);
  } else if (key === 'parlays') {
    renderParlaysSection(content);
  } else if (key === 'ucl') {
    renderSportSection(content, 'ucl');
  } else {
    renderSportSection(content, key);
  }
}

function renderLoading() {
  document.getElementById('content').innerHTML =
    `<div class="loading"><div class="spinner"></div>Cargando predicciones…</div>`;
}

/* ── Single sport section ── */
function renderSportSection(container, league) {
  if (!data) return;
  const sport = data.sports[league];
  if (!sport) {
    container.innerHTML = emptyState('No hay datos para esta liga.');
    return;
  }

  const meta = SPORT_META[league] || { label: league.toUpperCase(), icon: '🏅', color: '#94a3b8' };
  let html = '';

  // Header
  html += `<div class="sport-header">
    <div class="sport-title" style="color:${meta.color}">${meta.icon} ${sport.name || meta.label}</div>`;

  if (sport.metrics) {
    html += `<div class="metrics-row">
      <span class="metric">Precisión <span>${(sport.metrics.accuracy * 100).toFixed(1)}%</span></span>
      <span class="metric">Log Loss <span>${sport.metrics.logloss.toFixed(3)}</span></span>
      <span class="metric">Brier <span>${sport.metrics.brier.toFixed(3)}</span></span>
    </div>`;
  }
  if (sport.analysis) {
    html += `<div class="claude-analysis"><span class="claude-badge">🤖 Análisis</span> ${escHtml(sport.analysis)}</div>`;
  }
  html += '</div>';

  // Status checks
  if (sport.status === 'skipped') {
    html += `<div class="empty-state"><div class="icon">📅</div><p>${sport.reason || 'Liga no disponible hoy'}</p></div>`;
    container.innerHTML = html;
    return;
  }
  if (sport.status === 'error') {
    html += `<div class="status-error">⚠️ Error al procesar ${meta.label}: ${sport.reason || 'desconocido'}</div>`;
    container.innerHTML = html;
    return;
  }
  if (!sport.picks || sport.picks.length === 0) {
    html += emptyState('No hay partidos programados para hoy.');
    container.innerHTML = html;
    return;
  }

  // Table
  const hasOdds  = sport.picks.some(p => p.house_implied_pct != null);
  const hasOU    = sport.picks.some(p => p.ou_line != null);
  const hasNotes = sport.picks.some(p => p.note);
  html += `<div class="table-wrap"><table>
    <thead><tr>
      <th class="td-num">#</th>
      <th>Partido</th>
      <th>Pick</th>
      <th>Modelo %</th>
      ${hasOdds ? '<th class="td-casa">Casa %</th><th class="td-edge">Edge</th>' : '<th class="td-odds">Cuota</th>'}
      ${hasOU ? '<th class="td-ou">Over/Under</th>' : ''}
      <th class="td-signal">Señal</th>
      ${hasNotes ? '<th class="td-note">Análisis</th>' : ''}
    </tr></thead>
    <tbody>`;

  sport.picks.forEach((p, i) => {
    const confPct = Math.round(p.p_win * 100);
    const barClass = p.signal === 'alta' ? '' : p.signal === 'media' ? 'medium' : 'low';
    const barWidth = confPct + '%';

    const injHome = (p.injuries_home || []).slice(0, 3)
      .map(x => `<span class="inj-badge inj-${x.status.toLowerCase().split(' ')[0]}">${escHtml(x.player)} <em>${escHtml(x.status)}</em></span>`)
      .join('');
    const injAway = (p.injuries_away || []).slice(0, 3)
      .map(x => `<span class="inj-badge inj-${x.status.toLowerCase().split(' ')[0]}">${escHtml(x.player)} <em>${escHtml(x.status)}</em></span>`)
      .join('');

    let oddsCell = '';
    if (hasOdds) {
      const housePct = p.house_implied_pct != null ? Math.round(p.house_implied_pct * 100) : null;
      const edge = p.edge_pct != null ? p.edge_pct : (housePct != null ? confPct - housePct : null);
      const edgeColor = edge == null ? 'var(--text-muted)' : edge >= 5 ? 'var(--green)' : edge >= 0 ? 'var(--yellow)' : 'var(--red)';
      const edgeIcon = edge == null ? '–' : edge >= 5 ? '✅' : edge >= 0 ? '⚠️' : '❌';
      const edgeStr = edge != null ? `${edge >= 0 ? '+' : ''}${Number(edge).toFixed(1)}%` : '–';
      oddsCell = `
      <td class="td-casa">${housePct != null ? housePct + '%' : '–'}</td>
      <td class="td-edge" style="color:${edgeColor}">${edgeIcon} ${edgeStr}</td>`;
    } else {
      oddsCell = `<td class="td-odds">${p.implied_odds.toFixed(2)}</td>`;
    }

    html += `<tr>
      <td class="td-num">${i + 1}</td>
      <td class="td-match">
        <div class="team-row"><span>${escHtml(p.home_team)}</span>${injHome ? `<span class="inj-row">${injHome}</span>` : ''}</div>
        <div class="team-row away-row"><span>vs ${escHtml(p.away_team)}</span>${injAway ? `<span class="inj-row">${injAway}</span>` : ''}</div>
        ${p.date ? `<div class="game-date">${p.date}</div>` : ''}
      </td>
      <td class="td-pick">${escHtml(p.pick_label)}</td>
      <td class="td-conf">
        <div class="conf-bar">
          <div class="conf-fill ${barClass}" style="width:${barWidth}; max-width:60px; display:inline-block"></div>
          <span style="color:${signalColor(p.signal)}">${confPct}%</span>
        </div>
      </td>
      ${oddsCell}
      ${hasOU ? `<td class="td-ou">${ouCell(p)}</td>` : ''}
      <td class="td-signal">${signalBadge(p.signal)}</td>
      ${hasNotes ? `<td class="td-note">${p.note ? escHtml(p.note) : '<span style="color:var(--text-muted)">–</span>'}</td>` : ''}
    </tr>`;
  });

  html += '</tbody></table></div>';
  container.innerHTML = html;
}

/* ── Soccer multi-league section ── */
function renderSoccerSection(container) {
  if (!data) return;

  // Build sub-nav
  let html = `<div class="soccer-nav" id="soccer-nav">`;
  SOCCER_LEAGUES.forEach(lg => {
    const m = SPORT_META[lg];
    const picks = data.sports[lg]?.picks || [];
    const cnt = picks.length;
    const active = lg === activeSoccerLeague ? ' active' : '';
    html += `<button class="soccer-btn${active}" onclick="switchSoccer('${lg}')">${m.icon} ${m.label} ${cnt > 0 ? `<span style="opacity:0.6">(${cnt})</span>` : ''}</button>`;
  });
  html += '</div>';
  html += `<div id="soccer-content"></div>`;

  container.innerHTML = html;
  renderSoccerLeague(activeSoccerLeague);
}

function switchSoccer(lg) {
  activeSoccerLeague = lg;
  document.querySelectorAll('.soccer-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('onclick').includes(`'${lg}'`));
  });
  renderSoccerLeague(lg);
}

function renderSoccerLeague(lg) {
  const container = document.getElementById('soccer-content');
  if (!container) return;
  renderSportSection(container, lg);
}

/* ── Parlays section ── */
function renderParlaysSection(container) {
  if (!data) return;

  let html = `<div class="parlay-nav">
    <button class="parlay-tab-btn ${activeParlayTab === 'por_deporte' ? 'active' : ''}" onclick="switchParlayTab('por_deporte')">🏅 Por Deporte</button>
    <button class="parlay-tab-btn ${activeParlayTab === 'combinados' ? 'active' : ''}" onclick="switchParlayTab('combinados')">🌐 Combinados</button>
  </div>
  <div id="parlay-body"></div>`;

  container.innerHTML = html;
  renderParlayBody(activeParlayTab);
}

function switchParlayTab(tab) {
  activeParlayTab = tab;
  document.querySelectorAll('.parlay-tab-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('onclick').includes(`'${tab}'`));
  });
  renderParlayBody(tab);
}

function renderParlayBody(tab) {
  const body = document.getElementById('parlay-body');
  if (!body || !data?.parlays) return;

  if (tab === 'por_deporte') {
    renderParlaysBySport(body);
  } else {
    renderParlaysCombined(body);
  }
}

function renderParlaysBySport(container) {
  const byS = data.parlays?.by_sport || {};
  const sportGroups = [
    { key: 'nba',    icon: '🏀', label: 'NBA'             },
    { key: 'nhl',    icon: '🏒', label: 'NHL'             },
    { key: 'soccer', icon: '⚽', label: 'Fútbol (todas)'  },
  ];

  let html = '';
  sportGroups.forEach(({ key, icon, label }) => {
    const parlays = byS[key];
    if (!parlays) return;

    html += `<div class="accordion open" id="acc-${key}">
      <div class="accordion-header" onclick="toggleAccordion('acc-${key}')">
        <div class="accordion-title">${icon} ${label}</div>
        <div class="accordion-arrow">▶</div>
      </div>
      <div class="accordion-body">
        <div class="parlay-grid">`;

    PARLAY_TARGETS.forEach(t => {
      const pk = 'cuota_' + t;
      const p = parlays[pk];
      html += parlayCard(p, t, false);
    });

    html += `</div></div></div>`;
  });

  container.innerHTML = html || emptyState('No hay datos de parlays disponibles.');
}

function renderParlaysCombined(container) {
  const combined = data.parlays?.combined || {};
  let html = '<div class="parlay-grid">';

  PARLAY_TARGETS.forEach(t => {
    const pk = 'cuota_' + t;
    const p = combined[pk];
    html += parlayCard(p, t, true);
  });

  html += '</div>';
  container.innerHTML = html;
}

/* ── Parlay card HTML ── */
function parlayCard(p, target, showSport) {
  const cls = 'cuota-' + target;
  if (!p || !p.legs || p.legs.length === 0) {
    return `<div class="parlay-card">
      <div class="parlay-card-head">
        <div class="parlay-target ${cls}">~${target}x</div>
        <div class="parlay-meta">Sin datos</div>
      </div>
      <div class="parlay-body" style="color:var(--text-muted);font-size:0.8rem;padding:1rem">
        No hay suficientes partidos hoy para esta cuota.
      </div>
    </div>`;
  }

  const actualOdds = p.total_odds.toFixed(2);
  const prob = (p.combined_prob * 100).toFixed(1);
  const n = p.n_legs || p.legs.length;

  let legsHtml = '';
  p.legs.forEach(leg => {
    const sportLabel = SPORT_META[leg.sport]?.label || leg.sport.toUpperCase();
    legsHtml += `<div class="parlay-leg">
      <span class="leg-bullet">▸</span>
      <div class="leg-info">
        <div class="leg-match">${escHtml(leg.match)}</div>
        <div class="leg-pick">${escHtml(leg.pick)}</div>
      </div>
      ${showSport ? `<span class="leg-sport-tag">${sportLabel}</span>` : ''}
      <span class="leg-odds-pill">${leg.implied_odds.toFixed(2)}</span>
    </div>`;
  });

  return `<div class="parlay-card">
    <div class="parlay-card-head">
      <div class="parlay-target ${cls}">~${target}x</div>
      <div class="parlay-meta">${n} piernas<br><strong style="color:var(--text)">${actualOdds}</strong> real</div>
    </div>
    <div class="parlay-body">${legsHtml}</div>
    <div class="parlay-foot">
      <div class="parlay-total ${cls}">${actualOdds}x</div>
      <div class="parlay-prob">Prob combinada: <span>${prob}%</span></div>
    </div>
  </div>`;
}

/* ── Accordion toggle ── */
function toggleAccordion(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('open');
}

/* ── Helpers ── */
function signalColor(s) {
  if (s === 'alta')  return 'var(--green)';
  if (s === 'media') return 'var(--yellow)';
  return 'var(--red)';
}

function signalBadge(s) {
  if (s === 'alta')  return `<span class="badge badge-alta">● Alta</span>`;
  if (s === 'media') return `<span class="badge badge-media">● Media</span>`;
  return `<span class="badge badge-baja">● Baja</span>`;
}

function ouCell(p) {
  if (p.ou_line == null) return '–';
  const icon  = p.ou_pick === 'over'  ? '⬆️' : p.ou_pick === 'under' ? '⬇️' : '';
  const label = p.ou_pick === 'over'  ? 'Over'
              : p.ou_pick === 'under' ? 'Under' : 'O/U';
  const color = p.ou_pick === 'over'  ? 'var(--green)'
              : p.ou_pick === 'under' ? 'var(--red)' : 'var(--text-muted)';
  return `<span style="color:${color};font-weight:700;white-space:nowrap">${icon} ${label} ${p.ou_line}</span>`;
}

function emptyState(msg) {
  return `<div class="empty-state"><div class="icon">📅</div><p>${escHtml(msg)}</p></div>`;
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
