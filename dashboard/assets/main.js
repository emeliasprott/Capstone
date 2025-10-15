const precompOutputs = window.precomp_outputs || {};

const state = {
  terms: new Set(),
  topics: new Set(['All']),
  chamber: 'Both',
  highlightedRoute: null,
  highlightedStage: null,
  selectedBills: new Set(),
  selectedCommittee: null,
  selectedCounty: null,
  selectedLegislator: null,
  tokenMin: 0,
  selectedHeatmap: null
};

const elements = {
  pages: Array.from(document.querySelectorAll('.page')),
  navButtons: Array.from(document.querySelectorAll('nav button')),
  termSelect: document.getElementById('term-select'),
  topicSelect: document.getElementById('topic-select'),
  chamberButtons: Array.from(document.querySelectorAll('.chamber-toggle button')),
  filtersBar: document.getElementById('active-filters'),
  routeTopicChips: document.getElementById('route-topic-chips'),
  routeSelectionPill: document.getElementById('route-selection-pill'),
  routeSelectionLabel: document.querySelector('#route-selection-pill span'),
  tokenSlider: document.getElementById('token-min-slider'),
  tokenSliderValue: document.getElementById('token-min-value'),
  fundingMetric: document.getElementById('funding-metric'),
  legislatorSelect: document.getElementById('legislator-select'),
  billTableSearch: document.getElementById('bill-search'),
  billTableReset: document.getElementById('clear-bill-filters'),
  drawer: document.getElementById('drawer'),
  drawerTitle: document.getElementById('drawer-title'),
  drawerContent: document.getElementById('drawer-content'),
  drawerClose: document.getElementById('drawer-close'),
  fundingCountyDetail: document.getElementById('funding-county-detail')
};

let dataTable = null;
let leafletMap = null;
let countyLayer = null;

function buildOptions(data, accessor) {
  const values = new Set();
  (data || []).forEach(row => {
    const value = accessor(row);
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        value.forEach(v => values.add(v));
      } else {
        values.add(value);
      }
    }
  });
  return Array.from(values).filter(Boolean).sort();
}

function initializeSelect(selectEl, values, includeAll = false) {
  selectEl.innerHTML = '';
  if (includeAll) {
    const option = document.createElement('option');
    option.value = 'All';
    option.textContent = 'All';
    option.selected = true;
    selectEl.appendChild(option);
  }
  values.forEach(value => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    if (includeAll) option.selected = false;
    selectEl.appendChild(option);
  });
}

function updateSelectSelections() {
  Array.from(elements.termSelect.options).forEach(option => {
    option.selected = state.terms.has(option.value);
  });
  const activeTopics = state.topics.size ? Array.from(state.topics) : ['All'];
  Array.from(elements.topicSelect.options).forEach(option => {
    option.selected = activeTopics.includes(option.value);
  });
}

function updateChamberButtons() {
  elements.chamberButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.chamber === state.chamber);
  });
}

function updateFiltersBar() {
  const container = elements.filtersBar;
  container.innerHTML = '';
  const filters = [];
  if (state.terms.size) filters.push({ label: 'Terms', value: Array.from(state.terms).join(', ') });
  if (state.topics.size && !(state.topics.size === 1 && state.topics.has('All'))) {
    filters.push({ label: 'Topics', value: Array.from(state.topics).join(', ') });
  }
  if (state.chamber !== 'Both') filters.push({ label: 'Chamber', value: state.chamber });
  if (state.selectedCommittee) filters.push({ label: 'Committee', value: state.selectedCommittee });
  if (state.selectedCounty) filters.push({ label: 'County', value: state.selectedCounty });
  if (state.highlightedStage) filters.push({ label: 'Funnel stage', value: state.highlightedStage });
  if (state.highlightedRoute) filters.push({ label: 'Route', value: state.highlightedRoute });

  filters.forEach(filter => {
    const pill = document.createElement('span');
    pill.className = 'filter-pill';
    pill.textContent = `${filter.label}: ${filter.value}`;
    container.appendChild(pill);
  });

  if (filters.length) {
    const clear = document.createElement('button');
    clear.className = 'clear';
    clear.textContent = 'Clear filters';
    clear.addEventListener('click', () => {
      state.terms.clear();
      state.topics = new Set(['All']);
      state.chamber = 'Both';
      state.selectedCommittee = null;
      state.selectedCounty = null;
      state.highlightedStage = null;
      state.highlightedRoute = null;
      updateSelectSelections();
      updateChamberButtons();
      updateRouteTopicChips();
      renderAll();
    });
    container.appendChild(clear);
  }
}

function updateRouteTopicChips() {
  const topics = buildOptions(precompOutputs.route_archetypes, d => d.topic);
  const container = elements.routeTopicChips;
  container.innerHTML = '';
  const allChip = document.createElement('div');
  allChip.className = `chip ${state.topics.has('All') ? 'active' : ''}`;
  allChip.textContent = 'All';
  allChip.addEventListener('click', () => {
    state.topics = new Set(['All']);
    renderAll();
  });
  container.appendChild(allChip);
  topics.forEach(topic => {
    const chip = document.createElement('div');
    chip.className = `chip ${state.topics.has('All') ? '' : state.topics.has(topic) ? 'active' : ''}`;
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      if (state.topics.has('All')) state.topics.delete('All');
      if (chip.classList.contains('active')) {
        state.topics.delete(topic);
        if (!state.topics.size) state.topics.add('All');
      } else {
        state.topics.add(topic);
      }
      renderAll();
    });
    container.appendChild(chip);
  });
  elements.routeSelectionPill.hidden = !state.highlightedRoute;
  if (state.highlightedRoute) elements.routeSelectionLabel.textContent = state.highlightedRoute;
}

function handleNavClick(targetId) {
  elements.pages.forEach(page => {
    page.classList.toggle('active', page.id === targetId);
  });
  elements.navButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.target === targetId);
  });
}

elements.navButtons.forEach(btn => {
  btn.addEventListener('click', () => handleNavClick(btn.dataset.target));
});

elements.termSelect.addEventListener('change', () => {
  state.terms = new Set(Array.from(elements.termSelect.selectedOptions).map(opt => opt.value));
  renderAll();
});

elements.topicSelect.addEventListener('change', () => {
  const selection = Array.from(elements.topicSelect.selectedOptions).map(opt => opt.value);
  state.topics = new Set(selection.length ? selection : ['All']);
  if (state.topics.has('All') && state.topics.size > 1) state.topics = new Set(['All']);
  renderAll();
});

elements.chamberButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    state.chamber = btn.dataset.chamber;
    updateChamberButtons();
    renderAll();
  });
});

elements.tokenSlider.addEventListener('input', () => {
  state.tokenMin = Number(elements.tokenSlider.value);
  elements.tokenSliderValue.textContent = `${state.tokenMin}+`;
  renderTokenLift();
});

elements.fundingMetric.addEventListener('change', () => {
  renderFundingMap();
});

elements.legislatorSelect.addEventListener('change', () => {
  state.selectedLegislator = elements.legislatorSelect.value;
  renderLegislatorTopicMix();
});

elements.billTableReset.addEventListener('click', () => {
  state.selectedBills.clear();
  state.highlightedRoute = null;
  state.highlightedStage = null;
  state.selectedCommittee = null;
  state.selectedCounty = null;
  renderAll();
  if (dataTable) dataTable.search('').draw();
});

elements.billTableSearch.addEventListener('input', () => {
  if (dataTable) dataTable.search(elements.billTableSearch.value).draw();
});

document.querySelectorAll('.term button').forEach(btn => {
  btn.addEventListener('click', () => {
    const parent = btn.closest('.term');
    parent.classList.toggle('open');
  });
});

elements.drawerClose.addEventListener('click', () => closeDrawer());

function openDrawer(title, contentHtml) {
  elements.drawerTitle.textContent = title;
  elements.drawerContent.innerHTML = contentHtml;
  elements.drawer.classList.add('open');
}

function closeDrawer() {
  elements.drawer.classList.remove('open');
}

function filterByState(rows) {
  const termFilter = state.terms.size ? state.terms : null;
  const topicsFilter = state.topics.has('All') ? null : state.topics;
  const chamberFilter = state.chamber;
  return (rows || []).filter(row => {
    if (!row) return false;
    if (termFilter) {
      const term = row.term || row.Term || row.session;
      if (term && !termFilter.has(term)) return false;
    }
    if (topicsFilter) {
      const topic = row.topic || row.Topic || row.policy_area;
      if (topic && !topicsFilter.has(topic)) return false;
    }
    if (chamberFilter && chamberFilter !== 'Both') {
      const chamber = row.chamber || row.Chamber || row.body;
      if (chamber && chamber !== chamberFilter) return false;
    }
    if (state.selectedCommittee) {
      const committees = row.committee || row.committees;
      if (committees && typeof committees === 'string' && !committees.includes(state.selectedCommittee)) return false;
    }
    return true;
  });
}

function filterBills() {
  return filterByState(precompOutputs.bills_table);
}
function renderPipelineSankey() {
  const data = filterByState(precompOutputs.pipeline_stage_funnel);
  if (!data.length) {
    Plotly.react('pipeline-sankey', [{ type: 'scatter', x: [0.5], y: [0.5], text: ['No pipeline data for the current filters'], mode: 'text' }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const nodes = new Map();
  let index = 0;
  data.forEach(row => {
    if (!nodes.has(row.from)) nodes.set(row.from, index++);
    if (!nodes.has(row.to)) nodes.set(row.to, index++);
  });
  const source = [];
  const target = [];
  const value = [];
  const colors = [];
  const labels = [];
  const edgeTexts = [];
  const nodeColors = [];
  const passRateScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);
  data.forEach(row => {
    const passRate = Number(row.pass_rate) || 0;
    source.push(nodes.get(row.from));
    target.push(nodes.get(row.to));
    value.push(Number(row.entered) || 0);
    colors.push(passRateScale(passRate));
    edgeTexts.push(`Median: ${(row.median_days ?? 'n/a')} days`);
  });
  nodes.forEach((idx, name) => {
    labels[idx] = name;
    nodeColors[idx] = '#d0d5dd';
  });
  const plotData = [{
    type: 'sankey',
    orientation: 'h',
    node: {
      label: labels,
      color: nodeColors,
      pad: 24,
      thickness: 20
    },
    link: {
      source,
      target,
      value,
      color: colors,
      hovertemplate: data.map(row => `From: ${row.from}<br>To: ${row.to}<br>Entered: ${row.entered}<br>Advanced: ${row.advanced}<br>Pass rate: ${(Number(row.pass_rate) * 100).toFixed(1)}%<br>Median days: ${row.median_days}<extra></extra>`),
      label: edgeTexts
    }
  }];
  const layout = { margin: { l: 20, r: 20, t: 20, b: 20 }, hovermode: 'closest', font: { family: 'Inter, sans-serif' } };
  Plotly.react('pipeline-sankey', plotData, layout);
  const sankeyElement = document.getElementById('pipeline-sankey');
  sankeyElement.on('plotly_click', dataPoint => {
    const pointIndex = dataPoint.points[0].pointNumber;
    const row = data[pointIndex];
    if (!row) return;
    state.highlightedStage = row.to;
    renderBillsTable();
    updateFiltersBar();
  });
}

function parseRoute(routeKey) {
  if (!routeKey) return [];
  if (Array.isArray(routeKey)) return routeKey;
  if (typeof routeKey === 'string') return routeKey.split('>').map(step => step.trim()).filter(Boolean);
  return [];
}

function renderRouteMetro() {
  const svg = d3.select('#route-metro');
  svg.selectAll('*').remove();
  const width = svg.node().clientWidth || 800;
  const height = svg.node().clientHeight || 420;
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const filtered = filterByState(precompOutputs.route_archetypes);
  if (!filtered.length) {
    svg.append('text').attr('x', width / 2).attr('y', height / 2).attr('text-anchor', 'middle').text('No route data for the current filters');
    return;
  }
  const topicsFilter = state.topics.has('All') ? null : state.topics;
  const display = topicsFilter ? filtered.filter(row => topicsFilter.has(row.topic)) : filtered;
  const colorScale = d3.scaleSequential().domain([0, 1]).interpolator(d3.interpolatePuOr);
  const thicknessScale = d3.scaleSqrt().domain(d3.extent(display, d => Number(d.n) || 1)).range([4, 16]);
  const routes = display.slice(0, 20);
  const maxStops = d3.max(routes, r => parseRoute(r.route_key).length) || 4;
  const verticalSpacing = height / (routes.length + 1);
  const horizontalSpacing = width / (maxStops + 1);
  const routeGroup = svg.append('g').attr('fill', 'none').attr('stroke-linecap', 'round');
  routes.forEach((route, i) => {
    const stops = parseRoute(route.route_key).slice(0, 5);
    const yOffset = verticalSpacing * (i + 1);
    const line = d3.line().x((d, idx) => horizontalSpacing * (idx + 1)).y((d, idx) => yOffset + Math.sin(idx * 1.2) * 12);
    routeGroup.append('path')
      .datum(stops)
      .attr('d', line)
      .attr('stroke', colorScale(Number(route.pass_rate) || 0))
      .attr('stroke-width', thicknessScale(Number(route.n) || 1))
      .attr('opacity', state.highlightedRoute && state.highlightedRoute !== route.route_key ? 0.25 : 0.9)
      .style('cursor', 'pointer')
      .on('click', () => {
        state.highlightedRoute = route.route_key;
        elements.routeSelectionPill.hidden = false;
        elements.routeSelectionLabel.textContent = route.route_key;
        renderBillsTable();
        updateFiltersBar();
        renderRouteMetro();
        showRouteBillsDrawer(route);
      })
      .on('mouseenter', function () {
        d3.select(this).attr('opacity', 1);
      })
      .on('mouseleave', function () {
        d3.select(this).attr('opacity', state.highlightedRoute && state.highlightedRoute !== route.route_key ? 0.25 : 0.9);
      });
    stops.forEach((stop, idx) => {
      routeGroup.append('circle')
        .attr('cx', horizontalSpacing * (idx + 1))
        .attr('cy', yOffset + Math.sin(idx * 1.2) * 12)
        .attr('r', 6)
        .attr('fill', '#fff')
        .attr('stroke', colorScale(Number(route.pass_rate) || 0))
        .attr('stroke-width', 2);
      routeGroup.append('text')
        .attr('x', horizontalSpacing * (idx + 1))
        .attr('y', yOffset + Math.sin(idx * 1.2) * 12 - 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#475467')
        .attr('font-size', 12)
        .text(stop);
    });
    routeGroup.append('text')
      .attr('x', width - 160)
      .attr('y', yOffset)
      .attr('fill', '#101828')
      .attr('font-weight', 600)
      .text(`${route.topic} • ${(Number(route.pass_rate) * 100).toFixed(1)}% pass`);
  });
}

function showRouteBillsDrawer(route) {
  const bills = filterBills().filter(bill => bill.route_key === route.route_key && (!bill.topic || bill.topic === route.topic || state.topics.has('All')));
  const items = bills.slice(0, 25).map(bill => `
    <div class="mini-list-item">
      <strong>${bill.bill_ID || bill.bill_id || bill.Bill}</strong>
      <div>${bill.title || bill.short_title || 'No summary available'}</div>
      <div class="pill">Outcome: ${bill.outcome || bill.status || 'Unknown'}</div>
    </div>
  `).join('');
  openDrawer('Bills on this route', `<div class="mini-list">${items || '<p>No bills available for this route.</p>'}</div>`);
}
function renderAmendmentScatter() {
  const data = filterByState(precompOutputs.amendment_churn);
  if (!data.length) {
    Plotly.react('amendment-scatter', [{ type: 'scatter', mode: 'text', x: [0.5], y: [0.5], text: ['No amendment data for the current filters'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const colorScale = d3.scaleOrdinal(d3.schemeTableau10);
  const trace = {
    type: 'scattergl',
    mode: 'markers',
    x: data.map(d => Number(d.n_versions) || 0),
    y: data.map(d => Number(d.median_sim) || 0),
    marker: {
      color: data.map(d => colorScale(d.topic || 'Other')),
      size: 10,
      opacity: 0.8
    },
    text: data.map(d => d.bill_ID || d.bill_id),
    hovertemplate: data.map(d => `Bill: ${d.bill_ID || d.bill_id}<br>Versions: ${d.n_versions}<br>Median similarity: ${(Number(d.median_sim) * 100).toFixed(1)}%<br>Topic: ${d.topic || 'Unknown'}<extra></extra>`)
  };
  const layout = {
    xaxis: { title: 'Number of versions', zeroline: false },
    yaxis: { title: 'Median similarity', range: [0, 1], tickformat: '.0%' },
    margin: { l: 60, r: 20, t: 20, b: 60 },
    hovermode: 'closest',
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('amendment-scatter', [trace], layout);
  const scatterEl = document.getElementById('amendment-scatter');
  scatterEl.on('plotly_click', point => {
    const index = point.points[0].pointIndex;
    const bill = data[index];
    if (bill) showBillDetailDrawer(bill.bill_ID || bill.bill_id);
  });
}

function renderTokenLift() {
  const data = filterByState(precompOutputs.text_lift_top_tokens);
  const filtered = (data || []).filter(row => (Number(row.pos) || 0) + (Number(row.neg) || 0) >= state.tokenMin);
  if (!filtered.length) {
    Plotly.react('token-bars', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['Adjust filters to view token lift insights.'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const sorted = filtered.sort((a, b) => (Number(a.log_lift_pass_vs_other) || 0) - (Number(b.log_lift_pass_vs_other) || 0));
  const x = sorted.map(d => Number(d.log_lift_pass_vs_other) || 0);
  const y = sorted.map(d => d.token);
  const counts = sorted.map(d => (Number(d.pos) || 0) + (Number(d.neg) || 0));
  const opacityScale = d3.scaleLinear().domain(d3.extent(counts)).range([0.4, 1]);
  const colors = x.map(value => value >= 0 ? '#1f77b4' : '#d62728');
  const trace = {
    type: 'bar',
    x,
    y,
    orientation: 'h',
    marker: {
      color: colors,
      opacity: counts.map(c => opacityScale(c))
    },
    hovertemplate: sorted.map(d => `Token: ${d.token}<br>Log lift: ${d.log_lift_pass_vs_other}<br>Pass count: ${d.pos}<br>Other count: ${d.neg}<extra></extra>`)
  };
  const layout = {
    margin: { l: 160, r: 20, t: 20, b: 20 },
    xaxis: { title: 'Log lift (pass vs. other)', zeroline: true, zerolinecolor: '#98a2b3' },
    yaxis: { automargin: true },
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('token-bars', [trace], layout);
  document.getElementById('token-bars').on('plotly_click', point => {
    const token = sorted[point.points[0].pointIndex]?.token;
    if (!token) return;
    const bills = filterBills().filter(bill => (bill.text_tokens || '').includes(token) || (bill.summary || '').includes(token)).slice(0, 20);
    const list = bills.map(bill => `
      <div class="mini-list-item">
        <strong>${bill.bill_ID || bill.bill_id}</strong>
        <div>${bill.title || bill.short_title || 'No summary available'}</div>
        <div>Outcome: ${bill.outcome || bill.status || 'Unknown'}</div>
      </div>
    `).join('');
    openDrawer(`Bills featuring "${token}"`, `<div class="mini-list">${list || '<p>No bills match this token yet.</p>'}</div>`);
  });
}

function renderGatekeeping() {
  const data = filterByState(precompOutputs.committee_gatekeeping);
  if (!data.length) {
    Plotly.react('gatekeeping-lollipop', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No committee data for the current filters'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const sorted = [...data].sort((a, b) => (Number(b.gatekeeping) || 0) - (Number(a.gatekeeping) || 0));
  const committees = sorted.map(d => d.committee);
  const scores = sorted.map(d => Number(d.gatekeeping) || 0);
  const colorScale = d3.scaleLinear().domain([0, 0.5, 1]).range(['#16a34a', '#facc15', '#dc2626']);
  const lines = {
    type: 'scatter',
    mode: 'lines',
    x: scores,
    y: committees,
    line: { color: '#d0d5dd', width: 2 },
    hoverinfo: 'skip'
  };
  const dots = {
    type: 'scatter',
    mode: 'markers',
    x: scores,
    y: committees,
    marker: { color: scores.map(s => colorScale(s)), size: 16 },
    hovertemplate: sorted.map(d => `${d.committee}<br>Entries: ${d.entries}<br>Exits: ${d.exits}<br>Gatekeeping: ${(Number(d.gatekeeping) * 100).toFixed(1)}%<extra></extra>`)
  };
  const layout = {
    margin: { l: 180, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Gatekeeping score', tickformat: '.0%' },
    yaxis: { automargin: true },
    hovermode: 'closest',
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('gatekeeping-lollipop', [lines, dots], layout);
  document.getElementById('gatekeeping-lollipop').on('plotly_click', point => {
    const committee = sorted[point.points[0].pointIndex]?.committee;
    if (!committee) return;
    state.selectedCommittee = committee;
    renderAll();
  });
}

function renderDriftBeeswarm() {
  const data = filterByState(precompOutputs.committee_floor_drift);
  if (!data.length) {
    Plotly.react('drift-beeswarm', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No drift data for the current filters'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const colorScale = d3.scaleOrdinal().domain(['Assembly', 'Senate']).range(['#1f77b4', '#ff7f0e']);
  const traces = d3.groups(data, d => d.chamber || 'Both').map(([chamber, rows]) => ({
    type: 'scatter',
    mode: 'markers',
    name: chamber,
    x: rows.map(r => Number(r.drift) || 0),
    y: rows.map(() => chamber),
    marker: { size: 12, color: colorScale(chamber), opacity: 0.7 },
    text: rows.map(r => r.legislator_name),
    hovertemplate: rows.map(r => `${r.legislator_name}<br>Committee yes: ${r.comm_yes}<br>Floor yes: ${r.floor_yes}<br>Drift: ${(Number(r.drift) * 100).toFixed(1)} pts<extra></extra>`)
  }));
  const layout = {
    margin: { l: 60, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Drift (floor - committee yes rate)', zeroline: true, zerolinecolor: '#94a3b8', ticksuffix: '%' },
    yaxis: { title: '', automargin: true },
    hovermode: 'closest',
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('drift-beeswarm', traces, layout);
  document.getElementById('drift-beeswarm').on('plotly_selected', event => {
    if (!event || !event.points?.length) return;
    const selected = new Set(event.points.map(pt => pt.text));
    const legislators = data.filter(row => selected.has(row.legislator_name));
    const list = legislators.map(row => `<div class="mini-list-item"><strong>${row.legislator_name}</strong><div>Drift: ${(Number(row.drift) * 100).toFixed(1)} pts</div></div>`).join('');
    openDrawer('Selected legislators', `<div class="mini-list">${list}</div>`);
  });
}
function renderVotingNetwork() {
  const edges = filterByState(precompOutputs.vote_similarity_edges).filter(edge => Number(edge.sim) >= 0.6);
  const svg = d3.select('#voting-network');
  svg.selectAll('*').remove();
  const width = svg.node().clientWidth || 800;
  const height = svg.node().clientHeight || 420;
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  if (!edges.length) {
    svg.append('text').attr('x', width / 2).attr('y', height / 2).attr('text-anchor', 'middle').text('No voting network data for the current filters');
    return;
  }
  const nodesMap = new Map();
  edges.forEach(edge => {
    if (!nodesMap.has(edge.u)) nodesMap.set(edge.u, { id: edge.u, connections: 0 });
    if (!nodesMap.has(edge.v)) nodesMap.set(edge.v, { id: edge.v, connections: 0 });
    nodesMap.get(edge.u).connections += 1;
    nodesMap.get(edge.v).connections += 1;
  });
  const nodes = Array.from(nodesMap.values());
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(140).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2));
  const opacityScale = d3.scaleLinear().domain(d3.extent(edges, e => Number(e.sim) || 0.6)).range([0.2, 0.9]);
  const link = svg.append('g').attr('stroke', '#cbd5f5').attr('stroke-width', 1.5)
    .selectAll('line').data(edges).enter().append('line').attr('stroke-opacity', d => opacityScale(Number(d.sim) || 0.6));
  const node = svg.append('g').attr('stroke', '#fff').attr('stroke-width', 1.5)
    .selectAll('circle').data(nodes).enter().append('circle')
    .attr('r', d => 8 + Math.sqrt(d.connections))
    .attr('fill', '#4f46e5')
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', event => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      })
      .on('drag', event => {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      })
      .on('end', event => {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }))
    .on('click', (event, d) => {
      const bills = filterBills().filter(bill => (bill.legislators || []).includes(d.id)).slice(0, 10);
      const list = bills.map(bill => `<div class="mini-list-item"><strong>${bill.bill_ID || bill.bill_id}</strong><div>${bill.title || 'No summary'}</div></div>`).join('');
      openDrawer(`Profile: ${d.id}`, `<p>Connections: ${d.connections}</p><div class="mini-list">${list || '<p>No bill activity found.</p>'}</div>`);
    });
  const labels = svg.append('g').selectAll('text').data(nodes).enter().append('text')
    .attr('font-size', 11)
    .attr('fill', '#1f2937')
    .attr('text-anchor', 'middle')
    .text(d => d.id);
  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    labels.attr('x', d => d.x).attr('y', d => d.y - 14);
  });
}

function renderControversyHeatmap() {
  const data = filterByState(precompOutputs.topic_controversy);
  if (!data.length) {
    Plotly.react('controversy-heatmap', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No controversy data available'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    Plotly.purge('controversy-dumbbell');
    return;
  }
  const topics = Array.from(new Set(data.map(d => d.topic))).sort();
  const terms = Array.from(new Set(data.map(d => d.term))).sort();
  const z = topics.map(topic => terms.map(term => {
    const row = data.find(d => d.topic === topic && d.term === term);
    return row ? Number(row.party_line_share) : null;
  }));
  const trace = {
    type: 'heatmap',
    x: terms,
    y: topics,
    z,
    colorscale: 'Viridis',
    hovertemplate: topics.map(topic => terms.map(term => {
      const row = data.find(d => d.topic === topic && d.term === term);
      if (!row) return `${topic}<br>${term}<extra>None</extra>`;
      return `${topic}<br>${term}<br>Party-line share: ${(Number(row.party_line_share) * 100).toFixed(1)}%<br>Mean polarization: ${(Number(row.mean_polarization) * 100).toFixed(1)}%<br>Roll calls: ${row.n_rollcalls}<extra></extra>`;
    }))
  };
  const layout = {
    margin: { l: 140, r: 20, t: 20, b: 80 },
    xaxis: { title: 'Term' },
    yaxis: { title: 'Topic' },
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('controversy-heatmap', [trace], layout);
  document.getElementById('controversy-heatmap').on('plotly_click', point => {
    const topic = point.points[0].y;
    const term = point.points[0].x;
    state.selectedHeatmap = { topic, term };
    renderControversyDumbbell();
    showRollcallDrawer(topic, term);
  });
  renderControversyDumbbell();
}

function renderControversyDumbbell() {
  const selection = state.selectedHeatmap;
  const data = filterByState(precompOutputs.topic_controversy);
  if (!data.length) {
    Plotly.purge('controversy-dumbbell');
    return;
  }
  const term = selection?.term || Array.from(new Set(data.map(d => d.term))).sort().slice(-1)[0];
  const rows = data.filter(d => d.term === term);
  if (!rows.length) {
    Plotly.react('controversy-dumbbell', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['Select a term from the heatmap'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const topics = rows.map(d => d.topic);
  const dem = rows.map(d => Number(d.dem_yes_rate) || 0);
  const rep = rows.map(d => Number(d.rep_yes_rate) || 0);
  const lines = {
    type: 'scatter',
    mode: 'lines',
    x: dem.concat(rep),
    y: topics.concat(topics),
    line: { color: '#98a2b3' },
    hoverinfo: 'skip'
  };
  const demTrace = {
    type: 'scatter',
    mode: 'markers',
    name: 'Democratic yes rate',
    x: dem,
    y: topics,
    marker: { color: '#1f77b4', size: 10 }
  };
  const repTrace = {
    type: 'scatter',
    mode: 'markers',
    name: 'Republican yes rate',
    x: rep,
    y: topics,
    marker: { color: '#ff7f0e', size: 10 }
  };
  const annotations = rows.map((row, idx) => ({
    x: Math.max(dem[idx], rep[idx]),
    y: row.topic,
    xanchor: 'left',
    text: `${Math.abs(dem[idx] - rep[idx]).toFixed(2)} gap`,
    showarrow: false,
    font: { color: '#475467', size: 11 }
  }));
  const layout = {
    margin: { l: 160, r: 20, t: 30, b: 40 },
    xaxis: { title: 'Yes-rate', tickformat: '.0%' },
    yaxis: { automargin: true },
    legend: { orientation: 'h', y: -0.2 },
    annotations,
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('controversy-dumbbell', [lines, demTrace, repTrace], layout);
}

function showRollcallDrawer(topic, term) {
  const rows = (precompOutputs.rollcall_party_splits || []).filter(row => (!topic || row.topic === topic) && (!term || row.term === term)).sort((a, b) => (Number(b.polarization) || 0) - (Number(a.polarization) || 0)).slice(0, 5);
  const items = rows.map(row => `
    <div class="mini-list-item">
      <strong>${row.rollcall_id || row.bill_id}</strong>
      <div>${row.summary || 'No summary available'}</div>
      <div class="pill">Polarization: ${(Number(row.polarization) * 100).toFixed(1)}%</div>
    </div>
  `).join('');
  openDrawer(`Most split roll calls – ${topic} (${term})`, `<div class="mini-list">${items || '<p>No roll calls available for this selection.</p>'}</div>`);
}
function renderFundingMap() {
  if (!leafletMap) {
    leafletMap = L.map('funding-map', { scrollWheelZoom: false, zoomControl: true }).setView([37.5, -119.5], 5.7);
    L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(leafletMap);
  }
  const geo = precompOutputs.ca_legislator_funding_geo;
  const data = filterByState(precompOutputs.ca_legislator_funding);
  if (!geo || !geo.features) {
    return;
  }
  if (countyLayer) {
    countyLayer.remove();
  }
  const metric = elements.fundingMetric.value || 'total_donations';
  const valueMap = new Map();
  data.forEach(row => {
    const key = row.county || row.County || row.geography;
    if (!key) return;
    valueMap.set(key, Number(row[metric]) || 0);
  });
  const values = Array.from(valueMap.values());
  const scale = values.length ? d3.scaleSequential(d3.interpolateBlues).domain(d3.extent(values)) : () => '#e5e7eb';
  countyLayer = L.geoJSON(geo, {
    style: feature => {
      const name = feature.properties?.name || feature.properties?.COUNTY;
      const value = valueMap.get(name) || 0;
      return {
        color: '#ffffff',
        weight: 1,
        fillColor: scale(value),
        fillOpacity: 0.8
      };
    },
    onEachFeature: (feature, layer) => {
      const name = feature.properties?.name || feature.properties?.COUNTY;
      const value = valueMap.get(name) || 0;
      layer.bindTooltip(`${name}: ${value.toLocaleString()}`);
      layer.on('click', () => {
        state.selectedCounty = name;
        updateFiltersBar();
        renderCountyDetail(name, metric);
      });
    }
  }).addTo(leafletMap);
  if (state.selectedCounty) renderCountyDetail(state.selectedCounty, metric);
}

function renderCountyDetail(county, metric) {
  const container = elements.fundingCountyDetail;
  const data = (precompOutputs.ca_legislator_funding || []).filter(row => (row.county || row.County) === county);
  const metricLabel = metric.replace(/_/g, ' ');
  const items = data.sort((a, b) => (Number(b[metric]) || 0) - (Number(a[metric]) || 0)).slice(0, 8).map(row => `
    <div class="mini-list-item">
      <strong>${row.legislator || row.legislator_name || row.district}</strong>
      <div>${metricLabel}: ${(Number(row[metric]) || 0).toLocaleString()}</div>
      <a class="link-button" href="#" data-bill-filter="${row.legislator}">View related bills</a>
    </div>
  `).join('');
  container.innerHTML = `
    <h3>${county || 'Select a county'}</h3>
    <div class="mini-list">${items || '<p>No funding records for this county.</p>'}</div>
  `;
  container.querySelectorAll('a[data-bill-filter]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      const legislator = event.currentTarget.dataset.billFilter;
      if (!legislator) return;
      if (dataTable) {
        dataTable.column(0).search(legislator, true, false).draw();
      }
    });
  });
}

function renderTopicFundingTerm() {
  const data = filterByState(precompOutputs.topic_funding_by_term);
  if (!data.length) {
    Plotly.react('topic-funding-term', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No funding data available'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const topics = Array.from(new Set(data.map(d => d.topic))).sort();
  const terms = Array.from(new Set(data.map(d => d.term))).sort();
  const traces = ['total_donations', 'total_lobbying'].map(key => ({
    name: key.replace('total_', '').replace('_', ' '),
    type: 'bar',
    x: terms,
    y: terms.map(term => d3.sum(data.filter(row => row.term === term), row => Number(row[key]) || 0)),
    marker: { opacity: key === 'total_donations' ? 0.9 : 0.6 }
  }));
  const layout = {
    barmode: 'stack',
    margin: { l: 60, r: 20, t: 20, b: 60 },
    xaxis: { title: 'Term' },
    yaxis: { title: 'Amount (USD)' },
    legend: { orientation: 'h', y: -0.2 },
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('topic-funding-term', traces, layout);
  document.getElementById('topic-funding-term').on('plotly_click', point => {
    const term = point.points[0].x;
    const topic = topics[point.points[0].pointNumber % topics.length];
    showTopicTermDetail(topic, term);
  });
}

function renderLegislatorTopicMix() {
  const data = filterByState(precompOutputs.topic_funding_by_leg);
  if (!data.length) {
    Plotly.react('legislator-topic-mix', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['Select a legislator to view mix'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const legislators = Array.from(new Set(data.map(d => d.canon || d.legislator || d.legislator_name))).sort();
  if (!elements.legislatorSelect.options.length) {
    initializeSelect(elements.legislatorSelect, legislators);
    elements.legislatorSelect.value = legislators[0];
    state.selectedLegislator = legislators[0];
  }
  const selected = state.selectedLegislator || elements.legislatorSelect.value;
  const rows = data.filter(row => (row.canon || row.legislator || row.legislator_name) === selected);
  if (!rows.length) {
    Plotly.react('legislator-topic-mix', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No data for this legislator'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const total = d3.sum(rows, row => Number(row.total) || Number(row.donations) + Number(row.lobbying) || 0);
  const trace = {
    type: 'bar',
    x: rows.map(row => row.topic),
    y: rows.map(row => (Number(row.total) || Number(row.donations) + Number(row.lobbying) || 0) / (total || 1)),
    marker: { color: '#4f46e5' }
  };
  const layout = {
    margin: { l: 60, r: 20, t: 40, b: 80 },
    xaxis: { title: 'Topic', automargin: true },
    yaxis: { title: 'Share of total', tickformat: '.0%' },
    font: { family: 'Inter, sans-serif' }
  };
  Plotly.react('legislator-topic-mix', [trace], layout);
  document.getElementById('legislator-topic-mix').on('plotly_click', point => {
    const topic = rows[point.points[0].pointIndex]?.topic;
    if (!topic) return;
    showTopicTermDetail(topic, rows[0]?.term);
  });
}

function showTopicTermDetail(topic, term) {
  const bills = filterBills().filter(bill => (!topic || bill.topic === topic) && (!term || bill.term === term)).slice(0, 15);
  const list = bills.map(bill => `
    <div class="mini-list-item">
      <strong>${bill.bill_ID || bill.bill_id}</strong>
      <div>${bill.title || bill.short_title || 'No summary available'}</div>
      <div class="pill">Outcome: ${bill.outcome || bill.status || 'Unknown'}</div>
    </div>
  `).join('');
  openDrawer(`Bills in ${topic || 'topic'} (${term || 'all terms'})`, `<div class="mini-list">${list || '<p>No bills available for this combination.</p>'}</div>`);
}

function outcomeChip(outcome) {
  const normalized = (outcome || '').toLowerCase();
  if (normalized.includes('chapter') || normalized.includes('pass')) {
    return '<span class="status-chip success">✔︎ Chaptered</span>';
  }
  if (normalized.includes('veto') || normalized.includes('fail')) {
    return '<span class="status-chip danger">✖︎ Failed</span>';
  }
  return '<span class="status-chip neutral">• Pending</span>';
}

function renderBillsTable() {
  const data = filterBills().filter(row => {
    if (state.highlightedStage) {
      return (row.stages || '').includes(state.highlightedStage);
    }
    if (state.highlightedRoute) {
      return row.route_key === state.highlightedRoute;
    }
    if (state.selectedCommittee) {
      return (row.committees || '').includes(state.selectedCommittee);
    }
    if (state.selectedCounty) {
      return (row.county || '') === state.selectedCounty;
    }
    return true;
  });
  if (dataTable) {
    dataTable.clear();
    dataTable.rows.add(data.map(formatBillRow));
    dataTable.draw();
    return;
  }
  dataTable = new DataTable('#bill-table', {
    data: data.map(formatBillRow),
    columns: [
      { title: 'Bill' },
      { title: 'Topic' },
      { title: 'Term' },
      { title: 'First action' },
      { title: 'Longevity (days)' },
      { title: 'Versions' },
      { title: 'Similarity' },
      { title: 'Outcome' }
    ]
  });
  document.querySelector('#bill-table tbody').addEventListener('click', event => {
    const rowEl = event.target.closest('tr');
    if (!rowEl) return;
    const rowData = dataTable.row(rowEl).data();
    if (!rowData) return;
    const billId = rowData[0];
    showBillDetailDrawer(billId);
  });
}

function formatBillRow(row) {
  const billId = row.bill_ID || row.bill_id || row.Bill || row.bill;
  return [
    billId,
    row.topic || row.Topic || 'Unknown',
    row.term || row.Term || '—',
    row.First_action || row.first_action || '—',
    row.longevity_days || row.longevity || '—',
    row.n_versions || row.bill_version_count || '—',
    row.median_sim ? `${(Number(row.median_sim) * 100).toFixed(1)}%` : '—',
    outcomeChip(row.outcome || row.status)
  ];
}

function showBillDetailDrawer(billId) {
  const bills = precompOutputs.bills_table || [];
  const bill = bills.find(row => (row.bill_ID || row.bill_id || row.Bill) === billId);
  if (!bill) return;
  const route = bill.route_key || 'Unknown route';
  const versions = bill.n_versions || bill.bill_version_count || '—';
  const similarity = bill.median_sim ? `${(Number(bill.median_sim) * 100).toFixed(1)}%` : '—';
  const votePreview = (precompOutputs.rollcall_party_splits || []).filter(row => row.bill_id === billId).slice(0, 5);
  const votesHtml = votePreview.map(row => `<div>${row.rollcall_id}: ${(Number(row.polarization) * 100).toFixed(1)}% polarization</div>`).join('');
  const content = `
    <div>
      <h4>${billId}</h4>
      <p>${bill.title || bill.short_title || 'No summary available'}</p>
      <div class="mini-list">
        <div class="mini-list-item"><strong>Route</strong><div>${route}</div></div>
        <div class="mini-list-item"><strong>Amendments</strong><div>${versions} versions • similarity ${similarity}</div></div>
        <div class="mini-list-item"><strong>Outcome</strong><div>${bill.outcome || bill.status || 'Unknown'}</div></div>
      </div>
      <h4>Recent votes</h4>
      <div class="mini-list">${votesHtml || '<p>No roll calls recorded.</p>'}</div>
    </div>
  `;
  openDrawer(`Bill ${billId}`, content);
}

function initializeFilters() {
  initializeSelect(elements.termSelect, buildOptions(precompOutputs.bills_table, row => row.term || row.Term));
  initializeSelect(elements.topicSelect, buildOptions(precompOutputs.bills_table, row => row.topic || row.Topic), true);
  updateChamberButtons();
  updateRouteTopicChips();
}

function renderAll() {
  updateSelectSelections();
  updateFiltersBar();
  renderPipelineSankey();
  renderRouteMetro();
  renderAmendmentScatter();
  renderTokenLift();
  renderGatekeeping();
  renderDriftBeeswarm();
  renderVotingNetwork();
  renderControversyHeatmap();
  renderFundingMap();
  renderTopicFundingTerm();
  renderLegislatorTopicMix();
  renderBillsTable();
}

document.addEventListener('DOMContentLoaded', () => {
  initializeFilters();
  renderAll();
});
