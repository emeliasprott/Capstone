const precomp = window.precomp_outputs || {};

const state = {
  terms: new Set(),
  topics: new Set(['All']),
  chamber: 'Both',
  selectedRoute: null,
  selectedStage: null,
  selectedCounty: null,
  tokenMin: 5,
  selectedLegislator: null
};

const elements = {
  pages: Array.from(document.querySelectorAll('.page')),
  navButtons: Array.from(document.querySelectorAll('nav button')),
  termSelect: document.getElementById('term-select'),
  topicSelect: document.getElementById('topic-select'),
  chamberButtons: Array.from(document.querySelectorAll('.chamber-toggle button')),
  filtersBar: document.getElementById('active-filters'),
  routeChips: document.getElementById('route-topic-chips'),
  routeSelection: document.getElementById('route-selection-pill'),
  routeSelectionLabel: document.querySelector('#route-selection-pill span'),
  tokenSlider: document.getElementById('token-min-slider'),
  tokenSliderValue: document.getElementById('token-min-value'),
  fundingMetric: document.getElementById('funding-metric'),
  legislatorSelect: document.getElementById('legislator-select'),
  billSearch: document.getElementById('bill-search'),
  billReset: document.getElementById('clear-bill-filters'),
  drawer: document.getElementById('drawer'),
  drawerTitle: document.getElementById('drawer-title'),
  drawerContent: document.getElementById('drawer-content'),
  drawerClose: document.getElementById('drawer-close'),
  riskRegister: document.getElementById('risk-register'),
  fundingCountyDetail: document.getElementById('funding-county-detail')
};

let billsTableInstance = null;
let mapInstance = null;
let mapLayer = null;

function uniqueValues(rows, accessor) {
  const values = new Set();
  (rows || []).forEach(row => {
    const value = accessor(row);
    if (Array.isArray(value)) {
      value.forEach(v => v && values.add(v));
    } else if (value !== undefined && value !== null && value !== '') {
      values.add(value);
    }
  });
  return Array.from(values).sort();
}

function renderRouteTopicChips() {
  if (!elements.routeChips) return;
  const topics = uniqueValues(precomp.route_archetypes || [], row => row.topic);
  elements.routeChips.innerHTML = '';
  const allChip = document.createElement('div');
  allChip.className = `chip ${state.topics.has('All') ? 'active' : ''}`;
  allChip.textContent = 'All';
  allChip.addEventListener('click', () => {
    state.topics = new Set(['All']);
    updateSelectSelections();
    updateFilterPills();
    renderAll();
  });
  elements.routeChips.appendChild(allChip);
  topics.forEach(topic => {
    const chip = document.createElement('div');
    chip.className = `chip ${state.topics.has('All') ? '' : state.topics.has(topic) ? 'active' : ''}`;
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      if (state.topics.has('All')) state.topics = new Set();
      if (state.topics.has(topic)) {
        state.topics.delete(topic);
        if (!state.topics.size) state.topics.add('All');
      } else {
        state.topics.add(topic);
      }
      updateSelectSelections();
      updateFilterPills();
      renderAll();
    });
    elements.routeChips.appendChild(chip);
  });
}

function initializeSelect(select, values, includeAll = false) {
  select.innerHTML = '';
  if (includeAll) {
    const option = document.createElement('option');
    option.value = 'All';
    option.textContent = 'All';
    option.selected = true;
    select.appendChild(option);
  }
  values.forEach(value => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function filterData(rows = []) {
  const termFilter = state.terms.size ? state.terms : null;
  const topicFilter = state.topics.has('All') ? null : state.topics;
  const chamberFilter = state.chamber && state.chamber !== 'Both' ? state.chamber : null;
  return rows.filter(row => {
    if (!row) return false;
    const term = row.term || row.Term;
    const topic = row.topic || row.Topic;
    const chamber = row.chamber || row.Chamber;
    if (termFilter && term && !termFilter.has(String(term))) return false;
    if (topicFilter && topic && !topicFilter.has(topic)) return false;
    if (chamberFilter && chamber && chamber !== chamberFilter) return false;
    if (state.selectedRoute && row.route_key && row.route_key !== state.selectedRoute) return false;
    if (state.selectedStage && row.stage && row.stage !== state.selectedStage && row.from !== state.selectedStage) return false;
    if (state.selectedCounty && row.county && row.county !== state.selectedCounty) return false;
    return true;
  });
}

function updateFilterPills() {
  const pills = [];
  if (state.terms.size) pills.push({ label: 'Terms', value: Array.from(state.terms).join(', ') });
  if (!state.topics.has('All')) pills.push({ label: 'Topics', value: Array.from(state.topics).join(', ') });
  if (state.chamber !== 'Both') pills.push({ label: 'Chamber', value: state.chamber });
  if (state.selectedRoute) pills.push({ label: 'Route', value: state.selectedRoute });
  if (state.selectedStage) pills.push({ label: 'Stage', value: state.selectedStage });
  if (state.selectedCounty) pills.push({ label: 'County', value: state.selectedCounty });

  elements.filtersBar.innerHTML = '';
  pills.forEach(({ label, value }) => {
    const span = document.createElement('span');
    span.className = 'filter-pill';
    span.textContent = `${label}: ${value}`;
    elements.filtersBar.appendChild(span);
  });
  if (pills.length) {
    const clear = document.createElement('button');
    clear.className = 'clear';
    clear.textContent = 'Clear filters';
    clear.addEventListener('click', () => {
      state.terms.clear();
      state.topics = new Set(['All']);
      state.chamber = 'Both';
      state.selectedRoute = null;
      state.selectedStage = null;
      state.selectedCounty = null;
      updateSelectSelections();
      updateChamberButtons();
      updateFilterPills();
      renderAll();
    });
    elements.filtersBar.appendChild(clear);
  }
}

function updateSelectSelections() {
  Array.from(elements.termSelect.options).forEach(opt => {
    opt.selected = state.terms.has(opt.value);
  });
  Array.from(elements.topicSelect.options).forEach(opt => {
    opt.selected = state.topics.has('All') ? opt.value === 'All' : state.topics.has(opt.value);
  });
}

function updateChamberButtons() {
  elements.chamberButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.chamber === state.chamber);
  });
}

function openDrawer(title, html) {
  elements.drawerTitle.textContent = title;
  elements.drawerContent.innerHTML = html;
  elements.drawer.classList.add('open');
}

function closeDrawer() {
  elements.drawer.classList.remove('open');
}

elements.drawerClose.addEventListener('click', closeDrawer);

elements.navButtons.forEach(button => {
  button.addEventListener('click', () => {
    const target = button.dataset.target;
    elements.pages.forEach(page => page.classList.toggle('active', page.id === target));
    elements.navButtons.forEach(btn => btn.classList.toggle('active', btn === button));
  });
});

elements.termSelect.addEventListener('change', () => {
  state.terms = new Set(Array.from(elements.termSelect.selectedOptions).map(opt => opt.value));
  updateFilterPills();
  renderAll();
});

elements.topicSelect.addEventListener('change', () => {
  const selected = Array.from(elements.topicSelect.selectedOptions).map(opt => opt.value);
  if (!selected.length || selected.includes('All')) {
    state.topics = new Set(['All']);
  } else {
    state.topics = new Set(selected);
  }
  updateFilterPills();
  renderAll();
});

elements.chamberButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    state.chamber = btn.dataset.chamber;
    updateChamberButtons();
    updateFilterPills();
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

elements.billReset.addEventListener('click', () => {
  if (billsTableInstance) billsTableInstance.search('').draw();
  state.selectedRoute = null;
  state.selectedStage = null;
  state.selectedCounty = null;
  updateFilterPills();
  renderAll();
});

elements.billSearch.addEventListener('input', () => {
  if (billsTableInstance) billsTableInstance.search(elements.billSearch.value).draw();
});

function renderPipelineSankey() {
  const data = filterData(precomp.pipeline_stage_funnel || []);
  if (!data.length) {
    Plotly.react('pipeline-sankey', [{ type: 'scatter', mode: 'text', x: [0.5], y: [0.5], text: ['No pipeline data for selection'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const aggregated = d3.rollups(
    data,
    rows => ({
      entered: d3.sum(rows, r => Number(r.entered) || 0),
      advanced: d3.sum(rows, r => Number(r.advanced) || 0),
      median: d3.mean(rows, r => Number(r.median_days) || 0)
    }),
    r => r.from,
    r => r.to
  );
  const nodes = new Map();
  aggregated.forEach(([from, targets]) => {
    if (!nodes.has(from)) nodes.set(from, nodes.size);
    targets.forEach(([to]) => { if (!nodes.has(to)) nodes.set(to, nodes.size); });
  });
  const source = [];
  const target = [];
  const value = [];
  const text = [];
  aggregated.forEach(([from, targets]) => {
    const fromIndex = nodes.get(from);
    targets.forEach(([to, stats]) => {
      source.push(fromIndex);
      target.push(nodes.get(to));
      value.push(stats.entered);
      const pct = stats.entered ? ((stats.advanced / stats.entered) * 100).toFixed(1) : '0.0';
      text.push(`Entered: ${stats.entered}<br>Advanced: ${stats.advanced}<br>Pass rate: ${pct}%<br>Median days: ${stats.median?.toFixed(1) ?? '—'}`);
    });
  });
  const labels = Array.from(nodes.entries()).sort((a, b) => a[1] - b[1]).map(([label]) => label.replace(/_/g, ' '));
  Plotly.react('pipeline-sankey', [{
    type: 'sankey',
    orientation: 'h',
    node: { label: labels, pad: 18, thickness: 18, color: '#d0d5dd' },
    link: {
      source,
      target,
      value,
      hovertemplate: text.map(t => `${t}<extra></extra>`)
    }
  }], { margin: { l: 12, r: 12, t: 12, b: 12 } });
  const sankeyEl = document.getElementById('pipeline-sankey');
  sankeyEl.on('plotly_click', evt => {
    const idx = evt.points[0].pointNumber;
    state.selectedStage = labels[target[idx]];
    updateFilterPills();
    renderAll();
  });
}

function renderStageDurations() {
  const data = filterData(precomp.pipeline_stage_durations || []);
  if (!data.length) {
    Plotly.react('stage-duration', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No stage timing data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const aggregated = d3.rollups(data, rows => ({
    median: d3.mean(rows, r => Number(r.median_days) || 0),
    p90: d3.mean(rows, r => Number(r.p90_days) || 0)
  }), r => r.stage.replace(/_/g, ' '));
  const stages = aggregated.map(([stage]) => stage);
  const medians = aggregated.map(([_, stats]) => stats.median);
  const p90s = aggregated.map(([_, stats]) => stats.p90);
  const traceMedian = { type: 'bar', name: 'Median', x: stages, y: medians, marker: { color: '#4f46e5' } };
  const traceP90 = { type: 'bar', name: 'P90', x: stages, y: p90s, marker: { color: '#94a3b8' } };
  Plotly.react('stage-duration', [traceMedian, traceP90], {
    barmode: 'group',
    margin: { l: 40, r: 20, t: 10, b: 60 },
    yaxis: { title: 'Days' },
    xaxis: { tickangle: -20 }
  });
}

function renderRiskRegister() {
  const data = filterData(precomp.risk_register || []);
  const sorted = data.filter(row => Number(row.risk_score) > 0).sort((a, b) => (Number(b.risk_score) || 0) - (Number(a.risk_score) || 0)).slice(0, 12);
  if (!sorted.length) {
    elements.riskRegister.innerHTML = '<p>No bills currently flagged.</p>';
    return;
  }
  elements.riskRegister.innerHTML = sorted.map(row => `
    <div class="mini-list-item">
      <strong>${row.bill_id}</strong>
      <div>${row.topic || 'Unknown topic'}</div>
      <div class="pill">Score: ${row.risk_score}</div>
      <div class="caption">${row.reasons || 'No notes'}${row.days_since_last ? ` • ${row.days_since_last} days since activity` : ''}</div>
    </div>
  `).join('');
}

function renderRouteArchetypes() {
  const data = filterData(precomp.route_archetypes || []);
  const filtered = state.topics.has('All') ? data : data.filter(row => state.topics.has(row.topic));
  if (elements.routeSelection) {
    elements.routeSelection.hidden = !state.selectedRoute;
    if (state.selectedRoute) elements.routeSelectionLabel.textContent = state.selectedRoute;
  }
  if (!filtered.length) {
    d3.select('#route-metro').selectAll('*').remove();
    d3.select('#route-metro').append('text').attr('x', 20).attr('y', 20).text('No route data');
    return;
  }
  const topRoutes = filtered.slice(0, 20);
  const svg = d3.select('#route-metro');
  svg.selectAll('*').remove();
  const width = svg.node().clientWidth || 800;
  const height = svg.node().clientHeight || 420;
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const yScale = d3.scaleBand().domain(topRoutes.map(d => d.route_key)).range([20, height - 20]).padding(0.3);
  const xScale = d3.scaleLinear().domain([0, d3.max(topRoutes, d => Number(d.n) || 1)]).range([160, width - 40]);
  const colorScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);
  svg.append('g').attr('transform', 'translate(150,0)').call(d3.axisLeft(yScale));
  const bars = svg.append('g');
  bars.selectAll('rect').data(topRoutes).enter().append('rect')
    .attr('x', 160)
    .attr('y', d => yScale(d.route_key))
    .attr('height', yScale.bandwidth())
    .attr('width', d => xScale(Number(d.n) || 0) - 160)
    .attr('fill', d => colorScale(Number(d.pass_rate) || 0))
    .style('cursor', 'pointer')
    .on('click', (_evt, d) => {
      state.selectedRoute = d.route_key;
      elements.routeSelection.hidden = false;
      elements.routeSelectionLabel.textContent = d.route_key;
      updateFilterPills();
      renderAll();
      showRouteDrawer(d.route_key, d.topic);
    });
  svg.append('g').selectAll('text').data(topRoutes).enter().append('text')
    .attr('x', d => xScale(Number(d.n) || 0) + 6)
    .attr('y', d => yScale(d.route_key) + yScale.bandwidth() / 2 + 4)
    .text(d => `${d.n} bills • ${(Number(d.pass_rate) * 100 || 0).toFixed(1)}% pass`)
    .attr('fill', '#475467');
}

function showRouteDrawer(routeKey, topic) {
  const rows = filterData(precomp.bills_table || []).filter(row => row.route_key === routeKey && (!topic || row.topic === topic));
  const list = rows.slice(0, 20).map(row => `
    <div class="mini-list-item">
      <strong>${row.bill_ID}</strong>
      <div>${row.topic || 'Unknown topic'}</div>
      <div class="caption">Versions: ${row.n_versions || 0} • Risk: ${row.risk_score || 0}</div>
    </div>
  `).join('');
  openDrawer(`Bills on ${routeKey}`, `<div class="mini-list">${list || '<p>No bills for this route.</p>'}</div>`);
}

function renderAmendmentScatter() {
  const data = filterData(precomp.amendment_churn || []);
  if (!data.length) {
    Plotly.react('amendment-scatter', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No amendment data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const trace = {
    type: 'scattergl',
    mode: 'markers',
    x: data.map(d => Number(d.n_versions) || 0),
    y: data.map(d => Number(d.median_sim) || 0),
    text: data.map(d => d.bill_id),
    marker: { color: data.map(d => Number(d.final_similarity) || 0), size: 9, colorscale: 'Viridis', showscale: true }
  };
  Plotly.react('amendment-scatter', [trace], {
    xaxis: { title: 'Number of versions' },
    yaxis: { title: 'Median similarity', range: [0, 1], tickformat: '.0%' },
    margin: { l: 50, r: 20, t: 20, b: 50 }
  });
}

function renderTokenLift() {
  const data = filterData(precomp.text_lift_top_tokens || []);
  const filtered = data.filter(row => (Number(row.count) || 0) >= state.tokenMin);
  if (!filtered.length) {
    Plotly.react('token-bars', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['Increase token threshold or adjust filters.'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const sorted = filtered.sort((a, b) => (Number(a.log_lift_pass_vs_other) || 0) - (Number(b.log_lift_pass_vs_other) || 0));
  const trace = {
    type: 'bar',
    orientation: 'h',
    x: sorted.map(row => Number(row.log_lift_pass_vs_other) || 0),
    y: sorted.map(row => row.token),
    marker: {
      color: sorted.map(row => (Number(row.log_lift_pass_vs_other) || 0) >= 0 ? '#16a34a' : '#dc2626'),
      opacity: sorted.map(row => Math.min(1, 0.4 + (Number(row.count) || 0) / 50))
    }
  };
  Plotly.react('token-bars', [trace], {
    margin: { l: 160, r: 20, t: 10, b: 10 },
    xaxis: { title: 'Log lift (pass vs other)' },
    yaxis: { automargin: true }
  });
}

function renderGatekeeping() {
  const data = filterData(precomp.committee_gatekeeping || []);
  if (!data.length) {
    Plotly.react('gatekeeping-lollipop', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No committee data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const sorted = [...data].sort((a, b) => (Number(b.gatekeeping) || 0) - (Number(a.gatekeeping) || 0));
  const trace = {
    type: 'bar',
    orientation: 'h',
    x: sorted.map(row => Number(row.gatekeeping) || 0),
    y: sorted.map(row => row.committee),
    marker: { color: '#ef4444' }
  };
  Plotly.react('gatekeeping-lollipop', [trace], {
    margin: { l: 220, r: 20, t: 20, b: 20 },
    xaxis: { title: 'Gatekeeping rate', tickformat: '.0%' },
    yaxis: { automargin: true }
  });
}

function renderCommitteeWorkload() {
  const data = filterData(precomp.committee_workload || []);
  if (!data.length) {
    Plotly.react('committee-workload', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No workload data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const trace = {
    type: 'scatter',
    mode: 'markers',
    x: data.map(row => Number(row.unique_bills) || 0),
    y: data.map(row => Number(row.gatekeeping) || 0),
    text: data.map(row => row.committee),
    marker: { size: data.map(row => 6 + Math.sqrt(Number(row.bills_heard) || 0)), color: '#2563eb', opacity: 0.7 }
  };
  Plotly.react('committee-workload', [trace], {
    margin: { l: 60, r: 20, t: 20, b: 50 },
    xaxis: { title: 'Distinct bills processed' },
    yaxis: { title: 'Gatekeeping', tickformat: '.0%' }
  });
}

function renderDriftBeeswarm() {
  const data = filterData(precomp.committee_floor_drift || []);
  if (!data.length) {
    Plotly.react('drift-beeswarm', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No drift data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const traces = d3.groups(data, d => d.chamber || 'Both').map(([label, rows]) => ({
    type: 'scatter',
    mode: 'markers',
    name: label,
    x: rows.map(row => Number(row.drift) || 0),
    y: rows.map(() => label),
    marker: { size: 10, opacity: 0.7 },
    text: rows.map(row => row.legislator_name || row.legislator)
  }));
  Plotly.react('drift-beeswarm', traces, {
    margin: { l: 80, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Floor minus committee yes-rate', tickformat: '.0%' },
    yaxis: { automargin: true }
  });
}

function renderVotingNetwork() {
  const edges = filterData(precomp.vote_similarity_edges || []);
  const svg = d3.select('#voting-network');
  svg.selectAll('*').remove();
  if (!edges.length) {
    svg.append('text').attr('x', 20).attr('y', 20).text('No network data for selection.');
    return;
  }
  const width = svg.node().clientWidth || 800;
  const height = svg.node().clientHeight || 420;
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  const nodesMap = new Map();
  edges.forEach(edge => {
    if (!nodesMap.has(edge.u)) nodesMap.set(edge.u, { id: edge.u });
    if (!nodesMap.has(edge.v)) nodesMap.set(edge.v, { id: edge.v });
  });
  const nodes = Array.from(nodesMap.values());
  const linkData = edges.map(edge => ({ source: edge.u, target: edge.v, sim: edge.sim }));
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(linkData).id(d => d.id).distance(140).strength(0.6))
    .force('charge', d3.forceManyBody().strength(-220))
    .force('center', d3.forceCenter(width / 2, height / 2));
  const link = svg.append('g').attr('stroke', '#cbd5f5').attr('stroke-opacity', 0.6).selectAll('line').data(linkData).enter().append('line')
    .attr('stroke-width', d => 1 + (Number(d.sim) || 0.6));
  const node = svg.append('g').attr('stroke', '#fff').attr('stroke-width', 1.5)
    .selectAll('circle').data(nodes).enter().append('circle')
    .attr('r', 6)
    .attr('fill', '#4f46e5')
    .call(d3.drag()
      .on('start', event => { if (!event.active) simulation.alphaTarget(0.3).restart(); event.subject.fx = event.subject.x; event.subject.fy = event.subject.y; })
      .on('drag', event => { event.subject.fx = event.x; event.subject.fy = event.y; })
      .on('end', event => { if (!event.active) simulation.alphaTarget(0); event.subject.fx = null; event.subject.fy = null; }));
  const labels = svg.append('g').selectAll('text').data(nodes).enter().append('text')
    .attr('font-size', 10)
    .attr('fill', '#1f2937')
    .text(d => d.id);
  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    labels.attr('x', d => d.x + 8).attr('y', d => d.y + 4);
  });
}

function renderSurvivalCurves() {
  const data = filterData(precomp.survival_curves || []);
  if (!data.length) {
    Plotly.react('survival-curves', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No survival data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const traces = d3.groups(data, d => d.topic).map(([topic, rows]) => ({
    type: 'scatter',
    mode: 'lines',
    name: topic,
    x: rows.map(row => row.date),
    y: rows.map(row => Number(row.survival) || 0)
  }));
  Plotly.react('survival-curves', traces, {
    margin: { l: 60, r: 20, t: 20, b: 50 },
    yaxis: { title: 'Probability alive', range: [0, 1], tickformat: '.0%' },
    xaxis: { title: 'Date' }
  });
}

function renderControversyHeatmap() {
  const data = precomp.rollcall_party_splits || [];
  if (!data.length) {
    Plotly.react('controversy-heatmap', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No roll call data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const terms = uniqueValues(data, row => row.term);
  const topics = uniqueValues(data, row => row.topic);
  const z = topics.map(topic => terms.map(term => {
    const match = data.find(row => row.topic === topic && row.term === term);
    return match ? Number(match.polarization) || 0 : 0;
  }));
  Plotly.react('controversy-heatmap', [{
    type: 'heatmap',
    x: terms,
    y: topics,
    z,
    colorscale: 'RdBu',
    reversescale: true
  }], {
    margin: { l: 140, r: 20, t: 20, b: 60 },
    xaxis: { title: 'Term' },
    yaxis: { automargin: true }
  });
}

function renderDumbbell() {
  const data = precomp.rollcall_party_splits || [];
  if (!data.length) {
    Plotly.react('controversy-dumbbell', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No party comparison data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const top = [...data].sort((a, b) => (Number(b.polarization) || 0) - (Number(a.polarization) || 0)).slice(0, 10);
  const y = top.map(row => row.topic);
  const dem = top.map(row => Number(row.dem_yes) || 0);
  const rep = top.map(row => Number(row.rep_yes) || 0);
  const line = { type: 'scatter', mode: 'lines', x: dem.concat(rep), y: y.concat(y), line: { color: '#d1d5db' }, showlegend: false };
  const demTrace = { type: 'scatter', mode: 'markers', name: 'Democratic yes', x: dem, y, marker: { color: '#2563eb', size: 10 } };
  const repTrace = { type: 'scatter', mode: 'markers', name: 'Republican yes', x: rep, y, marker: { color: '#f97316', size: 10 } };
  Plotly.react('controversy-dumbbell', [line, demTrace, repTrace], {
    margin: { l: 160, r: 20, t: 20, b: 40 },
    xaxis: { title: 'Yes-rate', tickformat: '.0%' },
    yaxis: { automargin: true },
    legend: { orientation: 'h', y: -0.2 }
  });
}

function renderFundingMap() {
  const geo = precomp.ca_legislator_funding_geo;
  const table = precomp.ca_legislator_funding || [];
  if (!geo || !geo.features || !geo.features.length) {
    document.getElementById('funding-map').innerHTML = '<p style="padding:16px">Geography unavailable.</p>';
    return;
  }
  if (!mapInstance) {
    mapInstance = L.map('funding-map', { scrollWheelZoom: false }).setView([37.5, -119.5], 5.7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors' }).addTo(mapInstance);
  }
  if (mapLayer) mapLayer.remove();
  const metric = elements.fundingMetric.value || 'total_received';
  const valueMap = new Map();
  table.forEach(row => {
    const key = (row.geography || row.beneficiary_lower || '').toLowerCase();
    if (!key) return;
    valueMap.set(key, (valueMap.get(key) || 0) + (Number(row[metric]) || 0));
  });
  const values = Array.from(valueMap.values());
  const color = values.length ? d3.scaleSequential(d3.interpolateBlues).domain(d3.extent(values)) : () => '#e5e7eb';
  mapLayer = L.geoJSON(geo, {
    style: feature => {
      const key = (feature.properties?.name || feature.properties?.district || '').toLowerCase();
      const val = valueMap.get(key) || 0;
      return { color: '#ffffff', weight: 1, fillColor: color(val), fillOpacity: 0.85 };
    },
    onEachFeature: (feature, layer) => {
      const key = (feature.properties?.name || feature.properties?.district || '').toLowerCase();
      const val = valueMap.get(key) || 0;
      layer.bindTooltip(`${feature.properties?.name || feature.properties?.district || 'District'}: $${val.toLocaleString()}`);
      layer.on('click', () => {
        state.selectedCounty = feature.properties?.name || feature.properties?.district || null;
        updateFilterPills();
        renderCountyDetail(metric);
      });
    }
  }).addTo(mapInstance);
  renderCountyDetail(metric);
}

function renderCountyDetail(metric) {
  const table = precomp.ca_legislator_funding || [];
  const target = (state.selectedCounty || '').toLowerCase();
  const rows = table.filter(row => (row.geography || row.beneficiary_lower || '').toLowerCase() === target);
  if (!rows.length) {
    elements.fundingCountyDetail.innerHTML = '<p>Select a geography on the map to see recipients.</p>';
    return;
  }
  const list = rows.sort((a, b) => (Number(b[metric]) || 0) - (Number(a[metric]) || 0)).slice(0, 10).map(row => `
    <div class="mini-list-item">
      <strong>${row.beneficiary_lower}</strong>
      <div>${metric.replace(/_/g, ' ')}: $${(Number(row[metric]) || 0).toLocaleString()}</div>
    </div>
  `).join('');
  elements.fundingCountyDetail.innerHTML = `<div class="mini-list">${list}</div>`;
}

function renderTopicFundingTerm() {
  const data = filterData(precomp.topic_funding_by_term || []);
  if (!data.length) {
    Plotly.react('topic-funding-term', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['No funding data'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const terms = Array.from(new Set(data.map(row => row.term))).sort();
  const donations = terms.map(term => d3.sum(data.filter(row => row.term === term), row => Number(row.total_donations) || 0));
  const lobbying = terms.map(term => d3.sum(data.filter(row => row.term === term), row => Number(row.total_lobbying) || 0));
  Plotly.react('topic-funding-term', [
    { type: 'bar', name: 'Donations', x: terms, y: donations, marker: { color: '#3b82f6' } },
    { type: 'bar', name: 'Lobbying', x: terms, y: lobbying, marker: { color: '#f97316' } }
  ], {
    barmode: 'stack',
    margin: { l: 60, r: 20, t: 20, b: 60 },
    yaxis: { title: 'Amount ($)' }
  });
}

function renderLegislatorTopicMix() {
  const data = precomp.topic_funding_by_leg || [];
  const legislator = state.selectedLegislator;
  const filtered = legislator ? data.filter(row => row.beneficiary_lower === legislator) : [];
  if (!filtered.length) {
    Plotly.react('legislator-topic-mix', [{ type: 'scatter', mode: 'text', x: [0], y: [0], text: ['Select a legislator to view mix.'] }], { xaxis: { visible: false }, yaxis: { visible: false } });
    return;
  }
  const topics = filtered.map(row => row.topic);
  const totals = filtered.map(row => Number(row.total) || 0);
  Plotly.react('legislator-topic-mix', [{ type: 'bar', x: topics, y: totals, marker: { color: '#22c55e' } }], {
    margin: { l: 60, r: 20, t: 20, b: 80 },
    yaxis: { title: 'Allocated funding ($)' }
  });
}

function renderBillsTable() {
  const rows = filterData(precomp.bills_table || []);
  const tableEl = document.getElementById('bill-table');
  if (!billsTableInstance) {
    billsTableInstance = new DataTable(tableEl, {
      data: rows,
      columns: [
        { data: 'bill_ID', title: 'Bill' },
        { data: 'topic', title: 'Topic' },
        { data: 'term', title: 'Term' },
        { data: 'first_action_date', title: 'First action' },
        { data: 'longevity_days', title: 'Longevity (days)' },
        { data: 'n_versions', title: 'Versions' },
        { data: 'median_sim', title: 'Median similarity', render: data => data ? `${(Number(data) * 100).toFixed(1)}%` : '—' },
        { data: 'risk_score', title: 'Risk score' }
      ],
      pageLength: 25,
      responsive: true
    });
  } else {
    billsTableInstance.clear();
    billsTableInstance.rows.add(rows);
    billsTableInstance.draw();
  }
}

function renderAll() {
  renderRouteTopicChips();
  renderPipelineSankey();
  renderStageDurations();
  renderRiskRegister();
  renderRouteArchetypes();
  renderAmendmentScatter();
  renderTokenLift();
  renderGatekeeping();
  renderCommitteeWorkload();
  renderDriftBeeswarm();
  renderVotingNetwork();
  renderSurvivalCurves();
  renderControversyHeatmap();
  renderDumbbell();
  renderFundingMap();
  renderTopicFundingTerm();
  renderLegislatorTopicMix();
  renderBillsTable();
}

function initializeFilters() {
  const bills = precomp.bills_table || [];
  initializeSelect(elements.termSelect, uniqueValues(bills, row => row.term));
  initializeSelect(elements.topicSelect, uniqueValues(bills, row => row.topic), true);
  elements.tokenSlider.value = state.tokenMin;
  elements.tokenSliderValue.textContent = `${state.tokenMin}+`;
  const legislatorOptions = uniqueValues(precomp.topic_funding_by_leg || [], row => row.beneficiary_lower);
  initializeSelect(elements.legislatorSelect, legislatorOptions);
  elements.legislatorSelect.insertAdjacentHTML('afterbegin', '<option value="">Select a legislator</option>');
  elements.legislatorSelect.value = '';
  elements.routeSelection.hidden = true;
  renderRouteTopicChips();
  updateChamberButtons();
  updateFilterPills();
}

initializeFilters();
renderAll();
