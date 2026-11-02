/* 前端主逻辑：提问 → SSE 订阅四智能体过程 → 渲染报告 / 指标结果表 / 图表看板
 *           + 补充面板（横向对比 / 异常检测）。
 *
 * 与后端契约（api_server.py）：
 *   POST /chat/stream  text/event-stream，逐条 data: {...}
 *     {"type":"step","agent":"检索员","status":"running"|"done", ...detail}
 *     {"type":"retry","round":n,"feedback":[...]}
 *     {"type":"final","data":{"review_status","quality_score","review_rounds"}}
 *     {"type":"complete","domain","answer","report","indicators","analytics",
 *      "compare","anomalies","review","steps","review_rounds"}
 *     {"type":"error","error":"..."}
 *   GET  /health  关系库与知识库可用性
 *
 * 两个补充面板（报告之外，前端单列）：
 *   compare   {available, reason, label, unit, stat_date, n_entries,
 *              entries:[{label,value}], max, min, mean, max_entity, min_entity}
 *   anomalies {available, reason, scope, summary, total, shown, truncated,
 *              items:[{kind,kind_label,severity,ts,value,detail,scope}]}
 */

import { buildChartPanel, el, fmtVal } from './charts.js';

const AGENTS = [
  { key: '检索员', role: '意图解析 + 知识库检索' },
  { key: '分析师', role: '指标核算 + 统计分析' },
  { key: '撰稿人', role: '报告撰写' },
  { key: '审核员', role: '口径校验' },
];

const MODE_LABEL = {
  daily: '日值',
  mom: '环比',
  cumulative: '累计',
  evaluation: '日评价值',
  cumulative_evaluation: '累计评价值',
  degree_day: '度日数折标',
  season_cumulative: '标准采暖季累计',
  season_cumulative_evaluation: '标准采暖季累计评价值',
  season: '季',
};

const EXAMPLES = [
  '公司A今天的气耗指标是多少？',
  '项目5的本季收费率是多少？居民和非居民分别是多少？',
  '项目3的本季收费率是多少？居民和非居民分别是多少？',
  'L4换热站昨天的二次侧供回水温差正常吗？',
  '度日数折标后的日气耗量怎么计算？',
];

const $ = (id) => document.getElementById(id);

const ui = {
  form: null, input: null, askBtn: null, stopBtn: null,
  chips: null, agents: null, review: null,
  report: null, empty: null, paneRight: null, pills: null,
  process: null, processBody: null, processHint: null, processToggle: null,
};

let pipeline = [];      // 左栏过程日志：[{kind:'agent'|'retry', ...}]
let controller = null;  // 用于「中断」
let roleModels = {};    // 角色中文名 → {role,label,model,overridden}，来自 /health
let llmEnabled = true;  // /health 的 use_llm：关闭大模型时各角色行不该显示"在用某模型"
let procBlocks = new Map();  // 右栏「思考过程」里 角色 → {el, lines, stream, streamed}
let procSteps = 0;           // 已产生的过程条数，用于提示文案

/* ============================ 启动 ============================ */

function init() {
  ui.form = $('askForm');
  ui.input = $('question');
  ui.askBtn = $('askBtn');
  ui.stopBtn = $('stopBtn');
  ui.chips = $('exampleChips');
  ui.agents = $('agents');
  ui.review = $('reviewCard');
  ui.report = $('report');
  ui.empty = $('emptyState');
  ui.paneRight = $('reportPane');
  ui.pills = $('statusPills');
  ui.process = $('process');
  ui.processBody = $('processBody');
  ui.processHint = $('processHint');
  ui.processToggle = $('processToggle');

  EXAMPLES.forEach((q) => {
    ui.chips.appendChild(el('button', {
      class: 'chip', type: 'button', text: q, title: '点击直接开始分析',
      onclick: () => {
        if (ui.askBtn.disabled) return;   // 分析进行中，避免并发发起第二条
        ui.input.value = q;
        ui.form.requestSubmit();          // 等价于点「开始分析」
      },
    }));
  });

  ui.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = ui.input.value.trim();
    if (q) ask(q);
  });
  ui.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      ui.form.requestSubmit();
    }
  });
  ui.stopBtn.addEventListener('click', () => {
    if (controller) controller.abort();
  });
  ui.processToggle.addEventListener('click', () => {
    const collapsed = ui.process.classList.toggle('is-collapsed');
    ui.processToggle.textContent = collapsed ? '展开' : '收起';
    ui.processToggle.setAttribute('aria-expanded', String(!collapsed));
  });

  resetPipeline();
  checkHealth();
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const db = d.database || {};
    const kb = d.knowledge_base || {};
    const models = [].concat(d.models || []);
    roleModels = Object.fromEntries(models.map((m) => [m.label, m]));
    llmEnabled = d.use_llm !== false;
    const pills = [
      pill(`关系库 ${db.available ? '可用' : '不可用'}`, db.available ? 'is-ok' : 'is-bad'),
      pill(
        kb.available ? `知识库 ${kb.chunk_count} 分片` : '知识库不可用',
        kb.available ? 'is-ok' : 'is-bad',
      ),
    ];
    const mp = modelsPill(models, d.use_llm);
    if (mp) pills.push(mp);
    ui.pills.replaceChildren(...pills);
    renderPipeline();       // 让「协作过程」尽早带上各角色的模型名
  } catch (e) {
    ui.pills.replaceChildren(pill('后端未连接', 'is-bad'));
  }
}

/** 顶栏「模型」指示：一眼看出是统一模型还是按角色区分，悬停看完整映射。 */
function modelsPill(models, useLlm) {
  if (!models.length) return null;
  const distinct = [...new Set(models.map((m) => m.model).filter(Boolean))];
  const text = !useLlm
    ? '模型 未启用'
    : (distinct.length <= 1 ? `模型 统一：${distinct[0] || '—'}` : `模型 按角色：${distinct.length} 种`);
  const p = pill(text, '');
  p.title = models
    .map((m) => `${m.label}：${m.model}${m.overridden ? '（单独配置）' : ''}`)
    .join('\n');
  return p;
}

function pill(text, cls) {
  return el('span', { class: `pill ${cls || ''}` }, [
    el('span', { class: 'pill-dot', 'aria-hidden': 'true' }),
    el('span', { text }),
  ]);
}

/* ============================ 提问流程 ============================ */

async function ask(question) {
  resetPipeline();
  resetProcess();
  ui.askBtn.disabled = true;
  ui.stopBtn.hidden = false;
  ui.input.blur();

  // 上一份报告先撤掉：本轮的过程流要占主区，结束后再在下方渲染新报告
  ui.report.hidden = true;
  ui.process.hidden = false;
  ui.empty.hidden = true;
  ui.review.hidden = true;
  ui.paneRight.scrollTop = 0;

  controller = new AbortController();
  try {
    await streamAsk(question, controller.signal);
  } catch (err) {
    if (err.name === 'AbortError') {
      showError('已中断本次分析。');
    } else {
      showError(`分析失败：${err.message}`);
    }
  } finally {
    ui.askBtn.disabled = false;
    ui.stopBtn.hidden = true;
    controller = null;
    ui.report.classList.remove('is-loading');
  }
}

async function streamAsk(question, signal) {
  const resp = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const j = await resp.json();
      if (j && j.error) msg = j.error;
    } catch (e) { /* 保持默认 */ }
    throw new Error(msg);
  }
  if (!resp.body) throw new Error('当前浏览器不支持流式读取');

  const reader = resp.body.getReader();
  const dec = new TextDecoder('utf-8');
  let buf = '';

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });

    let cut;
    while ((cut = buf.indexOf('\n\n')) !== -1) {
      const chunk = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      // 忽略 ": ping" 心跳注释帧
      const data = chunk.split('\n')
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trimStart())
        .join('\n');
      if (!data) continue;
      handleEvent(JSON.parse(data));
    }
  }
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'step':
      onStep(ev);
      break;
    case 'trace':
      addTrace(ev);
      break;
    case 'stream':
      addStream(ev);
      break;
    case 'retry':
      pipeline.push({ kind: 'retry', round: ev.round, feedback: ev.feedback || [] });
      renderPipeline();
      addProcessDivider(`第 ${ev.round} 轮重算：${(ev.feedback || []).join('；') || '按审核意见重做'}`);
      break;
    case 'final':
      onFinal(ev.data || {});
      break;
    case 'complete':
      renderReport(ev);
      break;
    case 'error':
      showError(`分析失败：${ev.error || '未知错误'}`);
      break;
    default:
      break;
  }
}

function onStep(ev) {
  const item = pipeline.find((x) => x.kind === 'agent' && x.key === ev.agent);
  if (!item) return;
  item.status = ev.status;
  if (ev.model !== undefined) item.model = ev.model;
  if (ev.status === 'done') item.detail = stepDetail(ev);
  renderPipeline();
  // 右栏过程流同步：开块 / 收块
  if (ev.status === 'running') openProcessBlock(ev.agent, ev.model);
  else if (ev.status === 'done') closeProcessBlock(ev.agent, stepDetail(ev));
}

function onFinal(data) {
  const item = pipeline.find((x) => x.kind === 'agent' && x.key === '审核员');
  if (item) item.review = data;
  renderPipeline();
}

function stepDetail(ev) {
  if (ev.agent === '检索员') {
    const bits = [`数据域 ${ev.domain || '—'}`];
    if (ev.metrics && ev.metrics.length) bits.push(`指标 ${ev.metrics.join('/')}`);
    if (ev.stat_date) bits.push(`日期 ${ev.stat_date}`);
    if (ev.knowledge_hits !== undefined) bits.push(`知识库命中 ${ev.knowledge_hits} 条`);
    return bits.join(' · ');
  }
  if (ev.agent === '分析师') {
    if (ev.error) return ev.error;
    const bits = [`指标 ${ev.n_indicators || 0} 项（可用 ${ev.n_available || 0}）`];
    if (ev.analytics_keys && ev.analytics_keys.length) {
      bits.push(`统计分析 ${ev.analytics_keys.join('/')}`);
    }
    return bits.join(' · ');
  }
  if (ev.agent === '撰稿人') {
    const bits = [`《${ev.title || '—'}》`];
    if (ev.sections) bits.push(`分节 ${ev.sections.length}`);
    bits.push(`图表 ${ev.n_charts || 0}`);
    bits.push(`来源 ${ev.n_sources || 0}`);
    if (ev.n_caveats) bits.push(`口径说明 ${ev.n_caveats} 项`);
    return bits.join(' · ');
  }
  if (ev.agent === '审核员') {
    const ok = ev.status_text === 'approved';
    const bits = [ok ? '审核通过' : '审核未通过', `评分 ${ev.score}`];
    if (ev.issues && ev.issues.length) bits.push(`问题 ${ev.issues.length} 项`);
    return bits.join(' · ');
  }
  return '';
}

/* ============================ 协作过程 ============================ */

function resetPipeline() {
  pipeline = AGENTS.map((a) => ({ kind: 'agent', key: a.key, role: a.role, status: 'idle', detail: '' }));
  renderPipeline();
}

function renderPipeline() {
  const ol = el('ol', { class: 'agents' });
  for (const item of pipeline) {
    if (item.kind === 'retry') {
      const fb = item.feedback.length ? `：${item.feedback.join('；')}` : '';
      ol.appendChild(el('li', {
        class: 'retry-note',
        text: `第 ${item.round} 轮重算${fb}`,
      }));
      continue;
    }
    const cls = item.status === 'done'
      ? (item.review && item.review.review_status === 'rejected' ? 'is-warn' : 'is-done')
      : (item.status === 'running' ? 'is-running' : 'is-idle');
    const mark = item.status === 'done'
      ? (item.review && item.review.review_status === 'rejected' ? cross() : check())
      : null;
    const stateText = item.status === 'running' ? '进行中'
      : (item.status === 'done' ? '完成' : '等待');
    // 模型名：优先用该次运行实际用到的（step 事件带回），否则用 /health 里查到的；
    // 大模型整体关闭时不显示具体模型名，避免让人以为它真的被调用了
    const modelInfo = roleModels[item.key];
    const model = llmEnabled
      ? (item.model || (modelInfo ? modelInfo.model : '') || '—')
      : '模型未启用';

    ol.appendChild(el('li', { class: `agent ${cls}` }, [
      el('span', { class: 'agent-dot', 'aria-hidden': 'true' }, [mark]),
      el('div', {}, [
        el('div', { class: 'agent-name' }, [
          el('span', { text: item.key }),
          el('span', { class: 'agent-role', text: item.role }),
          el('span', {
            class: `agent-model${modelInfo && modelInfo.overridden ? ' is-override' : ''}`,
            text: model || '—',
            title: '该角色使用的模型（可在 .env 里用 LLM_MODEL_* 单独配置）',
          }),
          el('span', { class: 'agent-state', text: stateText, style: 'margin-left:auto' }),
        ]),
        item.detail ? el('div', { class: 'agent-detail', text: item.detail }) : null,
      ]),
    ]));
  }
  ui.agents.replaceChildren(ol);
}

function check() {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 12 12');
  s.setAttribute('width', '10');
  s.setAttribute('height', '10');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', 'M2.5 6.4 L4.8 8.6 L9.5 3.6');
  p.setAttribute('fill', 'none');
  p.setAttribute('stroke', 'var(--surface-1)');
  p.setAttribute('stroke-width', '1.8');
  p.setAttribute('stroke-linecap', 'round');
  p.setAttribute('stroke-linejoin', 'round');
  s.appendChild(p);
  return s;
}

function cross() {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 12 12');
  s.setAttribute('width', '9');
  s.setAttribute('height', '9');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', 'M3 3 L9 9 M9 3 L3 9');
  p.setAttribute('stroke', 'var(--surface-1)');
  p.setAttribute('stroke-width', '1.8');
  p.setAttribute('stroke-linecap', 'round');
  s.appendChild(p);
  return s;
}

/* ==================== 右栏「思考过程」流 ==================== */
/* 数据来自 SSE 的 trace（解析结果/指标/校验/分节）与 stream（模型逐字输出）两类事件。
   四个角色各占一块，运行中持续追加；跑完后报告接在本面板下方。 */

const LEVEL_CLASS = { ok: 'is-ok', warn: 'is-warn', info: '' };

function resetProcess() {
  procBlocks = new Map();
  procSteps = 0;
  ui.processBody.replaceChildren();
  ui.processHint.textContent = '四智能体协作中…';
  ui.process.classList.remove('is-collapsed');
  ui.processToggle.textContent = '收起';
  ui.processToggle.setAttribute('aria-expanded', 'true');
}

/** 取（或新建）某个角色的过程块。 */
function procBlock(agent) {
  if (procBlocks.has(agent)) return procBlocks.get(agent);

  const lines = el('ul', { class: 'plines' });
  const stream = el('div', { class: 'pstream' });
  stream.hidden = true;
  const role = (AGENTS.find((a) => a.key === agent) || {}).role || '';
  const info = roleModels[agent];
  const model = llmEnabled ? ((info && info.model) || '') : '模型未启用';

  const stateEl = el('span', { class: 'pblock-state', text: '进行中' });
  const block = el('div', { class: 'pblock is-running' }, [
    el('div', { class: 'pblock-head' }, [
      el('span', { class: 'pblock-dot', 'aria-hidden': 'true' }),
      el('span', { class: 'pblock-name', text: agent }),
      el('span', { class: 'pblock-role', text: role }),
      model ? el('span', { class: 'pblock-model', text: model, title: '该角色使用的模型' }) : null,
      stateEl,
    ]),
    lines,
    stream,
  ]);
  const rec = { el: block, lines, stream, streamed: false, stateEl };
  procBlocks.set(agent, rec);
  ui.processBody.appendChild(block);
  return rec;
}

function openProcessBlock(agent, model) {
  const rec = procBlock(agent);
  rec.el.classList.add('is-running');
  rec.el.classList.remove('is-done');
  rec.stateEl.textContent = '进行中';
  if (model) {
    const m = rec.el.querySelector('.pblock-model');
    if (m) m.textContent = model;
  }
  procSteps += 1;
  scrollProcess();
}

function closeProcessBlock(agent, detail) {
  const rec = procBlocks.get(agent);
  if (!rec) return;
  rec.el.classList.remove('is-running');
  rec.el.classList.add('is-done');
  rec.stateEl.textContent = '完成';
  // 流式区保留 —— 它是模型的实际产出，收块时不该抹掉（空的时候本来就没显示）
  if (detail) addProcessLine(rec, detail, 'info');
  procSteps += 1;
  scrollProcess();
}

function addProcessLine(rec, text, level) {
  rec.lines.appendChild(el('li', { class: LEVEL_CLASS[level] || '' }, [inline(text)]));
  procSteps += 1;
}

function addTrace(ev) {
  const rec = procBlock(ev.agent);
  addProcessLine(rec, ev.text || '', ev.level);
  scrollProcess();
}

function addStream(ev) {
  const rec = procBlock(ev.agent);
  if (!rec.streamed) {
    rec.streamed = true;
    rec.stream.hidden = false;
    rec.stream.appendChild(el('span', { class: 'pstream-label', text: '模型输出' }));
    rec.stream.appendChild(el('span', { class: 'pstream-text' }));
  }
  const t = rec.stream.querySelector('.pstream-text');
  t.textContent += ev.text || '';           // 逐字追加
  scrollProcess();
}

function addProcessDivider(text) {
  procBlocks = new Map();                   // 重算轮次另起一组块
  ui.processBody.appendChild(el('div', { class: 'pdivider' }, [
    el('span', { text: `↻ ${text}` }),
  ]));
  scrollProcess();
}

/** 贴近底部时才自动跟随，避免用户往回翻看时被强行拽回。 */
function scrollProcess() {
  const box = ui.paneRight;
  if (!box) return;
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function endProcess(hint) {
  ui.processHint.textContent = hint;
  procBlocks.forEach((rec) => {
    if (rec.el.classList.contains('is-running')) {
      rec.el.classList.remove('is-running');
      rec.el.classList.add('is-done');
      rec.stateEl.textContent = '完成';
    }
  });
}

/* ============================ 报告渲染 ============================ */

function renderReport(p) {
  const rep = p.report;
  if (!rep) {
    showError('后端未返回报告内容。');
    return;
  }
  const frag = document.createDocumentFragment();

  // —— 头部（标题 + 审核结论徽标）
  const head = el('div', { class: 'report-head' }, [
    el('h2', { text: rep.title || '分析报告' }),
    rep.subtitle ? el('p', { class: 'report-sub', text: rep.subtitle }) : null,
    el('p', { class: 'report-meta', text: `生成时间 ${rep.generated_at || '—'} · 数据域 ${p.domain || '—'}` }),
  ]);
  const verdict = verdictBadge(p.review, p.review_rounds);
  if (verdict) head.appendChild(verdict);
  frag.appendChild(head);

  const indicators = [].concat(p.indicators || []);
  const charts = [].concat(rep.charts || []);
  const sections = rep.sections || {};
  let chartsPlaced = false;

  const placeCharts = () => {
    if (chartsPlaced || !charts.length) return;
    chartsPlaced = true;
    frag.appendChild(el('div', { class: 'section' }, [
      el('h3', { text: indicators.length ? '图表看板' : '图表' }),
      buildChartPanel(charts, { unitOfLabel: unitOf(indicators) }),
    ]));
  };

  for (const [heading, lines] of Object.entries(sections)) {
    // 指标数据：有结构化指标时改由「指标结果表」呈现（信息是它的超集）
    if (heading === '指标数据' && indicators.length) {
      frag.appendChild(el('div', { class: 'section' }, [
        el('h3', { text: '指标结果表' }),
        indicatorsTable(indicators),
        reviewBreakdown(p.review),
      ]));
      placeCharts();
      continue;
    }
    if (!lines || !lines.length) continue;

    const sec = el('div', { class: `section${heading === '结论' ? ' section-conclusion' : ''}` }, [
      el('h3', { text: heading }),
    ]);
    const tableLines = lines.filter((l) => String(l).trim().startsWith('|'));
    if (tableLines.length === lines.length) {
      sec.appendChild(mdTable(lines));
    } else {
      const ul = el('ul');
      lines.forEach((l) => ul.appendChild(el('li', {}, [inline(l)])));
      sec.appendChild(ul);
    }
    frag.appendChild(sec);
  }
  placeCharts();

  // —— 补充面板：横向对比、异常检测（报告正文不呈现，故单列）
  const cmpSec = comparePanel(p.compare);
  if (cmpSec) frag.appendChild(cmpSec);
  const anSec = anomalyPanel(p.anomalies);
  if (anSec) frag.appendChild(anSec);

  // —— 口径与数据说明
  if (rep.caveats && rep.caveats.length) {
    const box = el('div', { class: 'caveat' }, [
      el('div', { class: 'caveat-head' }, [warnIcon(), el('span', { text: '口径与数据说明' })]),
      el('ul', {}, rep.caveats.map((c) => el('li', {}, [inline(c)]))),
    ]);
    frag.appendChild(el('div', { class: 'section' }, [box]));
  }

  // —— 数据来源
  if (rep.sources && rep.sources.length) {
    const list = el('ul', { class: 'sources' });
    rep.sources.forEach((s, i) => {
      list.appendChild(el('li', {}, [
        el('span', { class: 'idx', text: `[${i + 1}]` }),
        el('span', { text: s }),
      ]));
    });
    frag.appendChild(el('div', { class: 'section' }, [
      el('h3', { text: '数据来源' }),
      list,
    ]));
  }

  // —— Markdown 工具
  const tools = el('div', { class: 'md-tools' });
  const raw = el('pre', { class: 'raw-md', text: p.answer || '' });
  raw.hidden = true;
  const rawBtn = el('button', {
    class: 'btn btn-sm', type: 'button', 'aria-expanded': 'false', text: 'Markdown 原文',
    onclick: () => {
      raw.hidden = !raw.hidden;
      rawBtn.setAttribute('aria-expanded', String(!raw.hidden));
      rawBtn.textContent = raw.hidden ? 'Markdown 原文' : '收起原文';
    },
  });
  tools.append(
    el('button', {
      class: 'btn btn-sm', type: 'button', text: '复制 Markdown',
      onclick: async (e) => {
        try {
          await navigator.clipboard.writeText(p.answer || '');
          e.target.textContent = '已复制';
        } catch (err) {
          e.target.textContent = '复制失败';
        }
        setTimeout(() => { e.target.textContent = '复制 Markdown'; }, 1600);
      },
    }),
    el('button', {
      class: 'btn btn-sm', type: 'button', text: '下载 .md',
      onclick: () => downloadMd(p),
    }),
    rawBtn,
  );
  frag.append(tools, raw);

  ui.report.replaceChildren(frag);
  ui.report.hidden = false;
  ui.empty.hidden = true;

  // 过程流留在上方（可折叠），报告接在其下方
  const rounds = p.review_rounds || 0;
  endProcess(`已完成：${procSteps} 步过程${rounds ? `·重算 ${rounds} 轮` : ''}；报告见下方`);
  if (!ui.process.hidden) {
    ui.report.scrollIntoView({ block: 'start', behavior: 'auto' });
  } else {
    ui.paneRight.scrollTop = 0;      // 过程面板被关掉时（如直接渲染）退回顶部
  }

  renderReviewCard(p);
}

function unitOf(indicators) {
  const map = new Map(indicators.map((r) => [r.label, r.unit || '']));
  return (label) => {
    const u = map.get(label);
    return u && !String(u).startsWith('按源') ? u : '';
  };
}

function verdictBadge(review, rounds) {
  const r = review || {};
  const ok = r.review_status === 'approved';
  if (!r.review_status) return null;
  return el('div', { class: 'verdict' }, [
    el('span', { class: `badge ${ok ? 'badge-ok' : 'badge-bad'}` }, [
      ok ? checkTinted('var(--good)') : crossTinted('var(--critical)'),
      el('span', { text: ok ? '审核通过' : '审核未通过' }),
    ]),
    el('span', {
      class: 'score',
      text: `重算轮次 ${rounds || 0} ｜ 质量评分 `,
    }, [el('b', { text: String(r.quality_score ?? '—') })]),
  ]);
}

function reviewBreakdown(review) {
  const r = review || {};
  const checks = [].concat(r.checks || []);
  if (!checks.length && !r.llm_note) return null;
  const box = el('div', { class: 'section' }, [el('h3', { text: '审核员校验明细' })]);
  const card = el('div', { class: 'card' });

  if (checks.length) {
    const ul = el('ul', { class: 'checks' });
    checks.forEach((c) => {
      const ok = c.status === 'pass';
      ul.appendChild(el('li', { class: ok ? 'is-pass' : 'is-fail' }, [
        el('span', { class: 'mark' }, [ok ? checkTinted('var(--good)') : crossTinted('var(--critical)')]),
        el('div', {}, [
          el('span', { class: 'item', text: c.item }),
          el('span', { text: `　${c.comment}` }),
        ]),
      ]));
    });
    card.appendChild(ul);
  }
  if (r.llm_note) {
    card.appendChild(el('div', { class: 'note' }, [
      el('span', { class: 'note-label', text: '大模型补充审查意见' }),
      el('span', { text: r.llm_note }),
    ]));
  }
  box.appendChild(card);
  return box;
}

function renderReviewCard(p) {
  const r = p.review || {};
  const ok = r.review_status === 'approved';
  const card = el('div', { class: 'card' });
  card.appendChild(el('div', { class: 'verdict' }, [
    el('span', { class: `badge ${ok ? 'badge-ok' : 'badge-bad'}` }, [
      ok ? checkTinted('var(--good)') : crossTinted('var(--critical)'),
      el('span', { text: ok ? '审核通过' : '审核未通过' }),
    ]),
    el('span', { class: 'score', text: '评分 ' }, [el('b', { text: String(r.quality_score ?? '—') })]),
  ]));

  const checks = [].concat(r.checks || []);
  if (checks.length) {
    const ul = el('ul', { class: 'checks' });
    checks.forEach((c) => {
      const pass = c.status === 'pass';
      ul.appendChild(el('li', { class: pass ? 'is-pass' : 'is-fail' }, [
        el('span', { class: 'mark' }, [pass ? checkTinted('var(--good)') : crossTinted('var(--critical)')]),
        el('div', {}, [
          el('span', { class: 'item', text: c.item }),
          el('span', { text: `　${c.comment}` }),
        ]),
      ]));
    });
    card.appendChild(ul);
  }
  if (r.llm_note) {
    card.appendChild(el('div', { class: 'note' }, [
      el('span', { class: 'note-label', text: '大模型补充审查意见' }),
      el('span', { text: r.llm_note }),
    ]));
  }
  ui.review.replaceChildren(card);
  ui.review.hidden = false;
}

/* ============================ 指标结果表 ============================ */

function indicatorsTable(indicators) {
  const t = el('table', { class: 'data' });
  t.appendChild(el('thead', {}, [el('tr', {}, [
    el('th', { text: '指标' }),
    el('th', { text: '口径' }),
    el('th', { class: 'num', text: '计算值' }),
    el('th', { text: '单位' }),
    el('th', { class: 'wrap', text: '所用公式' }),
    el('th', { class: 'wrap', text: '状态 / 说明' }),
  ])]));

  const tb = el('tbody');
  indicators.forEach((r) => {
    const mode = MODE_LABEL[r.mode] || r.mode || '';
    const unit = r.unit && !String(r.unit).startsWith('按源') ? r.unit : '';
    const row = el('tr', { class: r.available ? '' : 'is-na' });
    row.append(
      el('td', { text: r.label || r.metric || '' }),
      el('td', { text: mode }),
      el('td', { class: 'num', text: r.available ? fmtVal(r.value) : '—' }),
      el('td', { text: unit || '—' }),
      el('td', { class: 'wrap', text: r.formula || '—' }),
      el('td', { class: 'wrap' }, [
        r.available
          ? el('span', { text: xcheckNote(r.cross_check) || '' })
          : el('span', { class: 'na-reason', text: r.reason || '不可用' }),
      ]),
    );
    tb.appendChild(row);
  });
  t.appendChild(tb);
  return t;
}

function xcheckNote(cc) {
  if (!cc) return '';
  if (cc.recomputed !== undefined && cc.recomputed !== null) {
    return cc.consistent
      ? `重算 ${fmtVal(cc.recomputed)}% 与报表一致（偏差 ${fmtVal(cc.delta_pp)}pp）`
      : `★重算 ${fmtVal(cc.recomputed)}% 与报表 ${fmtVal(cc.reported)}% 不一致（偏差 ${fmtVal(cc.delta_pp)}pp），建议复核口径`;
  }
  return cc.note || '';
}

/* ================== 补充面板：横向对比 / 异常检测 ================== */

const SEV_CLASS = { 高: 'sev-high', 中: 'sev-mid', 低: 'sev-low' };

/** 横向对比：企业运营域多项目的排名（后端 compare 面板）。 */
function comparePanel(cmp) {
  if (!cmp) return null;
  const sec = el('div', { class: 'section' }, [el('h3', { text: '横向对比' })]);

  if (!cmp.available) {
    sec.appendChild(el('p', { class: 'panel-note', text: `本次未做横向对比：${cmp.reason || '—'}` }));
    return sec;
  }

  const entries = [].concat(cmp.entries || []);
  const unit = cmp.unit && !String(cmp.unit).startsWith('按源') ? cmp.unit : '';
  sec.appendChild(el('p', {
    class: 'panel-note',
    text: `${cmp.label}（计算日 ${cmp.stat_date || '—'}）：`
      + `最高 ${cmp.max_entity} ${fmtVal(cmp.max)}｜最低 ${cmp.min_entity} ${fmtVal(cmp.min)}`
      + `｜均值 ${fmtVal(cmp.mean)}｜共 ${cmp.n_entries} 个项目`,
  }));

  const t = el('table', { class: 'data' });
  t.appendChild(el('thead', {}, [el('tr', {}, [
    el('th', { class: 'num', text: '名次' }),
    el('th', { text: '项目' }),
    el('th', { class: 'num', text: cmp.label || '数值' }),
  ])]));
  const tb = el('tbody');
  entries.forEach((e, i) => {
    tb.appendChild(el('tr', {}, [
      el('td', { class: 'num', text: String(i + 1) }),
      el('td', { text: e.label || '—' }),
      el('td', { class: 'num', text: `${fmtVal(e.value)}${unit ? ` ${unit}` : ''}` }),
    ]));
  });
  t.appendChild(tb);
  sec.appendChild(t);
  sec.appendChild(buildChartPanel([{
    kind: 'bar',
    title: `${cmp.label} 横向对比`,
    x: entries.map((e) => e.label),
    series: [{ name: cmp.label || '数值', data: entries.map((e) => e.value) }],
    y_unit: unit,
    note: '按计算值降序排列；单位按源表',
  }]));
  return sec;
}

/** 异常检测：各数据域产出的异常列表（后端 anomalies 面板）。 */
function anomalyPanel(an) {
  if (!an) return null;
  const sec = el('div', { class: 'section' }, [el('h3', { text: '异常检测' })]);

  if (!an.available) {
    sec.appendChild(el('p', { class: 'panel-note', text: an.reason || '本次未做异常检测。' }));
    return sec;
  }

  sec.appendChild(el('p', {
    class: 'panel-note',
    text: `${an.scope || '检测对象'}：${an.summary || ''}`,
  }));

  const items = [].concat(an.items || []);
  if (!items.length) {
    sec.appendChild(el('p', { class: 'panel-note', text: '未发现异常。' }));
    return sec;
  }
  if (an.truncated) {
    sec.appendChild(el('p', {
      class: 'panel-note',
      text: `共 ${an.total} 条，下列为抽样示意 ${an.shown} 条。`,
    }));
  }

  const ul = el('ul', { class: 'anomalies' });
  items.forEach((a) => {
    ul.appendChild(el('li', {}, [
      el('span', {
        class: `sev ${SEV_CLASS[a.severity] || 'sev-mid'}`,
        text: a.severity || '—',
      }),
      el('span', { class: 'kind', text: a.kind_label || '' }),
      el('span', { class: 'ts', text: a.ts == null ? '' : String(a.ts) }),
      el('span', { class: 'detail', text: a.detail || '' }),
    ]));
  });
  sec.appendChild(ul);
  return sec;
}

/* ============================ 小工具 ============================ */

/** 渲染 **加粗**，其余按纯文本插入（避免注入）。 */
function inline(text) {
  const frag = document.createDocumentFragment();
  String(text).split('**').forEach((part, i) => {
    if (!part) return;
    frag.appendChild(i % 2 ? el('strong', { text: part }) : document.createTextNode(part));
  });
  return frag;
}

/** 解析后端产出的 Markdown 表格行（| a | b | 与分隔行）。 */
function mdTable(lines) {
  const rows = lines
    .map((l) => String(l).trim())
    .filter((l) => l.startsWith('|'))
    .map((l) => l.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim()))
    .filter((cells) => !cells.every((c) => /^:?-{2,}:?$/.test(c) || c === ''));
  const t = el('table', { class: 'data' });
  if (!rows.length) return t;
  t.appendChild(el('thead', {}, [el('tr', {}, rows[0].map((c, i) => el('th', {
    class: i > 0 ? 'num' : '', text: c,
  })))]));
  const tb = el('tbody');
  rows.slice(1).forEach((cells) => {
    tb.appendChild(el('tr', {}, cells.map((c, i) => el('td', {
      class: i > 0 ? 'num' : '', text: c,
    }))));
  });
  t.appendChild(tb);
  return t;
}

function svgGlyph(path, color) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 12 12');
  s.setAttribute('width', '12');
  s.setAttribute('height', '12');
  s.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', path);
  p.setAttribute('fill', 'none');
  p.setAttribute('stroke', color);
  p.setAttribute('stroke-width', '1.9');
  p.setAttribute('stroke-linecap', 'round');
  p.setAttribute('stroke-linejoin', 'round');
  s.appendChild(p);
  return s;
}

const checkTinted = (c) => svgGlyph('M2.2 6.4 L4.8 8.9 L9.8 3.3', c);
const crossTinted = (c) => svgGlyph('M3 3 L9 9 M9 3 L3 9', c);
const warnIcon = () => svgGlyph('M6 2.4 L10.6 10 L1.4 10 Z M6 5.2 L6 7.3 M6 8.5 L6 8.6', 'var(--warning)');

function downloadMd(p) {
  const name = (p.report && p.report.title ? p.report.title : 'report').replace(/[\\/:*?"<>|]/g, '_');
  const blob = new Blob([p.answer || ''], { type: 'text/markdown;charset=utf-8' });
  const a = el('a', { href: URL.createObjectURL(blob), download: `${name}.md` });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function showError(msg) {
  const box = el('div', { class: 'section' }, [el('div', { class: 'error-box', text: msg })]);
  if (ui.report.hidden) {
    ui.empty.hidden = true;
    ui.report.replaceChildren(box);
    ui.report.hidden = false;
  } else {
    ui.report.appendChild(box);
  }
  endProcess('本次分析未完成');       // 别让过程面板停在「进行中」
  ui.report.scrollIntoView({ block: 'start', behavior: 'auto' });
}

init();
