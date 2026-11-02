/* 图表渲染 —— 纯手写 SVG，不依赖任何图表库（便于离线演示）。
 *
 * 规格来自后端 report_generator.ChartSpec：
 *     { kind: 'line'|'bar', title, x: [...], series: [{name, data}], y_unit, note }
 *
 * 遵循的图表规范（依据数据可视化规范逐条落实）：
 *   · 单序列不画图例（标题已说明画的是什么）；仅 >=2 序列时才出现图例。
 *   · 条/柱 <=24px 厚，数据端 4px 圆角、基线端为直角；线宽 2px；标记 >=8px 且带 2px 表面描边。
 *   · 网格/轴线为 1px 实线发丝线，弱于数据；不用虚线。
 *   · 不逐点标数值：柱顶/条端标注（<=12 项且放得下时），线只标端点。
 *   · 文字一律用文字色，绝不穿数据色。
 *   · 类目轴为「长中文标签」时用横向条（避免标签旋转）；时间类目用纵向柱。
 *   · 默认带悬停层：横向条/柱逐条 tooltip + 高亮，折线用十字准线吸附最近点，键盘可达。
 *   · 每图配「表格」视图（悬停不是读数的唯一途径）。
 *   · 只有一个类目时不画单条柱（改为数值卡）；类目单位不一致时按单位分面，避免不同量纲同轴比较。
 */

const NS = 'http://www.w3.org/2000/svg';
const FONT_TICK = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
const FONT_CAT = '11.5px system-ui, -apple-system, "Segoe UI", sans-serif';

/* ---------------- DOM 辅助（文本一律 textContent，不拼 innerHTML） ---------------- */

export function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  applyAttrs(n, attrs);
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

function svgEl(tag, attrs = {}, children = []) {
  const n = document.createElementNS(NS, tag);
  applyAttrs(n, attrs);
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

function applyAttrs(n, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === 'text') n.textContent = v;
    // 事件回调必须用 addEventListener：若走 setAttribute，函数体被转成字符串后
    // 会在**全局作用域**求值，模块内声明的变量（如 app.js 的 ui）不可见，点击即抛错。
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
}

let _mctx = null;
function measure(text, font = FONT_CAT) {
  if (!_mctx) _mctx = document.createElement('canvas').getContext('2d');
  _mctx.font = font;
  return _mctx.measureText(String(text)).width;
}

function truncate(text, maxW, font = FONT_CAT) {
  let s = String(text);
  if (measure(s, font) <= maxW) return s;
  while (s.length > 1 && measure(s + '…', font) > maxW) s = s.slice(0, -1);
  return s + '…';
}

/* ---------------- 数值格式 ---------------- */

function trimZeros(s) {
  return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

/** 数值显示与后端 report_generator._fmt 对齐：
 *  >=1e4 用千分位（最多 3 位小数），其余取 4 位有效数字，极小值转科学计数法。
 *  这样表格/图表里的数字与报告"结论"里的写法完全一致，也不会丢失有效位。 */
export function fmtVal(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e4) return v.toLocaleString('zh-CN', { maximumFractionDigits: 3 });
  if (a !== 0 && a < 1e-4) {
    const [m, e] = v.toExponential(3).split('e');
    const n = Number(e);
    return `${m}e${n < 0 ? '-' : '+'}${String(Math.abs(n)).padStart(2, '0')}`;
  }
  return trimZeros(v.toPrecision(4));
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e9) return trimZeros((v / 1e9).toFixed(1)) + 'B';
  if (a >= 1e6) return trimZeros((v / 1e6).toFixed(1)) + 'M';
  if (a >= 1e3) return trimZeros((v / 1e3).toFixed(1)) + 'K';
  if (a < 1) return trimZeros(v.toFixed(3));
  return trimZeros(v.toFixed(a >= 100 ? 0 : 1));
}

function niceTicks(min, max, count = 4) {
  let lo = min, hi = max;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const out = [];
  for (let v = start; v <= end + step * 1e-6; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function goodPoints(series) {
  return [].concat(series || []).map((s) => ({
    name: s && s.name != null ? String(s.name) : '',
    data: [].concat((s && s.data) || []),
  }));
}

/* ---------------- 形状 ---------------- */

/** 数据端 4px 圆角、基线端直角的条。dir: 'right'|'left' 水平，'up' 纵向。 */
function barPath(x, y, w, h, dir, r = 4) {
  const rr = Math.max(0, Math.min(r, w, h / 2));
  if (dir === 'up') {
    return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} `
         + `Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`;
  }
  if (dir === 'right') {
    return `M${x},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h - rr} `
         + `Q${x + w},${y + h} ${x + w - rr},${y + h} H${x} Z`;
  }
  return `M${x + w},${y} H${x + rr} Q${x},${y} ${x},${y + rr} V${y + h - rr} `
       + `Q${x},${y + h} ${x + rr},${y + h} H${x + w} Z`;
}

function decimate(xs, ds, maxPoints) {
  const n = xs.length;
  if (n <= maxPoints) return { xs, ds };
  const stride = Math.ceil(n / maxPoints);
  const ax = [], ad = [];
  for (let i = 0; i < n; i += stride) { ax.push(xs[i]); ad.push(ds[i]); }
  if (ax[ax.length - 1] !== xs[n - 1]) { ax.push(xs[n - 1]); ad.push(ds[n - 1]); }
  return { xs: ax, ds: ad };
}

/* ---------------- tooltip ---------------- */

function makeTip(body) {
  const tip = el('div', { class: 'tip', role: 'status', 'aria-live': 'polite' });
  body.appendChild(tip);
  return {
    node: tip,
    show(x, y, valueText, labelText) {
      tip.replaceChildren(
        el('div', { class: 'tip-value', text: valueText }),
        el('div', { class: 'tip-row' }, [
          el('span', { class: 'tip-key', 'aria-hidden': 'true' }),
          el('span', { text: labelText }),
        ]),
      );
      tip.classList.add('is-on');
      const w = tip.offsetWidth;
      const left = Math.max(4, Math.min(x - w / 2, body.clientWidth - w - 4));
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(2, y - tip.offsetHeight - 12)}px`;
    },
    hide() { tip.classList.remove('is-on'); },
  };
}

/* ---------------- 横向条（类目为长文本） ---------------- */

function buildBarChart(spec, width, opts = {}) {
  const series = goodPoints(spec.series);
  const labels = [].concat(spec.x || []).map((v) => String(v));
  const unit = opts.unit || spec.y_unit || '';

  const vals = series.flatMap((s) => s.data.map((v) => (typeof v === 'number' ? v : null)));
  const finite = vals.filter((v) => v !== null && Number.isFinite(v));
  const dMax = finite.length ? Math.max(0, ...finite) : 1;
  const dMin = finite.length ? Math.min(0, ...finite) : 0;

  const labelW = Math.min(230, Math.max(...labels.map((l) => measure(l)), 40) + 10);
  const valueW = 62;
  const m = { t: 6, r: valueW, b: 22, l: labelW + 10 };
  const plotW = Math.max(60, width - m.l - m.r);
  const rowH = 30;
  const plotH = Math.max(rowH, labels.length * rowH);
  const height = m.t + plotH + m.b;

  const xOf = (v) => m.l + ((v - dMin) / (dMax - dMin || 1)) * plotW;
  const zero = xOf(0);

  const body = el('div', { class: 'chart-body' });
  const node = svgEl('svg', {
    class: 'chart', width, height, role: 'img',
    'aria-label': `${spec.title || '对比图'}，共 ${labels.length} 个类目，详见下方表格视图`,
  });

  // 零基线
  node.appendChild(svgEl('line', {
    class: 'axis-line', x1: zero, y1: m.t, x2: zero, y2: m.t + plotH,
  }));

  const thickness = Math.min(24, rowH - 10);
  const nS = series.length;
  const subH = nS > 1 ? (thickness - 2 * (nS - 1)) / nS : thickness;
  const bars = [];

  labels.forEach((label, i) => {
    const rowY = m.t + i * rowH;

    series.forEach((s, si) => {
      const v = s.data[i];
      const has = typeof v === 'number' && Number.isFinite(v);
      const y = rowY + (rowH - thickness) / 2 + si * (subH + 2);
      if (!has) return;
      const x1 = xOf(v);
      const x = Math.min(x1, zero);
      const w = Math.max(Math.abs(x1 - zero), 1.5);
      const dir = v >= 0 ? 'right' : 'left';
      const bar = svgEl('path', {
        class: 'bar', d: barPath(x, y, w, subH, dir, 4),
        fill: nS > 1 ? `var(--series-${(si % 8) + 1})` : null,
      });
      node.appendChild(bar);
      bars.push({ i, bar });
    });

    // 类目标签
    node.appendChild(svgEl('text', {
      class: 'cat', x: m.l - 8, y: rowY + rowH / 2 + 4, 'text-anchor': 'end',
      text: truncate(label, labelW),
    }));

    // 值标在条端（单序列时；多序列靠 tooltip 与表格，避免拥挤）
    if (nS === 1) {
      const v = series[0].data[i];
      if (typeof v === 'number' && Number.isFinite(v)) {
        const x1 = xOf(v);
        const outside = x1 >= zero;
        node.appendChild(svgEl('text', {
          class: 'val',
          x: outside ? x1 + 7 : x1 - 7,
          y: rowY + rowH / 2 + 4,
          'text-anchor': outside ? 'start' : 'end',
          text: fmtVal(v),
        }));
      }
    }

    // 命中区：整行（比条本身大得多，便于悬停/触屏）
    const hit = svgEl('rect', {
      class: 'hit', x: 0, y: rowY, width, height: rowH,
      tabindex: '0', role: 'img',
      'aria-label': nS === 1
        ? `${label}：${fmtVal(series[0].data[i])}${unit ? ' ' + unit : ''}`
        : `${label}：${series.map((s) => `${s.name} ${fmtVal(s.data[i])}`).join('，')}`,
    });
    node.appendChild(hit);
    hit.dataset.row = String(i);
  });

  body.appendChild(node);
  const tip = makeTip(body);

  const highlight = (i, on) => {
    bars.filter((b) => b.i === i).forEach((b) => b.bar.classList.toggle('is-hot', on));
  };

  const show = (i, target) => {
    highlight(i, true);
    const r = target.getBoundingClientRect();
    const box = body.getBoundingClientRect();
    const v = series[0].data[i];
    tip.show(
      r.left - box.left + r.width / 2,
      r.top - box.top,
      `${fmtVal(v)}${unit ? ' ' + unit : ''}`,
      nS === 1 ? labels[i] : `${labels[i]} · ${series[0].name}`,
    );
  };

  body.addEventListener('pointerover', (e) => {
    const hit = e.target.closest ? e.target.closest('.hit') : null;
    if (hit) show(Number(hit.dataset.row), hit);
  });
  body.addEventListener('pointerleave', () => { tip.hide(); bars.forEach((b) => b.bar.classList.remove('is-hot')); });
  body.addEventListener('focusin', (e) => {
    if (e.target.classList && e.target.classList.contains('hit')) {
      show(Number(e.target.dataset.row), e.target);
    }
  });
  body.addEventListener('focusout', () => { tip.hide(); bars.forEach((b) => b.bar.classList.remove('is-hot')); });

  return body;
}

/* ---------------- 纵向柱（时间类目） ---------------- */

function buildColumnChart(spec, width, opts = {}) {
  const series = goodPoints(spec.series);
  const labels = [].concat(spec.x || []).map((v) => String(v));
  const unit = opts.unit || spec.y_unit || '';
  const vals = (series[0] ? series[0].data : []).map((v) => (typeof v === 'number' ? v : null));
  const finite = vals.filter((v) => v !== null && Number.isFinite(v));
  const ticks = niceTicks(Math.min(0, ...finite), Math.max(0, ...finite), 4);
  const dMin = Math.min(...ticks);
  const dMax = Math.max(...ticks);

  const m = { t: 20, r: 10, b: 30, l: 0 };
  const sample = ticks.map((t) => fmtTick(t));
  m.l = Math.max(...sample.map((s) => measure(s, FONT_TICK))) + 12;
  const plotW = Math.max(60, width - m.l - m.r);
  const plotH = 170;
  const height = m.t + plotH + m.b;

  const band = plotW / Math.max(1, labels.length);
  const yOf = (v) => m.t + plotH - ((v - dMin) / (dMax - dMin || 1)) * plotH;
  const zero = yOf(0);

  const body = el('div', { class: 'chart-body' });
  const node = svgEl('svg', {
    class: 'chart', width, height, role: 'img',
    'aria-label': `${spec.title || '趋势图'}，共 ${labels.length} 个类目，详见下方表格视图`,
  });

  for (const t of ticks) {
    const y = yOf(t);
    node.appendChild(svgEl('line', { class: 'grid-line', x1: m.l, y1: y, x2: m.l + plotW, y2: y }));
    node.appendChild(svgEl('text', { class: 'tick', x: m.l - 8, y: y + 4, 'text-anchor': 'end', text: fmtTick(t) }));
  }
  node.appendChild(svgEl('line', { class: 'axis-line', x1: m.l, y1: zero, x2: m.l + plotW, y2: zero }));

  const thickness = Math.min(24, band * 0.62);
  const showCap = labels.length <= 12;
  const bars = [];

  labels.forEach((label, i) => {
    const v = vals[i];
    const cx = m.l + band * (i + 0.5);
    const x = cx - thickness / 2;
    if (typeof v === 'number' && Number.isFinite(v)) {
      const y = Math.min(yOf(v), zero);
      const h = Math.max(Math.abs(zero - yOf(v)), 1.5);
      const bar = svgEl('path', { class: 'bar', d: barPath(x, y, thickness, h, 'up', 4) });
      node.appendChild(bar);
      bars.push({ i, bar });

      if (showCap && measure(fmtVal(v)) <= band - 4) {
        node.appendChild(svgEl('text', {
          class: 'val', x: cx, y: y - 5, 'text-anchor': 'middle', text: fmtVal(v),
        }));
      }
    }

    // x 轴标签：放不下就旋转
    const needRotate = measure(label, FONT_TICK) > band - 2;
    node.appendChild(svgEl('text', {
      class: 'tick', x: cx, y: m.t + plotH + 15,
      'text-anchor': needRotate ? 'end' : 'middle',
      transform: needRotate ? `rotate(-35 ${cx} ${m.t + plotH + 15})` : null,
      text: truncate(label, needRotate ? 78 : band),
    }));

    const hit = svgEl('rect', {
      class: 'hit', x: m.l + band * i, y: m.t, width: band, height: plotH,
      tabindex: '0', role: 'img',
      'aria-label': `${label}：${fmtVal(v)}${unit ? ' ' + unit : ''}`,
    });
    node.appendChild(hit);
    hit.dataset.row = String(i);
  });

  body.appendChild(node);
  const tip = makeTip(body);

  const show = (i, target) => {
    bars.filter((b) => b.i === i).forEach((b) => b.bar.classList.add('is-hot'));
    const r = target.getBoundingClientRect();
    const box = body.getBoundingClientRect();
    tip.show(r.left - box.left + r.width / 2, r.top - box.top,
      `${fmtVal(series[0].data[i])}${unit ? ' ' + unit : ''}`, labels[i]);
  };

  body.addEventListener('pointerover', (e) => {
    const hit = e.target.closest ? e.target.closest('.hit') : null;
    if (hit) show(Number(hit.dataset.row), hit);
  });
  body.addEventListener('pointerleave', () => { tip.hide(); bars.forEach((b) => b.bar.classList.remove('is-hot')); });
  body.addEventListener('focusin', (e) => {
    if (e.target.classList && e.target.classList.contains('hit')) show(Number(e.target.dataset.row), e.target);
  });
  body.addEventListener('focusout', () => { tip.hide(); bars.forEach((b) => b.bar.classList.remove('is-hot')); });

  return body;
}

/* ---------------- 折线（十字准线） ---------------- */

function buildLineChart(spec, width) {
  const series = goodPoints(spec.series);
  const rawX = [].concat(spec.x || []).map((v) => String(v));
  const unit = spec.y_unit || '';
  const height = 210;
  const m = { t: 16, r: 58, b: 30, l: 0 };

  const all = series.flatMap((s) => s.data.map((v) => (typeof v === 'number' ? v : null)));
  const finite = all.filter((v) => v !== null && Number.isFinite(v));
  const ticks = niceTicks(Math.min(...finite), Math.max(...finite), 4);
  m.l = Math.max(...ticks.map((t) => measure(fmtTick(t), FONT_TICK))) + 12;

  const plotW = Math.max(60, width - m.l - m.r);
  const plotH = height - m.t - m.b;
  const dMin = Math.min(...ticks);
  const dMax = Math.max(...ticks);

  const n = rawX.length;
  const step = n > 1 ? plotW / (n - 1) : 0;
  const xOf = (i) => m.l + i * step;
  const yOf = (v) => m.t + plotH - ((v - dMin) / (dMax - dMin || 1)) * plotH;

  const body = el('div', { class: 'chart-body' });
  const node = svgEl('svg', {
    class: 'chart', width, height, role: 'img',
    'aria-label': `${spec.title || '趋势图'}，共 ${n} 个数据点，详见下方表格视图`,
  });

  for (const t of ticks) {
    const y = yOf(t);
    node.appendChild(svgEl('line', { class: 'grid-line', x1: m.l, y1: y, x2: m.l + plotW, y2: y }));
    node.appendChild(svgEl('text', { class: 'tick', x: m.l - 8, y: y + 4, 'text-anchor': 'end', text: fmtTick(t) }));
  }
  node.appendChild(svgEl('line', { class: 'axis-line', x1: m.l, y1: yOf(dMin), x2: m.l + plotW, y2: yOf(dMin) }));

  // 折线（断点分段）+ 面积淡染（单序列时）
  series.forEach((s, si) => {
    const segs = [];
    let cur = [];
    s.data.forEach((v, i) => {
      if (typeof v === 'number' && Number.isFinite(v)) cur.push([xOf(i), yOf(v)]);
      else if (cur.length) { segs.push(cur); cur = []; }
    });
    if (cur.length) segs.push(cur);

    // 面积淡染只在 y 轴确实从 0 起时使用：轴不落在 0 上时填充会暗示一个并不存在的量
    if (segs.length === 1 && segs[0].length > 1 && dMin === 0) {
      const pts = segs[0];
      const base = yOf(dMin);
      node.appendChild(svgEl('path', {
        class: 'area',
        d: `M${pts[0][0]},${base} ` + pts.map((p) => `L${p[0]},${p[1]}`).join(' ')
           + ` L${pts[pts.length - 1][0]},${base} Z`,
        fill: si === 0 ? null : `var(--series-${(si % 8) + 1}-soft)`,
      }));
    }
    for (const seg of segs) {
      if (seg.length < 2) continue;
      node.appendChild(svgEl('path', {
        class: 'line', d: 'M' + seg.map((p) => `${p[0]},${p[1]}`).join(' L'),
        stroke: si === 0 ? null : `var(--series-${(si % 8) + 1})`,
      }));
    }
  });

  // 末端直接标注（只标端点，不逐点标）
  const s0 = series[0];
  if (s0) {
    for (let i = s0.data.length - 1; i >= 0; i -= 1) {
      const v = s0.data[i];
      if (typeof v === 'number' && Number.isFinite(v)) {
        node.appendChild(svgEl('circle', { class: 'marker', cx: xOf(i), cy: yOf(v), r: 4 }));
        node.appendChild(svgEl('text', {
          class: 'val', x: xOf(i) + 9, y: yOf(v) + 4, text: fmtVal(v),
        }));
        break;
      }
    }
  }

  // x 轴刻度：最多 6 个
  const every = Math.max(1, Math.ceil(n / 6));
  rawX.forEach((label, i) => {
    if (i % every !== 0 && i !== n - 1) return;
    node.appendChild(svgEl('text', {
      class: 'tick', x: xOf(i), y: m.t + plotH + 16, 'text-anchor': 'middle',
      text: truncate(label, Math.max(56, plotW / 6)),
    }));
  });

  // 交互层：十字准线吸附最近点
  const cross = svgEl('line', {
    class: 'crosshair', x1: 0, y1: m.t, x2: 0, y2: m.t + plotH, opacity: 0,
  });
  const dot = svgEl('circle', { class: 'marker', r: 4, opacity: 0 });
  const overlay = svgEl('rect', {
    class: 'hit', x: m.l, y: m.t, width: plotW, height: plotH,
    tabindex: '0', role: 'img',
    'aria-label': `${spec.title || '趋势图'}，可用左右方向键逐点读数`,
  });
  node.append(overlay, cross, dot);
  body.appendChild(node);
  const tip = makeTip(body);

  let active = -1;
  const focusIndex = (i) => {
    if (n === 0) return;
    active = Math.max(0, Math.min(n - 1, i));
    const px = xOf(active);
    cross.setAttribute('x1', px);
    cross.setAttribute('x2', px);
    cross.setAttribute('opacity', '1');
    const v = s0 ? s0.data[active] : null;
    if (typeof v === 'number' && Number.isFinite(v)) {
      dot.setAttribute('cx', px);
      dot.setAttribute('cy', yOf(v));
      dot.setAttribute('opacity', '1');
    } else {
      dot.setAttribute('opacity', '0');
    }
    const box = body.getBoundingClientRect();
    const r = node.getBoundingClientRect();
    tip.show(
      r.left - box.left + px,
      r.top - box.top + m.t + 6,
      `${fmtVal(v)}${unit ? ' ' + unit : ''}`,
      rawX[active],
    );
  };
  const clear = () => {
    active = -1;
    cross.setAttribute('opacity', '0');
    dot.setAttribute('opacity', '0');
    tip.hide();
  };

  node.addEventListener('pointermove', (e) => {
    const r = node.getBoundingClientRect();
    const rel = e.clientX - r.left - m.l;
    focusIndex(step ? Math.round(rel / step) : 0);
  });
  node.addEventListener('pointerleave', clear);
  overlay.addEventListener('focus', () => focusIndex(active < 0 ? n - 1 : active));
  overlay.addEventListener('blur', clear);
  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { focusIndex(active + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { focusIndex(active - 1); e.preventDefault(); }
    if (e.key === 'Home') { focusIndex(0); e.preventDefault(); }
    if (e.key === 'End') { focusIndex(n - 1); e.preventDefault(); }
  });

  return body;
}

/* ---------------- 表格视图 / 图例 / 数值卡 ---------------- */

function chartTable(spec, opts = {}) {
  const series = goodPoints(spec.series);
  const unit = opts.unit || spec.y_unit || '';
  const t = el('table', { class: 'data' });
  const head = el('tr', {}, [el('th', { text: opts.categoryLabel || '类别' })]);
  series.forEach((s) => head.appendChild(el('th', { class: 'num', text: s.name || '数值' })));
  t.appendChild(el('thead', {}, [head]));

  const tb = el('tbody');
  [].concat(spec.x || []).forEach((label, i) => {
    const row = el('tr', {}, [el('td', { text: String(label) })]);
    series.forEach((s) => {
      const v = s.data[i];
      row.appendChild(el('td', {
        class: 'num',
        text: typeof v === 'number' && Number.isFinite(v) ? fmtVal(v) : '—',
      }));
    });
    tb.appendChild(row);
  });
  t.appendChild(tb);
  return t;
}

function legend(spec) {
  const series = goodPoints(spec.series);
  if (series.length < 2) return null;  // 单序列不画图例：标题已说明
  const wrap = el('div', { class: 'tip-row' });
  series.forEach((s, i) => {
    wrap.appendChild(el('span', {
      class: 'tip-key',
      style: `background: var(--series-${(i % 8) + 1})`,
      'aria-hidden': 'true',
    }));
    wrap.appendChild(el('span', { text: s.name }));
  });
  return wrap;
}

function statTile(label, value, unit) {
  return el('div', { class: 'stat-tile' }, [
    el('div', { class: 'stat-label', text: label }),
    el('div', { class: 'stat-value' }, [
      document.createTextNode(fmtVal(value)),
      unit ? el('span', { class: 'stat-unit', text: unit }) : null,
    ]),
  ]);
}

function chartCard(spec, body, opts = {}) {
  const card = el('div', { class: 'chart-card' });
  const head = el('div', { class: 'chart-head' }, [
    el('span', { class: 'chart-title', text: opts.title || spec.title || '图表' }),
  ]);

  const table = chartTable(spec, opts);
  let showingTable = false;
  const toggle = el('button', {
    class: 'btn btn-sm', type: 'button', 'aria-pressed': 'false', text: '表格',
  });

  const legs = legend(spec);
  const holder = el('div', {}, [body]);
  if (legs) holder.insertBefore(legs, body);

  const tableHolder = el('div', {}, [table]);
  tableHolder.hidden = true;

  toggle.addEventListener('click', () => {
    showingTable = !showingTable;
    holder.hidden = showingTable;
    tableHolder.hidden = !showingTable;
    toggle.textContent = showingTable ? '图形' : '表格';
    toggle.setAttribute('aria-pressed', String(showingTable));
  });

  head.appendChild(toggle);
  card.append(head, holder, tableHolder);

  if (opts.note || spec.note) {
    card.appendChild(el('p', { class: 'chart-note', text: opts.note || spec.note }));
  }
  return card;
}

/* ---------------- 对外入口 ---------------- */

/**
 * 把后端 charts 规格渲染成图表面板。
 *   charts      report.charts
 *   unitOfLabel (label) => 单位 | undefined，用于识别「类目单位不一致」并分面
 */
export function buildChartPanel(charts, { unitOfLabel } = {}) {
  const wrap = el('div', { class: 'charts' });
  const list = [].concat(charts || []);

  for (const spec of list) {
    const labels = [].concat(spec.x || []).map((v) => String(v));
    const series = goodPoints(spec.series);
    const s0 = series[0];
    if (!s0 || labels.length === 0) continue;

    // 单位识别：类目单位不一致时按单位分面（不同量纲不能同轴比较）
    const units = labels.map((l) => (unitOfLabel ? unitOfLabel(l) : null) || '');
    const distinct = [...new Set(units.filter(Boolean))];
    const faceted = spec.kind === 'bar' && distinct.length > 1;

    if (faceted) {
      for (const u of distinct) {
        const idxs = labels.map((_, i) => i).filter((i) => units[i] === u);
        const sub = {
          ...spec,
          x: idxs.map((i) => labels[i]),
          series: [{ name: s0.name, data: idxs.map((i) => s0.data[i]) }],
        };
        wrap.appendChild(chartCard(sub, buildFacetBody(sub, u), {
          title: `${spec.title || '对比'} · ${u}`, unit: u,
        }));
      }
      continue;
    }

    wrap.appendChild(chartCard(spec, buildBody(spec, series), {}));
  }
  return wrap;
}

function buildBody(spec, series) {
  const labels = [].concat(spec.x || []).map((v) => String(v));

  if (spec.kind === 'line') {
    if (labels.length < 2) return statRow(spec, series);
    return measureInto((w) => buildLineChart(spec, w));
  }

  // bar
  if (labels.length === 1) return statRow(spec, series);
  const temporal = isTemporal(labels);
  return measureInto((w) => (temporal
    ? buildColumnChart(spec, w)
    : buildBarChart(spec, w)));
}

function buildFacetBody(spec, unit) {
  const labels = [].concat(spec.x || []).map((v) => String(v));
  if (labels.length === 1) return statRow(spec, goodPoints(spec.series), unit);
  return measureInto((w) => buildBarChart(spec, w, { unit }));
}

function statRow(spec, series, unit) {
  const labels = [].concat(spec.x || []).map((v) => String(v));
  const s0 = series[0];
  const u = unit || spec.y_unit || '';
  if (labels.length === 1) {
    // 单类目：数值本身就是结论，不画单条柱
    return el('div', { class: 'stat-row' }, [statTile(labels[0], s0.data[0], u)]);
  }
  // 折线数据点不足：退化为并排数值卡
  return el('div', { class: 'stat-row' },
    labels.map((l, i) => statTile(l, s0.data[i], u)).slice(0, 8));
}

/** 容器宽度要在布局后才知道，故先占位、再测量绘制，并在宽度变化时重绘。 */
function measureInto(draw) {
  const box = el('div', { class: 'chart-mount' });
  let lastW = 0;
  const render = () => {
    const w = Math.round(box.clientWidth);
    if (w < 40 || w === lastW) return;
    lastW = w;
    box.replaceChildren(draw(w));
  };
  requestAnimationFrame(() => {
    render();
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(render).observe(box);
    } else {
      window.addEventListener('resize', render);
    }
  });
  return box;
}

function isTemporal(labels) {
  return labels.length > 0 && labels.every((l) => /^\d{4}[-/年]?(\d{1,2})?/.test(l.trim()));
}
