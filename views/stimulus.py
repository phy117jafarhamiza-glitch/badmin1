"""مولّد المثيرات: صفحة JavaScript تعرض مثيرات عشوائية وتسجّل لحظة ظهور كل منها.

الاستخدام: توضع الشاشة أمام اللاعب (أو تُعرض بالـ Datashow)، وتُصوَّر الجلسة بالكاميرا.
تُنسخ «إزاحات» الأزمنة الناتجة إلى صفحة التحليل مع زمن أول مثير في الفيديو.
"""
from __future__ import annotations

import json

import streamlit.components.v1 as components

from core.exercises import Exercise, apply_difficulty

HTML = r"""
<div id="wrap" style="font-family:Tahoma,Arial,sans-serif;direction:rtl;text-align:center">
  <div id="stage" style="height:__H__px;background:#111;color:#fff;display:flex;flex-direction:column;
       align-items:center;justify-content:center;font-size:110px;border-radius:14px;transition:none;
       user-select:none">جاهز</div>
  <div style="margin:10px 0">
    <button id="start" style="font-size:18px;padding:8px 22px">▶ ابدأ</button>
    <button id="stop" style="font-size:18px;padding:8px 22px">■ إيقاف</button>
    <button id="fs" style="font-size:18px;padding:8px 22px">⛶ ملء الشاشة</button>
    <span id="info" style="margin-right:12px;color:#555"></span>
  </div>
  <div style="text-align:right;margin-bottom:4px;color:#444">إزاحات المثيرات بالثواني (انسخها إلى صفحة التحليل):</div>
  <textarea id="out" rows="2" readonly style="width:100%;direction:ltr;font-size:15px"></textarea>
  <div id="legend" style="text-align:right;color:#444;margin-top:6px;font-size:14px"></div>
</div>
<script>
const CFG = __CFG__;
const stage = document.getElementById('stage');
const info = document.getElementById('info');
const out = document.getElementById('out');
document.getElementById('legend').textContent = CFG.legend || '';
let running = false, t0 = null, offsets = [], labels = [];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const rnd = (a, b) => a + Math.random() * (b - a);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const COLORS = [["أحمر","#e53935"],["أخضر","#2e7d32"],["أزرق","#1e66d0"],["أصفر","#f9a825"]];
const ARROWS = [["←","يسار"],["→","يمين"],["↑","أمام"],["↓","خلف"]];
let audioCtx = null;
function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = 880; g.gain.value = 0.4; o.connect(g); g.connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + 0.18);
  } catch (e) {}
}
function paint(html, bg) {
  return new Promise(res => requestAnimationFrame(() => {
    stage.style.background = bg || '#111';
    stage.innerHTML = html;
    requestAnimationFrame(() => res(performance.now()));
  }));
}
function mark(t, label) {
  if (t0 === null) t0 = t;
  offsets.push(((t - t0) / 1000).toFixed(3)); labels.push(label);
  out.value = offsets.join(', ');
  info.textContent = 'المحاولة ' + offsets.length + ' من ' + CFG.n_trials;
}
async function showStimulus() {
  const m = CFG.mode;
  if (m === 'single_color') {
    const t = await paint('', '#2e7d32'); mark(t, 'لون'); return 1100;
  }
  if (m === 'four_colors') {
    const [n, c] = pick(COLORS); const t = await paint('<span style="font-size:60px">' + n + '</span>', c);
    mark(t, n); return 1100;
  }
  if (m === 'arrows') {
    const [a, n] = pick(ARROWS); const t = await paint(a, '#1e66d0'); mark(t, n); return 1100;
  }
  if (m === 'numbers') {
    const k = 1 + Math.floor(Math.random() * 6); const t = await paint(String(k), '#6a1b9a'); mark(t, 'منطقة ' + k); return 1100;
  }
  if (m === 'sound') {
    beep(); const t = await paint('🔊', '#f5f5f5'); mark(t, 'صوت'); return 400;
  }
  if (m === 'coach_cue') {
    const t = await paint('<span style="font-size:60px">تحرّك الآن</span>', '#2e7d32'); mark(t, 'مدرب'); return 900;
  }
  if (m === 'fake_arrow') {
    const first = pick(ARROWS);
    if (Math.random() < CFG.fake_prob) {
      await paint(first[0], '#1e66d0'); await sleep(250);
      let second = pick(ARROWS); while (second[0] === first[0]) second = pick(ARROWS);
      const t = await paint(second[0], '#c62828'); mark(t, second[1] + ' (خادع)');
    } else {
      const t = await paint(first[0], '#1e66d0'); mark(t, first[1]);
    }
    return 1100;
  }
  if (m === 'delayed_arrow') {
    await paint('<span style="font-size:60px">تهيّأ</span>', '#f9a825');
    await sleep(rnd(800, 1500));
    const [a, n] = pick(ARROWS); const t = await paint(a, '#1e66d0'); mark(t, n); return 1100;
  }
  if (m === 'skill_cue') {
    const skills = {net_straight: 'شبكة مستقيمة', net_cross: 'شبكة قطرية', net_lift: 'رفع'};
    const key = CFG.skill === 'mixed' ? pick(['net_straight', 'net_cross', 'net_lift']) : CFG.skill;
    const z = 1 + Math.floor(Math.random() * 4);
    const t = await paint('<span style="font-size:56px">' + skills[key] + '</span><span>منطقة ' + z + '</span>', '#1e66d0');
    mark(t, skills[key] + ' ' + z); return 1400;
  }
  return 1000;
}
async function run() {
  if (running) return; running = true; offsets = []; labels = []; t0 = null; out.value = '';
  for (let i = 3; i > 0 && running; i--) { await paint(String(i), '#111'); await sleep(700); }
  for (let k = 0; k < CFG.n_trials && running; k++) {
    await paint('<span style="font-size:48px;color:#ccc">استعد</span>', '#111');
    await sleep(rnd(CFG.delay_min, CFG.delay_max) * 1000);
    if (!running) break;
    const hold = await showStimulus();
    await sleep(hold);
    await paint('', '#111');
    await sleep(CFG.rest_rep * 1000 * 0.15 + 400);
  }
  await paint('انتهت المجموعة', '#111'); running = false;
  info.textContent += ' — انتهى';
}
document.getElementById('start').onclick = run;
document.getElementById('stop').onclick = () => { running = false; };
document.getElementById('fs').onclick = () => {
  const el = document.getElementById('stage');
  (el.requestFullscreen || el.webkitRequestFullscreen || function(){}).call(el);
};
</script>
"""

LEGEND = {
    "single_color": "المثير: لون أخضر واحد.",
    "four_colors": "أحمر = أمام يمين، أخضر = أمام يسار، أزرق = خلف يمين، أصفر = خلف يسار.",
    "arrows": "الأسهم: ← يسار، → يمين، ↑ أمام، ↓ خلف.",
    "numbers": "الرقم يمثل منطقة مرقمة من الملعب (1 إلى 6).",
    "sound": "إشارة صوتية مع وميض شاشة فاتح.",
    "coach_cue": "الشاشة تحدد لحظة بدء حركة المدرب؛ توقّع الاتجاه قبل خروج الريشة.",
    "fake_arrow": "قد يتغير السهم بعد ربع ثانية؛ الاستجابة للسهم الأخير (الأحمر).",
    "delayed_arrow": "تهيّأ (أصفر) ثم سهم بعد تأخير متغير.",
    "skill_cue": "مهارة الشبكة والمنطقة المستهدفة؛ المدرب يغذّي ريشة قصيرة بعد الإشارة.",
}


def render_generator(ex: Exercise, difficulty: int = 3, n_trials: int | None = None, height: int = 520) -> None:
    p = apply_difficulty(ex, difficulty)
    cfg = dict(p["stim"])
    cfg["n_trials"] = int(n_trials or p["reps"])
    cfg["rest_rep"] = p["rest_rep"]
    cfg["legend"] = LEGEND.get(cfg["mode"], "")
    html = HTML.replace("__CFG__", json.dumps(cfg, ensure_ascii=False)).replace("__H__", "320")
    components.html(html, height=height, scrolling=False)
