/* test.js — microforge 无头自检（Node）
 * 运行: node test.js
 * 退出码 0 = 全绿, 1 = 有失败
 */
const MF = require('./microforge.js');

console.log('╔══════════════════════════════════════════════════════╗');
console.log('║   microforge.js — 硬不变量交叉验证   v' + MF.version + '            ║');
console.log('╚══════════════════════════════════════════════════════╝');

const t0 = Date.now();
const rep = MF.selfTest();
const dt = Date.now() - t0;

let allPass = true;
rep.results.forEach((r, i) => {
  const mark = r.pass ? '✓' : '✗';
  if (!r.pass) allPass = false;
  console.log(`${mark} [${String(i + 1).padStart(2)}] ${r.name}` + (r.info ? `  (${r.info})` : ''));
});

console.log('────────────────────────────────────────────────────────');
console.log(`结果: ${rep.passed}/${rep.total} 通过  ·  耗时 ${dt}ms  ·  作者 晨星`);
console.log('────────────────────────────────────────────────────────');

if (!allPass) {
  console.error('✗ 存在失败项，请检查数值实现。');
  process.exit(1);
} else {
  console.log('✓ 全部不变量通过 — 世界顶级 AI 微指令基石库自检通过。');
  process.exit(0);
}
