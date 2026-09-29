"""Standalone HTML playback of archived action chunks; never controls a robot."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import numpy as np

from .labs_kinematics import URDFFK, pose_vector


def write_viewer(bundle, path):
    metadata = next(e for e in bundle['events'] if e['event'] == 'configuration')
    chunks = []
    with tempfile.TemporaryDirectory() as directory:
        solvers = {}
        for side, text in metadata['urdf'].items():
            p = Path(directory) / f'{side}.urdf'
            p.write_text(text)
            solvers[side] = URDFFK(p, side)

        def poses(q):
            if q is None or np.asarray(q).shape != (14,) or not np.isfinite(q).all():
                return None
            return np.concatenate([pose_vector(solvers[s](q[i:i+7])) for s, i in [('left', 0), ('right', 7)]]).tolist()

        for chunk in bundle['chunks']:
            samples = []
            for sample in chunk['tracking_samples']:
                status = sample['status']
                if status.get('phase') not in ('executing', 'holding'):
                    continue
                q, goal = sample['measured_joints'], status.get('holding_target')
                if poses(q) is None or poses(goal) is None:
                    continue
                samples.append({'time': sample['monotonic_ns']/1e9, 'q': q, 'goal': goal,
                                'actual_pose': poses(q), 'goal_pose': poses(goal)})
            if samples:
                start = samples[0]['time']
                for s in samples:
                    s['time'] -= start
            command = chunk.get('command', {})
            targets = command.get('targets', [])
            from .labs_joint_inference import COMMAND_SCHEMA as JOINT_SCHEMA, split_targets

            if targets and command.get('schema') == JOINT_SCHEMA:
                joints, _ = split_targets(targets)
                targets = [poses(q) for q in joints]
            chunks.append({'index': chunk['index'], 'execution': chunk['execution'],
                           'summary': chunk['summary'], 'samples': samples,
                           'actions': chunk['raw_response'].get('actions', []),
                           'targets': targets,
                           'rate': command.get('execution_rate_hz', metadata['execution_rate_hz'])})
    # JSON contains no executable text; base64 also prevents closing script tags.
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value
    payload = base64.b64encode(json.dumps(clean(chunks), allow_nan=False).encode()).decode()
    Path(path).write_text(HTML.replace('__PAYLOAD__', payload))


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Labs action 回放与跟踪对比</title>
<style>body{font:16px system-ui;background:#101821;color:#e0e8f0;margin:28px}select,button,input{margin:8px;padding:6px}canvas{background:#18232f;width:100%;height:260px;margin:8px 0}p{max-width:1000px;line-height:1.6}pre{white-space:pre-wrap}#cursor{width:50%}</style>
<h1>Action chunk 回放与控制跟踪</h1>
<p>离线查看，不发送机器人指令。蓝色：模型绝对目标；橙色：relay 已发布关节目标；绿色：实测反馈。
执行曲线来自约 20 Hz 异步采样，瞬时差值仅供诊断；终点误差和完成状态请一起看。缺失反馈不代表零误差。</p>
<label>Chunk <select id="chunk"></select></label><label>机械臂 <select id="arm"><option value="0">左臂</option><option value="1">右臂</option></select></label>
<label>坐标 <select id="axis"><option value="0">X</option><option value="1">Y</option><option value="2">Z</option></select></label>
<button id="play">播放 / 暂停</button><input id="cursor" type="range" min="0" max="1000" value="0"><span id="time"></span>
<pre id="summary"></pre><h3>模型目标（横轴：参考时间 s；纵轴：位置 m）</h3><canvas id="model"></canvas>
<h3>执行目标与实测（横轴：首个执行采样后的时间 s；纵轴：位置 m）</h3><canvas id="tracking"></canvas>
<h3>关节跟踪误差（实测 − relay 目标，rad）</h3><canvas id="error"></canvas>
<pre id="row"></pre><script>
const data=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__PAYLOAD__'),c=>c.charCodeAt(0))));
const $=id=>document.getElementById(id);let playing=false,last=0;
data.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${c.index} · ${c.execution}`;$('chunk').appendChild(o)});
function plot(id,lines,duration,fraction){let el=$(id);el.width=el.clientWidth*devicePixelRatio;el.height=260*devicePixelRatio;let ctx=el.getContext('2d');ctx.scale(devicePixelRatio,devicePixelRatio);let w=el.clientWidth,h=260;
let vals=lines.flatMap(l=>l.points.map(p=>p[1])).filter(Number.isFinite);if(!vals.length){ctx.fillStyle='#ccd';ctx.fillText('没有可用记录',30,40);return}
let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.1,.001);lo-=pad;hi+=pad;let x=t=>65+t/Math.max(duration,.01)*(w-85),y=v=>h-30-(v-lo)/(hi-lo)*(h-55);
ctx.font='12px system-ui';ctx.fillStyle='#abc';for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;ctx.fillText(v.toFixed(4),4,y(v));ctx.strokeStyle='#2a3b4c';ctx.beginPath();ctx.moveTo(60,y(v));ctx.lineTo(w,y(v));ctx.stroke()}
ctx.fillText('0 s',65,h-6);ctx.fillText(duration.toFixed(2)+' s',w-70,h-6);
for(let l of lines){ctx.strokeStyle=l.color;ctx.lineWidth=1.8;ctx.beginPath();let open=false;for(let p of l.points){if(!Number.isFinite(p[1])){open=false;continue}if(!open){ctx.moveTo(x(p[0]),y(p[1]));open=true}else ctx.lineTo(x(p[0]),y(p[1]))}ctx.stroke()}
ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(x(fraction*duration),15);ctx.lineTo(x(fraction*duration),h-25);ctx.stroke()}
function render(){if(!data.length){$('summary').textContent='没有收到完整的服务端 chunk';return}let c=data[+$('chunk').value],a=+$('arm').value,k=+$('axis').value,f=+$('cursor').value/1000,idx=a*9+k;
let md=c.targets.length/c.rate,td=c.samples.length?c.samples.at(-1).time:md,d=Math.max(md,td),t=f*d,row=Math.min(c.actions.length-1,Math.max(0,Math.floor(t*c.rate)));
$('time').textContent=t.toFixed(2)+' s';$('summary').textContent=JSON.stringify({execution:c.execution,...c.summary},null,2);
plot('model',[{color:'#62b1ff',points:c.targets.map((v,i)=>[(i+1)/c.rate,v[idx]])}],d,f);
plot('tracking',[{color:'#f7ad60',points:c.samples.map(v=>[v.time,v.goal_pose[idx]])},{color:'#65dfa3',points:c.samples.map(v=>[v.time,v.actual_pose[idx]])}],d,f);
let colors=['#ff9c9c','#ffcf80','#dcf18d','#78dfc2','#75baff','#b8a2ff','#f3a2d9'];plot('error',colors.map((color,j)=>({color,points:c.samples.map(v=>[v.time,v.q[a*7+j]-v.goal[a*7+j]])})),d,f);
$('row').textContent='原始 action['+row+'] = '+JSON.stringify(c.actions[row])+'\n绝对目标 = '+JSON.stringify(c.targets[row]);}
for(let id of ['chunk','arm','axis','cursor'])$(id).oninput=render;$('play').onclick=()=>{playing=!playing;last=performance.now()};
function tick(now){if(playing&&data.length){let c=data[+$('chunk').value],d=Math.max(c.targets.length/c.rate,c.samples.length?c.samples.at(-1).time:0,.1);$('cursor').value=(+$('cursor').value+(now-last)/d)%1001;render()}last=now;requestAnimationFrame(tick)}render();requestAnimationFrame(tick);window.onresize=render;
</script></html>'''
