"""
command_center.py - receives OBSERVATIONS from the drone and PREDICTS intruder / not intruder.

The drone only reports BEHAVIOUR data about a tracked person: speed, loitering, proximity to the protected
asset and deviation from the authorised route. (No temperature: everybody has the same body heat, and a thermal
camera far away cannot measure it reliably.) THIS server runs the trained model, answers THREAT / NOT_THREAT,
explains why, raises alerts, merges repeat sightings into incidents and sends the verdict back to the drone.

Needs only the Python standard library + threat_engine.py + datasets/threat_model.json
(create the model once with:  python patrol_core.py --train)

Run:   python command_center.py        Open:  http://127.0.0.1:8000

API
  POST /api/observations   drone -> CC  observation   (response = verdict)
  POST /api/world-state    drone -> CC  simulation counters (optional)
  POST /api/session/reset | /api/session/stop
  GET  /api/state          dashboard data       GET /api/model   model details
"""
import json
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from threat_engine import ThreatEngine

HOST, PORT = "127.0.0.1", 8000
MODEL_PATH = Path(__file__).resolve().parent / "datasets" / "threat_model.json"
LOCK = threading.Lock()


def load_engine():
    if not MODEL_PATH.exists():
        print(f"[CC] model file not found: {MODEL_PATH}\n[CC] trying to train it with patrol_core.py ...")
        try:
            import patrol_core
            patrol_core.train_model(patrol_core.Config())
        except Exception as e:
            print(f"[CC] could not train automatically ({e}).\n     Run once:  python patrol_core.py --train")
            sys.exit(1)
    return ThreatEngine.from_file(MODEL_PATH, alert_thr=0.5, suspicious_thr=0.3, w_new=0.5,
                                  incident_radius=40.0, incident_window=150.0, clear_age=12.0)


ENGINE = load_engine()
STATE = {}


def reset_state():
    ENGINE.reset()
    STATE.clear()
    STATE.update({"session_started": datetime.now().isoformat(timespec="seconds"), "active": True, "n_obs": 0,
                  "tracks": {}, "events": [], "world": {}, "drone": None,
                  "eval": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}})


reset_state()


def process_observation(obs):
    verdict = ENGINE.assess(obs)
    tid, truth = obs["track_id"], obs.get("sim_ground_truth", "?")
    STATE["n_obs"] += 1
    STATE["drone"] = {**obs["drone"], "sim_time": obs.get("sim_time")}
    tr = STATE["tracks"].setdefault(tid, {"first_wall": time.time(), "hist": [], "decision": None})
    tr.update(obs=obs, verdict=verdict, wall=time.time(), truth=truth)
    if verdict["p_threat"] is not None:
        tr["hist"] = (tr["hist"] + [verdict["p_threat"]])[-40:]
    a = verdict["alert"]
    if a in ("NEW", "MERGED", "CLEARED"):
        is_threat = a != "CLEARED"
        if tr["decision"] is None:                         # first decision on this track -> evaluation counters
            tr["decision"] = "THREAT" if is_threat else "NOT_THREAT"
            key = ("TP" if truth == "HOSTILE" else "FP") if is_threat else ("FN" if truth == "HOSTILE" else "TN")
            STATE["eval"][key] += 1
        STATE["events"].append({"time": datetime.now().strftime("%H:%M:%S"), "sim_time": obs.get("sim_time"),
                                "type": {"NEW": "ALERT", "MERGED": "MERGED", "CLEARED": "CLEARED"}[a],
                                "track_id": tid, "incident_id": verdict["incident_id"], "p_threat": verdict["p_threat"],
                                "position": obs["position"], "truth": truth})
        del STATE["events"][:-200]
        if a == "NEW":
            print(f"[CC] ALERT {verdict['incident_id']} track {tid} p={verdict['p_threat']} at "
                  f"({obs['position']['x']}, {obs['position']['y']}) m   [truth={truth}]")
    return verdict


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Night Patrol - Command Center</title>
<style>
 body{margin:0;background:#050a08;color:#aaffcd;font-family:Consolas,monospace;font-size:12px}
 header{padding:10px 16px;background:#09110d;border-bottom:2px solid #19412a;display:flex;justify-content:space-between}
 h1{margin:0;font-size:18px} .grid{display:grid;grid-template-columns:780px 1fr;gap:12px;padding:12px}
 .card{background:#09110d;border:1px solid #19412a;padding:10px;margin-bottom:12px} .card h2{margin:0 0 8px;font-size:13px;color:#28ff82}
 canvas{background:#08100c;border:1px solid #19412a;display:block} table{width:100%;border-collapse:collapse}
 th{color:#4b8260;text-align:left;font-weight:normal} td,th{padding:3px 5px;border-bottom:1px solid #12281b;white-space:nowrap}
 .THREAT,.ALERT{color:#ff3737;font-weight:bold} .SUSPICIOUS,.MERGED{color:#ffdc46} .NOT_THREAT,.CLEARED{color:#28ff82} .UNVERIFIED{color:#bbb}
 #banner{display:none;margin:12px;padding:12px;border:2px solid #ff3737;background:#2a0a0a;color:#ff7070;font-size:16px;font-weight:bold}
 .stat{display:inline-block;margin-right:20px} .stat b{font-size:18px;color:#fff}
 .bar{display:inline-block;height:8px;background:#28ff82;vertical-align:middle} .muted{color:#4b8260}
</style></head><body>
<header><h1>AI NIGHT-PATROL COMMAND CENTER <span class="muted" style="font-size:12px">- the drone sends behaviour data, THIS server decides THREAT / NOT_THREAT</span></h1><div id="sess"></div></header>
<div id="banner"></div>
<div class="grid">
 <div><div class="card"><h2>LIVE ANALYTICS <span class="muted" id="mapinfo"></span></h2>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div><div class="muted">P(THREAT) over time - per track</div><canvas id="c1" style="width:100%;height:230px"></canvas></div>
    <div><div class="muted">current P(THREAT) by track</div><canvas id="c2" style="width:100%;height:230px"></canvas></div>
    <div><div class="muted">decisions made (alerts / merged / cleared)</div><canvas id="c3" style="width:100%;height:200px"></canvas></div>
    <div><div class="muted">prediction quality vs simulator truth</div><canvas id="c4" style="width:100%;height:200px"></canvas></div>
   </div></div>
  <div class="card"><h2>SUMMARY</h2><div id="stats"></div></div>
  <div class="card"><h2>MODEL AT THIS SERVER</h2><div id="model" class="muted"></div></div></div>
 <div>
  <div class="card"><h2>TRACKS - observation received -> prediction made</h2>
   <table><thead><tr><th>TRACK</th><th>VERDICT</th><th>P(THREAT)</th><th></th><th>SPEED</th><th>LOITER</th><th>PROXIMITY</th><th>ROUTE DEV</th><th>WHY (push on threat score)</th><th>INCIDENT</th><th>SIM TRUTH*</th></tr></thead>
   <tbody id="tracks"></tbody></table></div>
  <div class="card"><h2>ALERT / DECISION LOG (newest first)</h2>
   <table><thead><tr><th>TIME</th><th>EVENT</th><th>TRACK</th><th>INCIDENT</th><th>P</th><th>XY (m)</th><th>SIM TRUTH*</th></tr></thead><tbody id="log"></tbody></table>
   <div class="muted" style="margin-top:6px">* SIM TRUTH (HOSTILE / BENIGN) comes from the simulator and is used ONLY for the evaluation counters, never for the prediction.</div></div>
 </div></div>
<script>
const col={THREAT:'#ff3b3b',SUSPICIOUS:'#ffd23f',NOT_THREAT:'#2dff88',UNVERIFIED:'#c9d4cf'};
const pc=p=>p>=0.5?'#ff3b3b':p>=0.3?'#ffd23f':'#2dff88';
function setup(id){const c=document.getElementById(id),r=c.getBoundingClientRect(),D=window.devicePixelRatio||1,W=Math.round(r.width)||360,H=Math.round(r.height)||200;
 if(c.width!==W*D||c.height!==H*D){c.width=W*D;c.height=H*D;}const g=c.getContext('2d');g.setTransform(D,0,0,D,0,0);g.clearRect(0,0,W,H);return {g,W,H};}
function T(g,t,x,y,c,al,f){g.font=f||'11px Consolas,monospace';g.fillStyle=c||'#6fbf8f';g.textAlign=al||'left';g.fillText(t,x,y);}
function empty(g,W,H,m){T(g,m,W/2,H/2,'#4b8260','center');}
function chartTime(st,now){const {g,W,H}=setup('c1'),L=30,R=44,Tp=8,B=18,w=W-L-R,h=H-Tp-B;
 const tr=Object.entries(st.tracks).filter(([id,t])=>(t.hist||[]).length>1).sort((a,b)=>b[1].wall-a[1].wall).slice(0,6);
 g.lineWidth=1;g.strokeStyle='#12301f';
 for(const v of [0,0.25,0.5,0.75,1]){const y=Tp+h*(1-v);g.beginPath();g.moveTo(L,y);g.lineTo(L+w,y);g.stroke();T(g,v.toFixed(2),L-4,y+4,'#4b8260','right','10px Consolas,monospace');}
 for(const [v,c] of [[0.5,'#ff3b3b'],[0.3,'#ffd23f']]){const y=Tp+h*(1-v);g.setLineDash([5,4]);g.strokeStyle=c;g.globalAlpha=0.6;g.beginPath();g.moveTo(L,y);g.lineTo(L+w,y);g.stroke();g.setLineDash([]);g.globalAlpha=1;}
 if(!tr.length){empty(g,W,H,'waiting for tracks...');return;}
 const N=40;g.lineJoin='round';g.lineCap='round';
 for(const [id,t] of tr){const hs=t.hist,c=col[t.verdict.status]||'#ddd',live=now-t.wall<=6;g.globalAlpha=live?1:0.35;g.strokeStyle=c;g.lineWidth=2;g.beginPath();
  hs.forEach((p,k)=>{const x=L+w*(k+N-hs.length)/(N-1),y=Tp+h*(1-p);k?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
  const p=hs[hs.length-1],x=L+w*(N-1)/(N-1),y=Tp+h*(1-p);g.beginPath();g.arc(x,y,3,0,7);g.fillStyle=c;g.fill();g.globalAlpha=1;T(g,id.replace('TRK-','T'),x+6,y+4,c,'left','10px Consolas,monospace');}
 T(g,'older',L,H-4,'#4b8260','left','10px Consolas,monospace');T(g,'now',L+w,H-4,'#4b8260','right','10px Consolas,monospace');}
function chartBars(st,now){const {g,W,H}=setup('c2'),L=52,R=44,rows=Object.entries(st.tracks).filter(([id,t])=>t.verdict.p_threat!=null).sort((a,b)=>b[1].verdict.p_threat-a[1].verdict.p_threat).slice(0,8);
 if(!rows.length){empty(g,W,H,'waiting for tracks...');return;}
 const rh=Math.min(26,(H-8)/rows.length),bw=W-L-R;
 rows.forEach(([id,t],k)=>{const p=t.verdict.p_threat,y=4+k*rh,live=now-t.wall<=6;g.globalAlpha=live?1:0.4;
  T(g,id,L-6,y+rh*0.62,'#aaffcd','right','11px Consolas,monospace');g.fillStyle='#0d1f15';g.fillRect(L,y+3,bw,rh-8);g.fillStyle=pc(p);g.fillRect(L,y+3,Math.max(2,bw*p),rh-8);
  T(g,p.toFixed(2),L+bw+6,y+rh*0.62,pc(p));g.globalAlpha=1;});
 for(const v of [0.3,0.5]){const x=L+bw*v;g.setLineDash([3,3]);g.strokeStyle=v==0.5?'#ff3b3b':'#ffd23f';g.globalAlpha=0.5;g.beginPath();g.moveTo(x,2);g.lineTo(x,H-2);g.stroke();g.setLineDash([]);g.globalAlpha=1;}}
function vbars(id,items){const {g,W,H}=setup(id),L=10,R=10,Tp=22,B=34,n=items.length,gap=18,bw=(W-L-R-gap*(n-1))/n,mx=Math.max(1,...items.map(i=>i[1]));
 items.forEach(([name,v,c],k)=>{const x=L+k*(bw+gap),h=(H-Tp-B)*v/mx,y=H-B-h;g.fillStyle=c+'33';g.fillRect(x,Tp,bw,H-Tp-B);g.fillStyle=c;g.fillRect(x,y,bw,h);
  T(g,String(v),x+bw/2,Math.min(y-5,H-B-5)+(v?0:0),'#fff','center','bold 13px Consolas,monospace');T(g,name,x+bw/2,H-14,c,'center');});}
function draw(st){const now=Date.now()/1000;chartTime(st,now);chartBars(st,now);
 const ev=st.events,E=st.eval;
 vbars('c3',[['ALERT',ev.filter(e=>e.type=='ALERT').length,'#ff3b3b'],['MERGED',ev.filter(e=>e.type=='MERGED').length,'#ffd23f'],['CLEARED',ev.filter(e=>e.type=='CLEARED').length,'#2dff88']]);
 vbars('c4',[['TP',E.TP,'#2dff88'],['TN',E.TN,'#2dff88'],['FP',E.FP,'#ffd23f'],['FN',E.FN,'#ff3b3b']]);
 const live=Object.values(st.tracks).filter(t=>now-t.wall<=6).length;
 document.getElementById('mapinfo').textContent='  |  live tracks: '+live+(st.drone?'  |  drone '+st.drone.mode:'');}
async function tick(){try{
 const st=await (await fetch('/api/state')).json(), now=Date.now()/1000;
 document.getElementById('sess').textContent='session '+st.session_started+(st.active?'  ● LIVE':'  ■ STOPPED')+'   observations: '+st.n_obs;
 const rows=Object.entries(st.tracks).sort((a,b)=>b[1].wall-a[1].wall).slice(0,18).map(([id,t])=>{const o=t.obs,v=t.verdict,f=o.features,p=v.p_threat;
  return `<tr style="opacity:${now-t.wall>6?0.45:1}"><td>${id}</td><td class="${v.status}">${v.status}</td><td>${p==null?'--':p.toFixed(2)}</td>
  <td>${p==null?'':'<span class="bar" style="width:'+Math.round(p*70)+'px;background:'+(p>=0.5?'#ff3737':p>=0.3?'#ffdc46':'#28ff82')+'"></span>'}</td>
  <td>${f?f.speed.toFixed(1)+' m/s':'--'}</td><td>${f?f.loiter.toFixed(2):'--'}</td><td>${f?f.proximity.toFixed(1)+' m':'--'}</td><td>${f?f.route_dev.toFixed(1)+' m':'--'}</td>
  <td class="muted">${v.why?Object.entries(v.why).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,2).map(([k,x])=>k+' '+(x>=0?'+':'')+x).join(', '):''}</td>
  <td>${v.incident_id||''}</td><td class="muted">${t.truth}</td></tr>`}).join('');
 document.getElementById('tracks').innerHTML=rows;
 document.getElementById('log').innerHTML=st.events.slice().reverse().slice(0,30).map(e=>`<tr><td>${e.time}</td><td class="${e.type}">${e.type}</td><td>${e.track_id}</td><td>${e.incident_id||''}</td><td>${e.p_threat==null?'':e.p_threat.toFixed(2)}</td><td>${e.position.x.toFixed(0)}, ${e.position.y.toFixed(0)}</td><td class="muted">${e.truth}</td></tr>`).join('');
 const E=st.eval,n=E.TP+E.FP+E.FN+E.TN,acc=n?((E.TP+E.TN)/n*100).toFixed(0)+' %':'n/a',rec=(E.TP+E.FN)?(E.TP/(E.TP+E.FN)*100).toFixed(0)+' %':'n/a',pre=(E.TP+E.FP)?(E.TP/(E.TP+E.FP)*100).toFixed(0)+' %':'n/a';
 document.getElementById('stats').innerHTML=`<span class="stat">alerts <b>${st.events.filter(e=>e.type=='ALERT').length}</b></span><span class="stat">merged <b>${st.events.filter(e=>e.type=='MERGED').length}</b></span><span class="stat">cleared <b>${st.events.filter(e=>e.type=='CLEARED').length}</b></span>
 <br><br><b>Prediction quality so far (vs simulator truth)</b><br>TP ${E.TP} &nbsp; FP ${E.FP} &nbsp; FN ${E.FN} &nbsp; TN ${E.TN} &nbsp;|&nbsp; accuracy ${acc} &nbsp; recall ${rec} &nbsp; precision ${pre}
 <br><br><span class="muted">targets in world: ${st.world.active_targets??0} (intruders ${st.world.active_intruders??0})</span>`;
 const act=Object.entries(st.tracks).filter(([id,t])=>t.verdict.status==='THREAT'&&now-t.wall<6),b=document.getElementById('banner');
 if(act.length){b.style.display='block';b.textContent='⚠ THREAT: '+act.map(([id,t])=>id+' ('+(t.verdict.incident_id||'')+') at ('+t.obs.position.x.toFixed(0)+', '+t.obs.position.y.toFixed(0)+') m').join('  |  ');}else b.style.display='none';
 draw(st);}catch(e){}}
async function model(){try{const m=await (await fetch('/api/model')).json();const i=m.info||{};
 document.getElementById('model').innerHTML='logistic regression on '+m.features.join(', ')+'<br>held-out test: accuracy '+((i.accuracy||0)*100).toFixed(0)+' %, recall '+((i.recall||0)*100).toFixed(0)+' %, AUC '+(i.auc||0).toFixed(2)+'<br>weights: '+JSON.stringify(i.weights||{});}catch(e){}}
setInterval(tick,500);tick();model();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/state":
            with LOCK:
                self._send(200, json.dumps(STATE))
        elif self.path == "/api/model":
            self._send(200, json.dumps({"features": ENGINE.features, "info": ENGINE.info}))
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html")
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, '{"error":"bad json"}')
        with LOCK:
            if self.path == "/api/observations":
                try:
                    return self._send(200, json.dumps(process_observation(body)))
                except (KeyError, TypeError) as e:
                    return self._send(400, json.dumps({"error": f"bad observation: {e}"}))
            if self.path == "/api/session/reset":
                reset_state()
                print("[CC] session reset")
            elif self.path == "/api/session/stop":
                STATE["active"] = False
                print("[CC] session stopped")
            elif self.path == "/api/world-state":
                STATE["world"] = body
            else:
                return self._send(404, '{"error":"not found"}')
        self._send(200, '{"ok":true}')


if __name__ == "__main__":
    print(f"Command center running ->  http://{HOST}:{PORT}   (Ctrl+C to stop)")
    print(f"Prediction model: {MODEL_PATH.name}  acc={ENGINE.info.get('accuracy', 0):.2f} AUC={ENGINE.info.get('auc', 0):.2f}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()