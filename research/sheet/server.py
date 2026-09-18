"""Local dashboard server. The CSV cache is the database; this is the face.

    python research/sheet/server.py           # http://127.0.0.1:8777
    python research/sheet/server.py --port 9000

WHY THIS EXISTS
The workbook holds everything but every question costs you a filter, a sort
and a horizontal scroll across 47 columns. This serves the same CSVs with the
friction removed: one search box, one click to sort, one click to open a full
card, one click to arm a price alert.

Nothing is duplicated. The CSVs written by build_workbook.py remain the single
source of truth, and this reads them on every request (they are small and the
OS caches them), so a background refresh shows up without restarting anything.

Binds to 127.0.0.1 only -- this is your machine, not a website.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
WATCH = os.path.join(HERE, "watchlist.csv")
FOLLOW = os.path.join(HERE, "following.csv")

app = Flask(__name__)


def load(name: str) -> pd.DataFrame:
    p = os.path.join(CACHE, f"{name}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def jsonable(df: pd.DataFrame) -> list[dict]:
    d = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(d.to_json(orient="records"))


@app.route("/api/<name>")
def api(name):
    ok = {"summary", "technicals", "fundamentals", "macro", "whyitmoved",
          "momentum", "bigmoney", "analysts", "buyzone"}
    if name not in ok:
        return jsonify({"error": "unknown table"}), 404
    df = load(name)
    return jsonify({"rows": jsonable(df), "cols": list(df.columns),
                    "n": len(df)})


@app.route("/api/logs")
def logs():
    """The three paper ledgers plus a summary line for each."""
    out, summ = {}, {}
    for strat in ("analyst", "buyzone", "bigmoney"):
        p = os.path.join(HERE, "logs", f"{strat}_log.csv")
        if not os.path.exists(p):
            out[strat], summ[strat] = [], {}
            continue
        d = pd.read_csv(p)
        out[strat] = jsonable(d)
        pn = pd.to_numeric(d.get("pnl_pct"), errors="coerce")
        vs = pd.to_numeric(d.get("vs_spy_pct"), errors="coerce")
        summ[strat] = {
            "n": len(d),
            "open": int((d.get("status") == "OPEN").sum()) if "status" in d else 0,
            "closed": int((d.get("status") != "OPEN").sum()) if "status" in d else 0,
            "win": round(float((pn > 0).mean() * 100), 1) if len(pn.dropna()) else None,
            "avg": round(float(pn.mean()), 2) if len(pn.dropna()) else None,
            "vs_spy": round(float(vs.mean()), 2) if len(vs.dropna()) else None,
            "best": round(float(pn.max()), 2) if len(pn.dropna()) else None,
            "worst": round(float(pn.min()), 2) if len(pn.dropna()) else None,
        }
    return jsonify({"logs": out, "summary": summ, "backtest": BACKTEST})


# what the backtest actually said about each rule, shown next to its ledger so
# a live P&L can never be read without the measured prior sitting beside it
BACKTEST = {
    "analyst": {"verdict": "FAILS", "edge": "-0.67% to -1.55% / month",
                "detail": "11,570 dated events, same-stock nearby-day control, "
                          "p~0.0005. Placebo on random dates: -0.01%. Entering "
                          "on the public rating date is measurably worse than "
                          "a nearby day."},
    "buyzone": {"verdict": "FAILS", "edge": "-0.23% per quarter",
                "detail": "158 rebalance dates on a point-in-time universe, "
                          "vs random baskets of the same size. Ahead on only "
                          "46% of dates, p=0.074. An earlier 2023-26 test put "
                          "it at -1.97%, p=0.002."},
    "bigmoney": {"verdict": "PROBATION", "edge": "+0.70% (proxy only)",
                 "detail": "No free historical option chains exist, so there is "
                           "no past to test. The closest proxy -- share volume "
                           "above 1.5x its 60-day average -- scored +0.70% with "
                           "p=0.359, i.e. indistinguishable from noise. This "
                           "ledger IS the experiment."},
}


@app.route("/api/meta")
def meta():
    p = os.path.join(CACHE, "summary.csv")
    stamp = (time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
             if os.path.exists(p) else "never")
    age_min = ((time.time() - os.path.getmtime(p)) / 60
               if os.path.exists(p) else -1)
    w = []
    if os.path.exists(WATCH):
        w = list(csv.DictReader(open(WATCH, encoding="utf-8")))
    f = []
    if os.path.exists(FOLLOW):
        f = list(csv.DictReader(open(FOLLOW, encoding="utf-8")))
    return jsonify({"updated": stamp, "age_min": round(age_min, 1),
                    "watchlist": w, "following": f})


@app.route("/api/watch", methods=["POST"])
def add_watch():
    b = request.get_json(force=True)
    row = {"symbol": (b.get("symbol") or "").upper(),
           "type": b.get("type") or "price_below",
           "level": b.get("level") or "",
           "note": b.get("note") or "", "fired": ""}
    if not row["symbol"]:
        return jsonify({"error": "no symbol"}), 400
    exists = os.path.exists(WATCH)
    rows = list(csv.DictReader(open(WATCH, encoding="utf-8"))) if exists else []
    rows = [r for r in rows
            if not (r.get("symbol") == row["symbol"]
                    and r.get("type") == row["type"])]
    rows.append(row)
    with open(WATCH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "type", "level", "note", "fired"])
        w.writeheader()
        w.writerows(rows)
    return jsonify({"ok": True, "count": len(rows)})


@app.route("/api/watch/<symbol>/<typ>", methods=["DELETE"])
def del_watch(symbol, typ):
    if not os.path.exists(WATCH):
        return jsonify({"ok": True})
    rows = [r for r in csv.DictReader(open(WATCH, encoding="utf-8"))
            if not (r.get("symbol") == symbol.upper() and r.get("type") == typ)]
    with open(WATCH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["symbol", "type", "level", "note", "fired"])
        w.writeheader()
        w.writerows(rows)
    return jsonify({"ok": True, "count": len(rows)})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>S&P 500 Desk</title>
<style>
:root{
 --bg:#0e1116; --panel:#161b22; --line:#262d36; --txt:#e6edf3; --dim:#8b949e;
 --up:#3fb950; --dn:#f85149; --warn:#d29922; --accent:#58a6ff; --chip:#1f2630;
}
:root[data-t="light"]{
 --bg:#f6f8fa; --panel:#fff; --line:#d8dee4; --txt:#1f2328; --dim:#636c76;
 --up:#1a7f37; --dn:#cf222e; --warn:#9a6700; --accent:#0969da; --chip:#eef1f4;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
 font:13px/1.45 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:20;background:var(--panel);
 border-bottom:1px solid var(--line);padding:10px 14px;display:flex;gap:10px;
 align-items:center;flex-wrap:wrap}
h1{font-size:14px;margin:0 10px 0 0;font-weight:700;letter-spacing:.3px}
input,select,button{background:var(--chip);color:var(--txt);
 border:1px solid var(--line);border-radius:7px;padding:7px 10px;font:inherit;outline:none}
input:focus,select:focus{border-color:var(--accent)}
#q{min-width:250px;flex:1}
button{cursor:pointer}
button:hover{border-color:var(--accent)}
.tabs{display:flex;gap:4px;flex-wrap:wrap;padding:8px 14px;background:var(--panel);
 border-bottom:1px solid var(--line);position:sticky;top:51px;z-index:19}
.tab{padding:5px 12px;border-radius:99px;cursor:pointer;border:1px solid transparent;
 color:var(--dim);font-size:12px;white-space:nowrap}
.tab:hover{color:var(--txt)}
.tab.on{background:var(--accent);color:#fff;font-weight:600}
.presets{display:flex;gap:6px;flex-wrap:wrap;padding:8px 14px}
.chip{padding:4px 11px;border-radius:99px;background:var(--chip);cursor:pointer;
 font-size:11.5px;border:1px solid var(--line);color:var(--dim)}
.chip:hover{color:var(--txt);border-color:var(--accent)}
.chip.on{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
#wrap{padding:0 14px 60px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{position:sticky;top:96px;background:var(--panel);text-align:right;padding:7px 9px;
 border-bottom:1px solid var(--line);cursor:pointer;font-size:11px;
 text-transform:uppercase;letter-spacing:.4px;color:var(--dim);white-space:nowrap;z-index:10}
th:first-child,td:first-child{text-align:left}
th:hover{color:var(--accent)}
td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
tr:hover td{background:var(--chip)}
tbody tr{cursor:pointer}
.sym{font-weight:700;color:var(--accent)}
.up{color:var(--up)} .dn{color:var(--dn)} .wn{color:var(--warn)} .dim{color:var(--dim)}
.pill{padding:1px 7px;border-radius:99px;font-size:10.5px;font-weight:600}
.pill.g{background:rgba(63,185,80,.15);color:var(--up)}
.pill.r{background:rgba(248,81,73,.15);color:var(--dn)}
.pill.y{background:rgba(210,153,34,.15);color:var(--warn)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.72);display:none;z-index:50;
 padding:36px 16px;overflow:auto}
#modal.on{display:block}
#card{max-width:960px;margin:0 auto;background:var(--panel);border:1px solid var(--line);
 border-radius:14px;padding:20px 22px}
#card h2{margin:0 0 2px;font-size:20px}
.verdict{background:var(--chip);border-left:3px solid var(--accent);padding:11px 13px;
 border-radius:8px;margin:12px 0;line-height:1.6;white-space:normal}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:8px;margin-top:12px}
.kv{background:var(--chip);border-radius:8px;padding:8px 10px}
.kv b{display:block;font-size:10px;color:var(--dim);text-transform:uppercase;
 letter-spacing:.4px;font-weight:600;margin-bottom:2px}
.kv span{font-size:14px;font-variant-numeric:tabular-nums}
.sec{margin-top:16px;font-size:11px;text-transform:uppercase;letter-spacing:.6px;
 color:var(--dim);border-bottom:1px solid var(--line);padding-bottom:5px;font-weight:700}
.alertbar{display:flex;gap:7px;align-items:center;margin-top:14px;flex-wrap:wrap}
#count{color:var(--dim);font-size:12px}
#stamp{color:var(--dim);font-size:11.5px;margin-left:auto}
.stale{color:var(--warn)!important}
</style></head><body>
<header>
 <h1>S&amp;P 500 DESK</h1>
 <input id="q" placeholder="search symbol, name, sector, anything...">
 <select id="sector"><option value="">all sectors</option></select>
 <button id="theme">theme</button>
 <span id="stamp"></span>
</header>
<div class="tabs" id="tabs"></div>
<div class="presets" id="presets"></div>
<div style="padding:0 14px 6px"><span id="count"></span></div>
<div id="wrap"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div>
<div id="modal"><div id="card"></div></div>
<script>
const TABS=[["summary","Summary"],["logs","Logs"],["momentum","Momentum"],
 ["whyitmoved","Why It Moved"],["analysts","Analysts"],["buyzone","Buy Zone"],
 ["bigmoney","Big Money"],["macro","Macro"],["fundamentals","Fundamentals"],
 ["technicals","Technicals"]];
const COLS={
 summary:["symbol","shortName","sector","price","chg_1d_pct","RATE","MOMENTUM_SCORE",
   "vs_200sma","zone_status","perfect_buy","discount_to_buy_pct","rsi14","halal_auto","hot","last_action"],
 momentum:["symbol","sector","price","MOMENTUM_SCORE","hot","ret_1m_pct","ret_3m_pct",
   "ret_6m_pct","mom_12_1_pct","rs_3m_vs_spy","rsi14","vs_200sma"],
 whyitmoved:["symbol","name","sector","move_pct","move_in_sigma","move_where",
   "volume_x_normal","vs_sector_pct","company_specific_pp","biggest_macro_driver","analyst_action"],
 analysts:["symbol","shortName","FRESH","days_ago","last_action","last_rating",
   "last_analyst","last_firm","last_target_upside_pct","consensus","tilt","net_upgrades"],
 buyzone:["symbol","price","RATE","zone_status","buy_zone_low","buy_zone_high",
   "perfect_buy","discount_to_buy_pct","support","anchor_trend","buy_note"],
 bigmoney:["symbol","shortName","unusual_score","UNUSUAL","skew","call_pct",
   "vol_vs_oi","notional_vs_mcap_bp","max_strike"],
 macro:["symbol","beta_spy","beta_tlt","beta_gld","beta_oil","beta_dxy","beta_vix",
   "r2","fomc_amplifier","resid_vol_ann_pct"],
 fundamentals:["symbol","shortName","sector","RATE","rate_grade","score_value",
   "score_quality","score_safety","score_growth","trailingPE","forwardPE","profitMargins",
   "returnOnEquity","debtToEquity","halal_auto"],
 technicals:["symbol","price","chg_1d_pct","sma50","sma200","vs_200sma","support",
   "support_touches","resistance","rsi14","atr_pct","pct_from_52w_high"]};
const PRESETS={summary:[
  ["Momentum leaders",r=>r.MOMENTUM_SCORE>=85],["HOT",r=>r.hot==="HOT"],
  ["In buy zone",r=>r.zone_status==="IN ZONE"||r.zone_status==="AT ZONE"],
  ["Near perfect buy",r=>r.discount_to_buy_pct!=null&&r.discount_to_buy_pct>=-3],
  ["Rate A/B",r=>r.RATE>=60],["Halal pass",r=>r.halal_auto==="pass"],
  ["Below 200d",r=>r.vs_200sma==="BELOW"],["Oversold",r=>r.rsi14<35],
  ["Moved >2%",r=>Math.abs(r.chg_1d_pct)>2]],
 whyitmoved:[["Big moves",r=>Math.abs(r.move_in_sigma)>2],
  ["Gapped",r=>r.move_where==="GAP (overnight)"],
  ["Volume confirmed",r=>r.volume_confirms==="YES"],
  ["Company-specific",r=>Math.abs(r.company_specific_pp)>Math.abs(r.explained_pp)*1.5]],
 analysts:[["Today",r=>r.FRESH==="TODAY-ISH"],["This week",r=>r.days_ago<=7],
  ["Upgrades",r=>String(r.last_action||"").startsWith("Upgrade")],
  ["Downgrades",r=>String(r.last_action||"").startsWith("Downgrade")],
  ["Upgrading tilt",r=>r.tilt==="UPGRADING"]],
 momentum:[["Top 10%",r=>r.MOMENTUM_SCORE>=90],["HOT",r=>r.hot==="HOT"]],
 bigmoney:[["Unusual",r=>r.UNUSUAL==="UNUSUAL"],["Call-heavy",r=>r.skew==="CALL-HEAVY"],
  ["Put-heavy",r=>r.skew==="PUT-HEAVY"]]};
let tab="summary",data={},sortk=null,sortd=-1,preset=null,meta={};
const $=s=>document.querySelector(s);
const num=v=>typeof v==="number"&&isFinite(v);
function fmt(k,v){
 if(v===null||v===undefined||v==="")return'<span class="dim">-</span>';
 if(!num(v))return String(v).length>46?String(v).slice(0,46)+"...":String(v);
 if(/price|support|resistance|sma|buy|target|anchor/i.test(k)&&Math.abs(v)>1)
   return"$"+v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
 if(/_pct|_pp|pct_|chg_|ret_|mom_|rs_|score|RATE|upside/i.test(k)){
   const c=v>0?"up":v<0?"dn":"dim";return`<span class="${c}">${v>0?"+":""}${v.toFixed(2)}</span>`;}
 return Math.abs(v)>=1e6?(v/1e6).toFixed(0)+"M":v.toFixed(2);
}
function pill(v){
 const s=String(v);
 if(["HOT","pass","UNUSUAL","IN ZONE","AT ZONE","UPGRADING","STRONG BUY","BUY","TODAY-ISH","ABOVE","YES","CALL-HEAVY"].includes(s))
   return`<span class="pill g">${s}</span>`;
 if(["FAIL","BELOW","DOWNGRADING","SELL","PUT-HEAVY","BELOW ZONE"].includes(s))
   return`<span class="pill r">${s}</span>`;
 if(["elevated","THIS WEEK","NEAR ZONE","HOLD","GAP (overnight)"].includes(s))
   return`<span class="pill y">${s}</span>`;
 return null;
}
async function get(t){if(!data[t])data[t]=(await (await fetch("/api/"+t)).json());return data[t];}
function rows(){
 const d=data[tab]?data[tab].rows:[];
 const q=$("#q").value.trim().toLowerCase(), sec=$("#sector").value;
 // an exact ticker wins outright: "ON" must mean ON Semiconductor, not every
 // row containing the letters "on" (Corporation, Johnson, Information...)
 const exact=q?d.filter(x=>String(x.symbol||"").toLowerCase()===q):[];
 let r=exact.length?exact.filter(x=>(!sec||x.sector===sec)&&(!preset||preset(x))):d.filter(x=>{
  if(sec&&x.sector!==sec)return false;
  if(preset&&!preset(x))return false;
  if(!q)return true;
  return Object.values(x).some(v=>v!==null&&String(v).toLowerCase().includes(q));});
 if(sortk)r=[...r].sort((a,b)=>{
  const x=a[sortk],y=b[sortk];
  if(x===null||x===undefined)return 1; if(y===null||y===undefined)return -1;
  return (num(x)&&num(y))?(x-y)*sortd:String(x).localeCompare(String(y))*sortd;});
 return r;
}
function draw(){
 if(!document.getElementById("th")||document.getElementById("logsbox")){
   $("#wrap").innerHTML='<table><thead id="th"></thead><tbody id="tb"></tbody></table>';}
 const cols=(COLS[tab]||[]).filter(c=>data[tab].cols.includes(c));
 $("#th").innerHTML="<tr>"+cols.map(c=>`<th data-k="${c}">${c.replace(/_/g," ")}${sortk===c?(sortd>0?" ^":" v"):""}</th>`).join("")+"</tr>";
 const r=rows();
 $("#count").textContent=`${r.length} of ${data[tab].n} rows`;
 $("#tb").innerHTML=r.slice(0,400).map((x,i)=>"<tr data-i="+i+">"+cols.map(c=>{
   const p=pill(x[c]); return"<td>"+(c==="symbol"?`<span class="sym">${x[c]}</span>`:(p||fmt(c,x[c])))+"</td>";}).join("")+"</tr>").join("");
 [...document.querySelectorAll("#th th")].forEach(t=>t.onclick=()=>{
   const k=t.dataset.k; sortd=(sortk===k)?-sortd:-1; sortk=k; draw();});
 [...document.querySelectorAll("#tb tr")].forEach(t=>t.onclick=()=>card(r[+t.dataset.i]));
}
function card(row){
 const all={};
 for(const t of Object.keys(data)) {
   const m=(data[t].rows||[]).find(z=>z.symbol===row.symbol);
   if(m)Object.assign(all,m);
 }
 const sk=["symbol","shortName","name","sector","industry"];
 const v=all.VERDICT?`<div class="verdict"><b>WHY IT MOVED</b><br>${all.VERDICT}</div>`:"";
 const bn=all.buy_note?`<div class="verdict" style="border-color:var(--warn)">${all.buy_note}</div>`:"";
 const groups=[["Price & trend",["price","chg_1d_pct","vs_200sma","pct_vs_200sma","sma50","sma200","high_52w","low_52w","pct_from_52w_high","rsi14","atr_pct"]],
  ["Levels & buy zone",["support","support_touches","pct_to_support","resistance","resistance_touches","zone_status","buy_zone_low","buy_zone_high","perfect_buy","discount_to_buy_pct"]],
  ["Quality",["RATE","rate_grade","score_value","score_quality","score_safety","score_growth","halal_auto","halal_auto_reason","marketCap","trailingPE","forwardPE","profitMargins","returnOnEquity","debtToEquity"]],
  ["Momentum",["MOMENTUM_SCORE","hot","ret_1m_pct","ret_3m_pct","ret_6m_pct","ret_12m_pct","mom_12_1_pct","rs_3m_vs_spy"]],
  ["Today's move",["move_pct","move_in_sigma","move_in_atr","move_where","gap_pct","intraday_pct","volume_x_normal","volume_confirms","sector_move_pct","vs_sector_pct","pct_peers_same_way","company_specific_pp","explained_pp","biggest_macro_driver"]],
  ["Macro betas",["beta_spy","beta_tlt","beta_gld","beta_oil","beta_dxy","beta_vix","r2","fomc_amplifier"]],
  ["Analysts",["FRESH","days_ago","last_action","last_rating","last_analyst","last_firm","last_target","last_target_upside_pct","consensus","targetMeanPrice","consensus_upside_pct","tilt","net_upgrades","n_ratings"]],
  ["Options",["unusual_score","UNUSUAL","skew","call_pct","vol_vs_oi","notional_vs_mcap_bp","max_strike"]],
  ["Sizing",["shares_for_$250_risk","cost_of_that_position","days_to_earnings","next_earnings"]]];
 let h=`<h2>${all.symbol} <span class="dim" style="font-size:14px;font-weight:400">${all.shortName||all.name||""}</span></h2>
  <div class="dim">${all.sector||""} ${all.industry?"· "+all.industry:""}</div>${v}${bn}`;
 for(const [title,keys] of groups){
   const have=keys.filter(k=>all[k]!==null&&all[k]!==undefined&&all[k]!=="");
   if(!have.length)continue;
   h+=`<div class="sec">${title}</div><div class="grid">`+have.map(k=>{
     const p=pill(all[k]);
     return`<div class="kv"><b>${k.replace(/_/g," ")}</b><span>${p||fmt(k,all[k])}</span></div>`;}).join("")+"</div>";
 }
 h+=`<div class="sec">Alert me</div><div class="alertbar">
  <select id="atype">
   <option value="price_below">price drops below</option>
   <option value="price_above">price rises above</option>
   <option value="rsi_below">RSI below</option>
   <option value="near_support">within % of support</option>
   <option value="sma200_cross_down">breaks below 200d</option>
   <option value="in_buy_zone">enters buy zone</option>
   <option value="new_52w_low">new 52-week low</option>
  </select>
  <input id="alvl" style="width:110px" value="${all.perfect_buy||all.price||""}">
  <input id="anote" placeholder="note (optional)" style="width:190px">
  <button id="addal">Add alert</button><span id="alres" class="dim"></span></div>
  <div style="margin-top:16px"><button onclick="document.getElementById('modal').classList.remove('on')">Close</button></div>`;
 $("#card").innerHTML=h; $("#modal").classList.add("on");
 $("#addal").onclick=async()=>{
   const r=await fetch("/api/watch",{method:"POST",headers:{"Content-Type":"application/json"},
     body:JSON.stringify({symbol:all.symbol,type:$("#atype").value,
       level:$("#alvl").value,note:$("#anote").value})});
   const j=await r.json();
   $("#alres").textContent=j.ok?`saved (${j.count} alerts armed)`:"failed";};
}
async function drawLogs(){
 const j=await (await fetch("/api/logs")).json();
 const V={FAILS:"r",WORKS:"g",PROBATION:"y"};
 let h="";
 for(const k of ["analyst","buyzone","bigmoney"]){
   const s=j.summary[k]||{},b=j.backtest[k]||{},rows=j.logs[k]||[];
   h+=`<div class="sec" style="font-size:13px;margin-top:20px">${k.toUpperCase()} LEDGER
     <span class="pill ${V[b.verdict]||''}" style="margin-left:8px">${b.verdict||''}</span></div>
    <div class="verdict" style="border-color:var(--${b.verdict==="FAILS"?"dn":b.verdict==="WORKS"?"up":"warn"})">
     <b>BACKTEST SAYS: ${b.edge||"n/a"}</b><br>${b.detail||""}</div>
    <div class="grid">
     <div class="kv"><b>positions</b><span>${s.n??0}</span></div>
     <div class="kv"><b>open</b><span>${s.open??0}</span></div>
     <div class="kv"><b>closed</b><span>${s.closed??0}</span></div>
     <div class="kv"><b>win rate</b><span>${s.win!=null?s.win+"%":"-"}</span></div>
     <div class="kv"><b>avg P&amp;L</b><span class="${(s.avg||0)>=0?'up':'dn'}">${s.avg!=null?(s.avg>0?"+":"")+s.avg+"%":"-"}</span></div>
     <div class="kv"><b>vs SPY</b><span class="${(s.vs_spy||0)>=0?'up':'dn'}">${s.vs_spy!=null?(s.vs_spy>0?"+":"")+s.vs_spy+"%":"-"}</span></div>
     <div class="kv"><b>best</b><span class="up">${s.best!=null?"+"+s.best+"%":"-"}</span></div>
     <div class="kv"><b>worst</b><span class="dn">${s.worst!=null?s.worst+"%":"-"}</span></div>
    </div>`;
   if(rows.length){
     const cs=["opened","symbol","entry","last_price","pnl_pct","vs_spy_pct",
               "days_held","status","rate","trigger"];
     h+=`<table style="margin-top:10px"><thead><tr>`+
        cs.map(c=>`<th style="position:static">${c.replace(/_/g," ")}</th>`).join("")+
        `</tr></thead><tbody>`+
        rows.slice(0,40).map(r=>"<tr>"+cs.map(c=>{
          const p=pill(r[c]);
          return"<td>"+(c==="symbol"?`<span class="sym">${r[c]}</span>`:
            c==="trigger"?`<span class="dim" style="white-space:normal">${String(r[c]||"").slice(0,64)}</span>`:
            (p||fmt(c,r[c])))+"</td>";}).join("")+"</tr>").join("")+
        `</tbody></table>`;
     if(rows.length>40)h+=`<div class="dim" style="padding:6px 0">... ${rows.length-40} more in logs/${k}_log.csv</div>`;
   } else h+=`<div class="dim" style="padding:8px 0">no entries yet -- run logs.py</div>`;
 }
 h+=`<div class="dim" style="margin:22px 0 40px;line-height:1.7">
   Paper only, marked at the close, no costs. <b>vs SPY is the number that
   matters</b> -- a ledger up 5% while the market rose 6% has lost. Nothing
   here is significant until a ledger has 40+ CLOSED positions, which is
   months. The backtest banner above each ledger is the prior you should
   read the live numbers against.</div>`;
 $("#count").textContent="three paper ledgers";
 $("#th").innerHTML="";$("#tb").innerHTML="";
 $("#wrap").innerHTML=`<div id="logsbox">${h}</div>
   <table><thead id="th"></thead><tbody id="tb"></tbody></table>`;
}
$("#modal").onclick=e=>{if(e.target.id==="modal")e.target.classList.remove("on")};
document.onkeydown=e=>{if(e.key==="Escape")$("#modal").classList.remove("on");
 if(e.key==="/"&&document.activeElement!==$("#q")){e.preventDefault();$("#q").focus();}};
$("#theme").onclick=()=>{const r=document.documentElement;
 const n=r.getAttribute("data-t")==="light"?"dark":"light";
 r.setAttribute("data-t",n);localStorage.setItem("t",n);};
if(localStorage.getItem("t"))document.documentElement.setAttribute("data-t",localStorage.getItem("t"));
$("#q").oninput=()=>draw();
$("#sector").onchange=()=>draw();
function chips(){
 const list=PRESETS[tab]||[];
 $("#presets").innerHTML=list.map((p,i)=>`<span class="chip" data-i="${i}">${p[0]}</span>`).join("");
 [...document.querySelectorAll(".chip")].forEach(c=>c.onclick=()=>{
   const f=list[+c.dataset.i][1];
   if(preset===f){preset=null;c.classList.remove("on");}
   else{preset=f;[...document.querySelectorAll(".chip")].forEach(x=>x.classList.remove("on"));c.classList.add("on");}
   draw();});
}
async function go(t){
 tab=t;preset=null;sortk=null;
 if(t==="logs"){
   [...document.querySelectorAll(".tab")].forEach(x=>x.classList.toggle("on",x.dataset.t===t));
   $("#presets").innerHTML="";await drawLogs();return;}
 [...document.querySelectorAll(".tab")].forEach(x=>x.classList.toggle("on",x.dataset.t===t));
 await get(t);
 if(t==="summary"||!$("#sector").options.length>1){
   const s=[...new Set((data[t].rows||[]).map(r=>r.sector).filter(Boolean))].sort();
   $("#sector").innerHTML='<option value="">all sectors</option>'+s.map(x=>`<option>${x}</option>`).join("");}
 chips();draw();
}
async function stamp(){
 meta=await (await fetch("/api/meta")).json();
 const e=$("#stamp");
 e.textContent=`data ${meta.updated} · ${meta.age_min}m ago · ${(meta.watchlist||[]).length} alerts armed`;
 e.className=meta.age_min>120?"stale":"";
}
$("#tabs").innerHTML=TABS.map(([k,l])=>`<div class="tab" data-t="${k}">${l}</div>`).join("");
[...document.querySelectorAll(".tab")].forEach(t=>t.onclick=()=>go(t.dataset.t));
// preload every table up front: the detail card merges across ALL of them, and
// lazy-loading meant a card was missing whatever tab you had not visited yet.
// Total payload is ~2MB from local disk, so this costs nothing worth saving.
(async()=>{await go("summary");stamp();
  for(const [k] of TABS){if(!data[k])await get(k);} })();
setInterval(async()=>{data={};await get(tab);draw();stamp();},60000);
</script></body></html>"""


if __name__ == "__main__":
    port = 8777
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"\n  S&P 500 desk  ->  http://127.0.0.1:{port}\n")
    print("  '/' focuses search · click a row for the full card · Esc closes")
    print("  data comes from research/sheet/cache/*.csv, re-read every request")
    print("  Ctrl+C to stop\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
