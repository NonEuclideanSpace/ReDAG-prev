"""Build a self-contained, offline HTML debugger for an executed PNG-CoS DAG."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "cos_png_dag"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_artifact(raw: str, run_dir: Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = run_dir / path
        if candidate.is_file():
            return candidate.resolve()
        candidate = run_dir / "nodes" / path
        if candidate.is_file():
            return candidate.resolve()
    # Node records contain absolute paths from the original workspace. Allow a
    # copied run directory to be visualized without silently dropping images.
    parts = path.parts
    if "nodes" in parts:
        suffix = Path(*parts[parts.index("nodes"):])
        candidate = run_dir / suffix
        if candidate.is_file():
            return candidate.resolve()
    if path.name == "final.png" and (run_dir / "final.png").is_file():
        return (run_dir / "final.png").resolve()
    return path


def artifact_record(raw: str, run_dir: Path, assets: dict) -> dict:
    path = resolve_artifact(raw, run_dir)
    exists = path.is_file()
    try:
        key = path.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        key = str(path)
    record = {"key": key, "label": path.name, "path": str(path), "exists": exists,
              "is_image": path.suffix.lower() == ".png"}
    if exists and record["is_image"]:
        if key not in assets:
            with Image.open(path) as image:
                size = list(image.size)
            assets[key] = {"data": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
                           "size": size, "bytes": path.stat().st_size}
        record["size"] = assets[key]["size"]
    return record


def topological_layers(nodes: dict) -> dict[str, int]:
    layers: dict[str, int] = {}
    active: set[str] = set()

    def visit(node_id: str) -> int:
        if node_id in layers:
            return layers[node_id]
        if node_id in active:
            raise ValueError(f"Cycle in DAG at {node_id}")
        if node_id not in nodes:
            raise ValueError(f"Unknown DAG node {node_id}")
        active.add(node_id)
        parents = nodes[node_id]["depends_on"]
        level = 1 + max((visit(parent) for parent in parents), default=-1)
        active.remove(node_id)
        layers[node_id] = level
        return level

    for name in nodes:
        visit(name)
    return layers


def build_payload(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    specification = read_json(run_dir / "graph.json")
    if not specification.get("nodes"):
        raise FileNotFoundError(f"Executed DAG graph.json not found in {run_dir}")
    layers = topological_layers(specification["nodes"])
    assets: dict = {}
    nodes = []
    missing = []
    for node_id, spec in specification["nodes"].items():
        node_dir = run_dir / "nodes" / node_id
        meta = read_json(node_dir / "node.json")
        result = read_json(node_dir / "result.json")
        instruction_path = node_dir / "instruction.txt"
        instruction = instruction_path.read_text(encoding="utf-8").strip() if instruction_path.is_file() else meta.get("target_scope", "")
        input_paths = meta.get("input_artifacts", spec.get("input_artifacts", []))
        output_paths = meta.get("output_artifacts", [])
        if not output_paths:
            output_paths = [str((node_dir / name).resolve()) for name in spec.get("output_artifacts", [])]
        full_draft = node_dir / "full_draft.png"
        if full_draft.is_file() and str(full_draft) not in output_paths:
            output_paths.append(str(full_draft))
        inputs = [artifact_record(str(path), run_dir, assets) for path in input_paths]
        outputs = [artifact_record(str(path), run_dir, assets) for path in output_paths]
        model_refs = [artifact_record(str(path), run_dir, assets) for path in result.get("reference_order", [])]
        missing_outputs = [item["key"] for item in outputs if not item["exists"]]
        if missing_outputs:
            missing.extend(node_id + ": " + key for key in missing_outputs)
        extra = {}
        for name in ("crop.json", "placements.json"):
            if (node_dir / name).is_file():
                extra[name] = read_json(node_dir / name)
        nodes.append({
            "id": node_id, "type": meta.get("node_type", spec.get("node_type", "unknown")),
            "layer": layers[node_id], "depends_on": spec["depends_on"],
            "status": "missing_output" if missing_outputs else result.get("status", "unknown"),
            "instruction": instruction, "inputs": inputs, "outputs": outputs,
            "model_refs": model_refs, "result": result, "extra": extra,
            "reference_count": result.get("reference_count"),
            "metadata_path": str(node_dir / "node.json"),
        })
    edges = []
    for node in nodes:
        for parent in node["depends_on"]:
            used = [item["label"] for item in node["inputs"]
                    if item["key"].startswith("nodes/" + parent + "/")]
            edges.append({"from": parent, "to": node["id"], "artifacts": used})
    comparison = read_json(HERE / "comparison_cos_png_dag.json")
    return {"title": "CoS-inspired PNG DAG · Visual Debugger", "run_dir": str(run_dir),
            "prompt_sha256": specification.get("prompt_sha256", ""),
            "backend": specification.get("backend", ""),
            "limitation": specification.get("limitation", ""),
            "prompt": (HERE / "prompt.txt").read_text(encoding="utf-8").strip(),
            "nodes": nodes, "edges": edges, "assets": assets, "missing": missing,
            "comparison": {"baseline": comparison.get("baseline", {}).get("satisfied"),
                           "dag": comparison.get("draftdag", {}).get("satisfied"),
                           "total": comparison.get("draftdag", {}).get("total")},
            "self_contained": True}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PNG-CoS DAG · Visual Debugger</title>
<style>
:root{--bg:#0b1020;--panel:#141d31;--panel2:#1c2941;--line:#35445e;--text:#eaf1ff;--muted:#9baccc;--accent:#70d9e7;--warn:#f1bb72;--bad:#ff8294}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input{font:inherit}button{cursor:pointer}a{color:#8fdcf5}header{padding:22px 28px 16px;border-bottom:1px solid #2b3a53;background:linear-gradient(125deg,#14223e,#0c1429 70%)}
h1{font-size:25px;letter-spacing:.2px;margin:0 0 8px}.subtitle{color:var(--muted);margin:0 0 12px}.facts{display:flex;flex-wrap:wrap;gap:8px}.fact{background:#22314e;border:1px solid #3b5278;border-radius:999px;padding:5px 10px;color:#cbd9f3}.fact.warn{border-color:#97764b;color:#ffd3a0}
.toolbar{display:flex;align-items:center;gap:12px;padding:12px 28px;border-bottom:1px solid #263650;flex-wrap:wrap;background:#10192b}.toolbar input{background:#182740;border:1px solid #405578;border-radius:8px;color:var(--text);padding:9px 11px;width:265px}.toolbar label{color:var(--muted)}.toolbar button{background:#1d3555;color:#eaf1ff;border:1px solid #406585;border-radius:8px;padding:8px 11px}.toolbar .hint{color:var(--muted);font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:calc(100vh - 178px);min-height:620px}.graph-scroll{overflow:auto;background-image:radial-gradient(#2b3851 0.7px,transparent .7px);background-size:22px 22px}.graph-canvas{position:relative;width:max-content;min-height:100%;padding:28px 34px 60px}.edges{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;z-index:0}.edge{fill:none;stroke:#6886aa;stroke-width:1.8;opacity:.39;transition:opacity .18s,stroke-width .18s}.edge.active{opacity:.95;stroke:#78d8e9;stroke-width:2.7}.edge.dim{opacity:.07}.columns{display:flex;align-items:flex-start;gap:58px;position:relative;z-index:1}.column{width:330px;flex:0 0 330px}.column h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#90a9cd;margin:0 0 15px;padding-left:4px}.node-card{margin:0 0 18px;background:var(--panel);border:1px solid #3a4f6f;border-radius:12px;padding:13px;box-shadow:0 10px 25px #0004;cursor:pointer;transition:transform .15s,border-color .15s,opacity .15s;outline:none}.node-card:hover,.node-card:focus-visible{transform:translateY(-2px);border-color:#78d8e9}.node-card.selected{border-color:#78d8e9;box-shadow:0 0 0 2px #70d9e745,0 10px 25px #0004}.node-card.dim{opacity:.27}.node-card.missing{border-color:#e59b63}.node-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.node-id{font-size:16px;font-weight:700;word-break:break-all}.badge{font-size:11px;color:#bbd2f6;background:#263d5f;border-radius:5px;padding:3px 6px;white-space:nowrap}.badge.ok{color:#8ee2b1;background:#1e4b43}.badge.bad{color:#ffb2b2;background:#68323b}.node-kind{color:var(--muted);font-size:12px;margin:3px 0 9px}.deps{font-size:11px;color:#a6bbd7;margin-bottom:10px;word-break:break-all}.deps strong{color:#d4e7ff}.section-label{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#8fb6d8;margin:9px 0 5px}.output-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.thumb{position:relative;min-width:0;background:#0b1322;border:1px solid #3a4a61;border-radius:6px;overflow:hidden;cursor:zoom-in}.thumb img{display:block;width:100%;height:82px;object-fit:contain;background:#fff}.thumb span{display:block;font-size:10px;color:#d5e2fa;white-space:nowrap;text-overflow:ellipsis;overflow:hidden;padding:3px 5px;background:#17243a}.thumb.missing-thumb{padding:10px;color:#ffb2b2;font-size:11px}.refs{display:flex;gap:6px;overflow-x:auto;padding-bottom:5px}.ref-thumb{flex:0 0 64px}.ref-thumb img{height:48px}.node-note{color:#d7b883;font-size:11px;margin-top:8px}
.inspector{overflow:auto;border-left:1px solid #344862;background:#101a2e;padding:19px}.inspector h2{font-size:20px;margin:0 0 6px}.inspector h3{font-size:13px;color:#a6c1e8;margin:19px 0 8px}.inspector p{margin:4px 0 10px;color:#bfd0e8}.inspector .path{font-size:11px;color:#9fb6d8;word-break:break-all}.inspector .img-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.inspector .img-grid .thumb img{height:115px}.inspector .img-grid .thumb span{white-space:normal;word-break:break-all}.inspector pre{white-space:pre-wrap;word-break:break-word;background:#0b1425;border:1px solid #33435a;padding:11px;border-radius:8px;color:#c9d9ef;font-size:11px;max-height:330px;overflow:auto}.meta-line{color:#a8c2df;font-size:12px;margin:6px 0}.notice{background:#3b3022;border:1px solid #866944;color:#ffe0aa;border-radius:8px;padding:9px;margin:10px 0}.empty{color:#8498b9;font-style:italic;font-size:12px}
dialog{border:1px solid #55708e;border-radius:11px;background:#101a2b;color:#eaf1ff;max-width:96vw;max-height:96vh;padding:12px;box-shadow:0 20px 100px #000b}dialog::backdrop{background:#000c}.modal-head{display:flex;justify-content:space-between;gap:15px;align-items:center;margin-bottom:8px}.modal-head button{background:#2b4263;color:#fff;border:1px solid #6480a5;border-radius:6px;padding:5px 9px}.modal-image{display:block;max-width:92vw;max-height:83vh;object-fit:contain;background:#fff}.modal-caption{font-size:12px;color:#a9bedc;word-break:break-all;margin-top:7px}
@media(max-width:980px){.layout{display:block;height:auto}.graph-scroll{height:65vh}.inspector{height:auto;min-height:35vh;border-left:0;border-top:1px solid #344862}}
</style>
</head>
<body>
<header><h1>CoS 启发式 PNG-DAG · Visual Debugger</h1><p class="subtitle">每条箭头表示真实节点依赖；每张卡展示草稿、渲染输出和实际参考图。点击节点查看全部输入、模型参考顺序与元数据。</p><div class="facts" id="facts"></div></header>
<div class="toolbar"><label for="search">搜索节点</label><input id="search" placeholder="例如 cat_truck / ladybug"><button id="clear">清除高亮</button><span class="hint">横向滚动浏览 DAG；点击图片放大。所有图片已内嵌，无需网络或本地服务。</span></div>
<div class="layout"><main class="graph-scroll"><div class="graph-canvas" id="canvas"><svg class="edges" id="edges"></svg><div class="columns" id="columns"></div></div></main><aside class="inspector" id="inspector"><h2>节点详情</h2><p>点击任意节点查看。</p></aside></div>
<dialog id="lightbox"><div class="modal-head"><strong id="modal-title"></strong><button id="modal-close">关闭 Esc</button></div><img class="modal-image" id="modal-image" alt="放大的节点图片"><div class="modal-caption" id="modal-caption"></div></dialog>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
  const data=JSON.parse(document.getElementById('payload').textContent);
  const byId=Object.fromEntries(data.nodes.map(n=>[n.id,n]));
  let selected=null;
  const facts=document.getElementById('facts');
  function pill(s,cl){const x=document.createElement('span');x.className='fact'+(cl?' '+cl:'');x.textContent=s;facts.appendChild(x)}
  pill(data.nodes.length+' 节点');pill(data.edges.length+' 依赖边');pill('已内嵌 '+Object.keys(data.assets).length+' 张 PNG');
  if(data.comparison.dag!==null&&data.comparison.dag!==undefined){pill('本 DAG '+data.comparison.dag+'/'+data.comparison.total);pill('Baseline '+data.comparison.baseline+'/'+data.comparison.total)}
  if(data.missing.length)pill(data.missing.length+' 个缺失输出','warn');
  pill('PNG references / 非 clean-latent','warn');
  const modal=document.getElementById('lightbox');
  function openImage(item){if(!item.exists||!data.assets[item.key])return;document.getElementById('modal-title').textContent=item.key;document.getElementById('modal-caption').textContent=item.path+' · '+item.size[0]+'×'+item.size[1];document.getElementById('modal-image').src=data.assets[item.key].data;modal.showModal()}
  document.getElementById('modal-close').onclick=()=>modal.close();modal.addEventListener('click',e=>{if(e.target===modal)modal.close()});
  function thumb(item,small,label){const box=document.createElement('div');box.className='thumb'+(small?' ref-thumb':'')+(item.exists?'':' missing-thumb');box.title=item.path;if(item.exists&&item.is_image&&data.assets[item.key]){const im=document.createElement('img');im.src=data.assets[item.key].data;im.alt=item.key;box.appendChild(im);box.onclick=(ev)=>{ev.stopPropagation();openImage(item)}}else{box.textContent=item.exists?'非图片输入':'缺失: '+item.label}const cap=document.createElement('span');cap.textContent=(label?label+' · ':'')+item.label;box.appendChild(cap);return box}
  function section(parent,label,items,small){const heading=document.createElement('div');heading.className='section-label';heading.textContent=label;parent.appendChild(heading);if(!items.length){const e=document.createElement('div');e.className='empty';e.textContent='无';parent.appendChild(e);return}const row=document.createElement('div');row.className=small?'refs':'output-grid';items.forEach((item,i)=>row.appendChild(thumb(item,small,small?String.fromCharCode(65+i):'')));parent.appendChild(row)}
  const cols=document.getElementById('columns');const levels=[...new Set(data.nodes.map(n=>n.layer))].sort((a,b)=>a-b);
  const labels=['0 · 独立单元','1 · 局部关系','2 · 跨分支关系','3 · 空间合成','4 · 最终生成'];
  levels.forEach(level=>{const col=document.createElement('div');col.className='column';const h=document.createElement('h2');h.textContent=labels[level]||('Layer '+level);col.appendChild(h);data.nodes.filter(n=>n.layer===level).forEach(n=>{
    const card=document.createElement('div');card.className='node-card'+(n.status==='missing_output'?' missing':'');card.id='node-'+n.id;card.tabIndex=0;
    const head=document.createElement('div');head.className='node-head';const id=document.createElement('div');id.className='node-id';id.textContent=n.id;head.appendChild(id);const status=document.createElement('span');status.className='badge '+(n.status==='success'?'ok':'bad');status.textContent=n.status;head.appendChild(status);card.appendChild(head);
    const kind=document.createElement('div');kind.className='node-kind';kind.textContent=n.type;card.appendChild(kind);
    const deps=document.createElement('div');deps.className='deps';deps.innerHTML='<strong>依赖：</strong>';deps.appendChild(document.createTextNode(n.depends_on.length?n.depends_on.join(' · '):'无（独立）'));card.appendChild(deps);
    const shown=n.outputs.filter(o=>o.is_image&&o.label!=='full_draft.png');section(card,'输出 · 草稿 / 渲染',shown,false);
    const refs=n.model_refs.length?n.model_refs:n.inputs.filter(i=>i.is_image);section(card,n.model_refs.length?'模型参考图 · A/B/…':'合成输入图 · 全部',refs,true);
    if(n.id==='scene_assembly'&&n.extra['placements.json']&&n.extra['placements.json'].photo_sources){const note=document.createElement('div');note.className='node-note';note.textContent='⚠ 金毛追球局部渲染被回退：使用金毛 partial ＋ 手工红球位置';card.appendChild(note)}
    card.onclick=()=>select(n.id);card.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n.id)}};col.appendChild(card)
  });cols.appendChild(col)});
  function imageGallery(parent,label,items){const h=document.createElement('h3');h.textContent=label;parent.appendChild(h);const images=items.filter(x=>x.is_image);if(!images.length){const p=document.createElement('p');p.className='empty';p.textContent='无图片';parent.appendChild(p);return}const grid=document.createElement('div');grid.className='img-grid';images.forEach((x,i)=>grid.appendChild(thumb(x,false,String.fromCharCode(65+i))));parent.appendChild(grid)}
  function addPre(parent,title,obj){const h=document.createElement('h3');h.textContent=title;parent.appendChild(h);const pre=document.createElement('pre');pre.textContent=typeof obj==='string'?obj:JSON.stringify(obj,null,2);parent.appendChild(pre)}
  function select(id){selected=id;const n=byId[id];const panel=document.getElementById('inspector');panel.replaceChildren();const title=document.createElement('h2');title.textContent=n.id;panel.appendChild(title);const badge=document.createElement('div');badge.className='meta-line';badge.textContent=n.type+' · Layer '+n.layer+' · '+n.status;panel.appendChild(badge);const dep=document.createElement('div');dep.className='meta-line';dep.textContent='depends_on: '+(n.depends_on.join(', ')||'[]');panel.appendChild(dep);
    if(n.reference_count!==null&&n.reference_count!==undefined){const count=document.createElement('div');count.className='meta-line';count.textContent='模型实际参考图数: '+n.reference_count;panel.appendChild(count)}
    if(n.id==='scene_assembly'&&n.extra['placements.json']&&n.extra['placements.json'].photo_sources){const warn=document.createElement('div');warn.className='notice';warn.textContent='调试注意：retriever_ball.partial 未进入照片合成；它退化为色块，改用 retriever.partial 与手工红球。详见 placements.json。';panel.appendChild(warn)}
    imageGallery(panel,'模型参考图（严格按 A/B/C 顺序）',n.model_refs);imageGallery(panel,'节点其他输入图（含确定性合成）',n.inputs);imageGallery(panel,'节点输出图',n.outputs);
    const textInputs=n.inputs.filter(x=>!x.is_image);if(textInputs.length){const p=document.createElement('p');p.className='path';p.textContent='非图像输入: '+textInputs.map(x=>x.path).join(' · ');panel.appendChild(p)}
    addPre(panel,'节点指令',n.instruction);if(Object.keys(n.extra).length)addPre(panel,'裁剪 / 放置 / 来源元数据',n.extra);addPre(panel,'执行结果',n.result);const path=document.createElement('p');path.className='path';path.textContent='节点记录: '+n.metadata_path;panel.appendChild(path);
    document.querySelectorAll('.node-card').forEach(el=>el.classList.toggle('selected',el.id==='node-'+id));highlight();
  }
  function related(id){const ids=new Set([id]);let changed=true;while(changed){changed=false;data.edges.forEach(e=>{if(ids.has(e.to)&&!ids.has(e.from)){ids.add(e.from);changed=true}})}return ids}
  function highlight(){const query=document.getElementById('search').value.trim().toLowerCase();const ancestor=selected?related(selected):null;document.querySelectorAll('.node-card').forEach(el=>{const id=el.id.slice(5);const match=!query||(id+' '+byId[id].type+' '+byId[id].instruction).toLowerCase().includes(query);el.classList.toggle('dim',!match||(ancestor&&!ancestor.has(id)))});document.querySelectorAll('.edge').forEach(path=>{const e=path.dataset;const on=ancestor&&ancestor.has(e.from)&&ancestor.has(e.to);path.classList.toggle('active',!!on);path.classList.toggle('dim',!!(ancestor&&!on))})}
  document.getElementById('search').addEventListener('input',highlight);document.getElementById('clear').onclick=()=>{selected=null;document.getElementById('search').value='';document.querySelectorAll('.node-card').forEach(el=>el.classList.remove('selected'));highlight()};
  function drawEdges(){const canvas=document.getElementById('canvas');const svg=document.getElementById('edges');svg.replaceChildren();const rect=canvas.getBoundingClientRect();svg.setAttribute('viewBox','0 0 '+canvas.scrollWidth+' '+canvas.scrollHeight);svg.setAttribute('width',canvas.scrollWidth);svg.setAttribute('height',canvas.scrollHeight);data.edges.forEach(edge=>{const from=document.getElementById('node-'+edge.from);const to=document.getElementById('node-'+edge.to);if(!from||!to)return;const a=from.getBoundingClientRect(),b=to.getBoundingClientRect();const x1=a.right-rect.left,y1=a.top+a.height/2-rect.top,x2=b.left-rect.left,y2=b.top+b.height/2-rect.top;const mid=(x1+x2)/2;const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d','M '+x1+' '+y1+' C '+mid+' '+y1+', '+mid+' '+y2+', '+x2+' '+y2);p.setAttribute('class','edge');p.dataset.from=edge.from;p.dataset.to=edge.to;const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=edge.from+' → '+edge.to+' · '+(edge.artifacts.join(', ')||'dependency');p.appendChild(t);svg.appendChild(p)});highlight()}
  window.addEventListener('resize',drawEdges);window.addEventListener('load',drawEdges);requestAnimationFrame(drawEdges);select('scene_assembly');
})();
</script>
</body></html>
"""


def write_html(run_dir: Path = DEFAULT_RUN, output: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    output = output or run_dir / "debug.html"
    payload = build_payload(run_dir)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_html(args.run_dir, args.output))


if __name__ == "__main__":
    main()
