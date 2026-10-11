"""Execute pipe.json including its five embedded subgraphs, without the GUI."""
import copy
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageOps

from image_nodes import (scale_image, grow_mask, mask_to_image, blend_image,
                         compositor_config, composite)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / 'pipe.json'
INPUT_NAMES = {2: 'background', 34: 'woman_red', 121: 'man_blue',
               140: 'woman_green', 148: 'man_ochre', 156: 'woman_lilac'}


def without_loras(workflow):
    """Remove both nodes and wire their model inputs through to their consumers."""
    workflow = copy.deepcopy(workflow)
    for node_id in (110, 53):
        incoming = next(link for link in workflow['links'] if link[3] == node_id and link[4] == 0)
        for link in workflow['links']:
            if link[1] == node_id:
                link[1:3] = incoming[1:3]
        workflow['nodes'] = [n for n in workflow['nodes'] if n['id'] != node_id]
        workflow['links'] = [l for l in workflow['links'] if l[3] != node_id]
    # Rebuild serialized socket metadata as well as the links table.
    links = {l[0]: l for l in workflow['links']}
    for node in workflow['nodes']:
        for socket in node.get('inputs', []):
            if socket.get('link') not in links:
                socket['link'] = None
        for i, socket in enumerate(node.get('outputs', [])):
            socket['links'] = [l[0] for l in links.values() if l[1:3] == [node['id'], i]]
    return workflow


def widget_inputs(node):
    values = list(node.get('widgets_values', []))
    names = [s['name'] for s in node.get('inputs', []) if 'widget' in s]
    # seed control_after_generate and LoadImage's upload control are frontend-only.
    if node['type'] == 'KSamplerAdvanced':
        values.pop(2)
    return dict(zip(names, values))


class Workflow:
    def __init__(self, workflow, run, rmbg=None, model_backend=None):
        self.workflow = workflow
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=True)
        self.subgraphs = {g['id']: g for g in workflow['definitions']['subgraphs']}
        self.rmbg = rmbg
        self.model_backend = model_backend
        self.trace = []
        self.cache = {}
        self.active = set()

    def execute(self, node_id=184):
        return self._evaluate(self.workflow, node_id, 'main', {})

    def _evaluate(self, graph, node_id, scope, external):
        key = (scope, node_id)
        if key in self.cache:
            return self.cache[key]
        if key in self.active:
            raise ValueError(f'Cycle at {key}')
        self.active.add(key)
        node = next(n for n in graph['nodes'] if n['id'] == node_id)
        links = {}
        for link in graph['links']:
            if isinstance(link, dict):
                links[link['id']] = (link['origin_id'], link['origin_slot'])
            else:
                links[link[0]] = (link[1], link[2])
        kwargs = widget_inputs(node)
        for socket in node.get('inputs', []):
            link_id = socket.get('link')
            if link_id is not None:
                origin, slot = links[link_id]
                kwargs[socket['name']] = external[slot] if origin == -10 else self._evaluate(graph, origin, scope, external)[slot]
        kind = node['type']
        started = time.time()
        if kind in self.subgraphs:
            subgraph = self.subgraphs[kind]
            # The exposed integer is unconnected in pipe.json; each inner widget
            # still supplies 1024. Preserve that widget default for the boundary.
            boundary = {i: kwargs.get(s['name'], 1024 if s['name'] == 'scale_to_length' else None)
                        for i, s in enumerate(subgraph['inputs'])}
            results = []
            for output in subgraph['outputs']:
                link = next(l for l in subgraph['links'] if l['id'] == output['linkIds'][0])
                result = self._evaluate(subgraph, link['origin_id'], f'{scope}.{node_id}', boundary)
                results.append(result[link['origin_slot']])
            # PreviewImage is a UI sink; execute it so all inner nodes are audited.
            for sink in subgraph['nodes']:
                if sink['type'] == 'PreviewImage':
                    self._evaluate(subgraph, sink['id'], f'{scope}.{node_id}', boundary)
            result = tuple(results)
        elif kind == 'LoadImage':
            path = self.run / 'inputs' / (INPUT_NAMES[node_id] + '.png')
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).copy()
            alpha = image.getchannel('A') if image.mode == 'RGBA' else Image.new('L', image.size, 255)
            result = image.convert('RGB'), ImageOps.invert(alpha)
        elif kind == 'LayerUtility: ImageScaleByAspectRatio V2':
            result = scale_image(**kwargs)
        elif kind == 'RMBG':
            if self.rmbg is None:
                raise RuntimeError('Supply a SAM2BackgroundRemoval backend')
            result = self.rmbg(**kwargs)
        elif kind == 'GrowMask':
            result = grow_mask(**kwargs)
        elif kind == 'Mask To Image (mtb)':
            result = mask_to_image(**kwargs)
        elif kind == 'LayerUtility: ImageBlendAdvance':
            result = blend_image(**kwargs)
        elif kind == 'PreviewImage':
            result = (kwargs['images'],)
        elif kind == 'GetImageSize':
            result = kwargs['image'].size
        elif kind == 'CompositorConfig3':
            result = compositor_config(**kwargs)
        elif kind == 'Compositor3':
            result = composite(**kwargs)
        elif kind == 'SaveImage':
            result = (kwargs['images'],)
        elif self.model_backend is not None:
            result = self.model_backend.execute(kind, kwargs)
        else:
            raise ValueError(f'No implementation for node {node_id}: {kind}')
        self.cache[key] = result
        self.active.remove(key)
        artifacts = []
        directory = self.run / 'nodes' / scope
        directory.mkdir(parents=True, exist_ok=True)
        for slot, value in enumerate(result):
            if isinstance(value, Image.Image):
                path = directory / f'{node_id}_{slot}.png'
                value.save(path)
                artifacts.append(str(path.relative_to(self.run)))
        record = dict(scope=scope, id=node_id, type=kind, artifacts=artifacts,
                      elapsed_seconds=round(time.time()-started, 3))
        self.trace.append(record)
        with (self.run / 'execution.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'EXECUTED {scope}/{node_id}: {kind}', flush=True)
        return result


def load_workflow():
    return without_loras(json.loads(SOURCE.read_text()))


def conditioning_fingerprint(workflow, run):
    """Invalidate the expensive conditioning cache when its actual inputs change."""
    digest = hashlib.sha256()
    nodes = {n['id']: n for n in workflow['nodes']}
    for node_id in (13,14,20,184):
        widgets = nodes[node_id]['widgets_values']
        if node_id == 184:
            widgets = json.loads(widgets[0])
        digest.update(json.dumps(widgets, sort_keys=True, ensure_ascii=False).encode())
    for path in [Path(run)/'inputs/background.png',
                 *(Path(run)/f'nodes/main/{node_id}_{slot}.png'
                   for node_id in (122,132,139,147,155) for slot in (0,1))]:
        digest.update(path.read_bytes())
    return digest.hexdigest()
