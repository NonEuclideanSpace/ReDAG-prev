import ast
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from image_nodes import scale_image, compositor_config, composite, grow_mask
from workflow import SOURCE, without_loras, widget_inputs, load_workflow, conditioning_fingerprint


def test_lora_removal_keeps_connected_model_chain():
    original = json.loads(SOURCE.read_text())
    workflow = without_loras(original)
    assert len(workflow['nodes']) == len(original['nodes']) - 2
    assert not any(n['type'].startswith('LoraLoader') for n in workflow['nodes'])
    model_input = next(l for l in workflow['links'] if l[3:5] == [68, 0])
    assert model_input[1:3] == [104, 0]
    node_ids = {n['id'] for n in workflow['nodes']}
    assert all(l[1] in node_ids and l[3] in node_ids for l in workflow['links'])
    assert any(n['id'] == 110 for n in original['nodes'])


def test_sampler_widget_mapping_drops_frontend_seed_control():
    node = next(n for n in load_workflow()['nodes'] if n['id'] == 23)
    kwargs = widget_inputs(node)
    assert kwargs['steps'] == 8
    assert kwargs['cfg'] == 1
    assert kwargs['sampler_name'] == 'euler_ancestral'
    assert kwargs['start_at_step'] == 0
    assert kwargs['end_at_step'] == 30


def test_scaler_matches_upstream_fill_and_rounding():
    # Execute the actual upstream helpers in a tiny namespace to compare the
    # resampling result, including its integer truncation before rounding.
    tree = ast.parse((ROOT / 'provenance/imagefunc.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('fit_resize_image', 'num_round_up_to_multiple')]
    namespace = {'Image': Image}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'upstream', 'exec'), namespace)
    image = Image.fromarray(np.random.default_rng(1).integers(0,255,(331,717,3),dtype=np.uint8))
    width = 2000
    height = namespace['num_round_up_to_multiple'](int(width / (717/331)), 8)
    expected = namespace['fit_resize_image'](image, width, height, 'fill', Image.Resampling.LANCZOS)
    result = scale_image(image,'original',1,1,'fill','lanczos','8','longest',2000,'#000000')[0]
    assert result.size == (2000, 928)
    np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))


def test_foreground_mask_and_overlapping_layers():
    background = Image.new('RGB',(10,10),'blue')
    foreground = Image.new('RGB',(4,4),'red')
    mask = Image.new('L',(4,4),0)
    mask.putpixel((1,1),255)
    config = compositor_config(10,10,0,False,False,False,'',image1=background,image2=foreground,mask2=mask)[0]
    transform = {'left':0,'top':0,'scaleX':1,'scaleY':1,'angle':0}
    t2 = dict(transform,left=3,top=2)
    result = composite(config,json.dumps({'transforms':[transform,t2]}))[1]
    assert result.getpixel((4,3)) == (255,0,0)
    assert result.getpixel((3,2)) == (0,0,255)
    assert grow_mask(mask,0,True)[0].tobytes() == mask.tobytes()


def test_generated_assets_have_nonempty_transparency():
    run = ROOT / 'runs/001-autumn-lounge'
    files = sorted((run / 'transparent_assets').glob('*.png'))
    assert len(files) == 5
    for path in files:
        image = Image.open(path)
        assert image.mode == 'RGBA'
        alpha = np.asarray(image.getchannel('A'))
        assert 0.1 < (alpha > 0).mean() < 0.5
        assert alpha[0,0] == 0


def test_conditioning_cache_changes_with_prompt_and_mask(tmp_path):
    workflow = load_workflow()
    (tmp_path/'inputs').mkdir()
    (tmp_path/'nodes/main').mkdir(parents=True)
    Image.new('RGB',(2,2),'white').save(tmp_path/'inputs/background.png')
    for node_id in (122,132,139,147,155):
        for slot in (0,1):
            Image.new('L',(2,2),255).save(tmp_path/f'nodes/main/{node_id}_{slot}.png')
    first = conditioning_fingerprint(workflow,tmp_path)
    prompt_node = next(n for n in workflow['nodes'] if n['id']==20)
    original = prompt_node['widgets_values'][0]
    prompt_node['widgets_values'][0] = 'Changed pose'
    assert conditioning_fingerprint(workflow,tmp_path) != first
    prompt_node['widgets_values'][0] = original
    Image.new('L',(2,2),0).save(tmp_path/'nodes/main/122_1.png')
    assert conditioning_fingerprint(workflow,tmp_path) != first
