# pipe.json 无 LoRA 复现实验

所有新代码、节点输出、失败尝试、运行日志和最终图片放在本目录；没有修改原始 `pipe.json` 或已有实验。两份 LoRA 的节点 **110、53 已删除并重连**，不会加载 Lightning 或 Put-it-here。

素材由 `/home/hdd4/lpz/Qwen-Image-2.1` 生成，使用 `/home/hdd/lpz/model/sam2.1` 的真实模型产生 alpha。最终原生工作流使用 Qwen-Image-Edit-2511 的 Q4_K_M、原工作流的文本编码器及 VAE，执行 ComfyUI 的真实采样和解码代码。

## 结果文件

- `runs/001-autumn-lounge/inputs/`：新生成的背景和五个人物。
- `runs/001-autumn-lounge/transparent_assets/`：五张 RGBA 人物素材，保留源 RGB，应用 SAM2.1 遮罩。
- `runs/001-autumn-lounge/composite_original_layout.png`：按原始 Fabric transforms 合成的拼贴图。
- `runs/001-autumn-lounge/composite_new_scene.png`：调整座位位置、统一人物缩放后的拼贴图。
- `runs/001-autumn-lounge/final_qwen21.png`：Qwen-Image 2.1 对新拼贴图的融合结果，1536×704、30 步、CFG 1。
- `runs/001-autumn-lounge/final_no_lora_8step.png`：原采样设置的无 LoRA 对比结果，8 步、CFG 1。
- `runs/001-autumn-lounge/final_no_lora_40step.png`：原生工作流的无 LoRA 40 步、CFG 4 对比结果。
- `runs/001-autumn-lounge/final_scene.png`：最终选用原生工作流的无 LoRA **8 步、CFG 1** 结果，2000×920；40 步、CFG 4 在此场景出现灰暗、颗粒和纹理伪影，因此未选为最终图。
- `runs/001-autumn-lounge/nodes/`、`execution.jsonl`：各节点及子图的输出和执行记录。
- `runs/001-autumn-lounge/comparison.jpg`、`report.json`：拼贴与三个融合结果的对比、节点覆盖、参数和权重校验记录。
- `runs/failed/`：框提示选中背景、仅按 SAM 预测分数选错第四个人物等失败尝试的合成图与遮罩记录。

`final_qwen21.png` 是独立的模型对比实验；它没有冒充原工作流的 KSamplerAdvanced 输出。原生结果使用新增座位布局 `layout_new_scene.json`；原布局拼贴图也完整保留。

## 节点如何复现

| 节点 | 实现及核对源码 |
| --- | --- |
| LoadImage、GetImageSize、SaveImage | Pillow 文件与尺寸操作；遵循 LoadImage 的 RGB / 反向 alpha MASK 约定 |
| ImageScaleByAspectRatio V2 | [LayerStyle 源码](https://github.com/chflame163/ComfyUI_LayerStyle/blob/main/py/image_scale_by_aspect_ratio_v2.py)：先取整比例尺寸，再向上取 8 的倍数，fill + Lanczos |
| RMBG | 核对 [RMBG 源码](https://github.com/1038lab/ComfyUI-RMBG/blob/main/py/AILab_RMBG.py)，按用户指定改为本地 SAM2.1；输出 RGBA、白色保留前景的 MASK、原 RGB |
| GrowMask | [ComfyUI 源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mask.py)：本文件 expand=0；也实现了相同十字/方形膨胀、腐蚀 |
| Mask To Image (mtb) | [MTB 源码](https://github.com/melMass/comfy_mtb/blob/main/nodes/image_processing.py)：白色前景、黑色背景、不反转 |
| ImageBlendAdvance | [LayerStyle 源码](https://github.com/chflame163/ComfyUI_LayerStyle/blob/main/py/image_blend_advance.py)：实现序列化的 normal、100% 不透明度、居中、无变换设置 |
| CompositorConfig3、Compositor3 | [合成器源码](https://github.com/erosDiffusion/ComfyUI-enricos-nodes/blob/master/Compositor3.py)：配置应用前景 alpha；离线渲染保存的 Fabric transforms |
| UnetLoaderGGUF | 直接调用 [city96 源码](https://github.com/city96/ComfyUI-GGUF/blob/main/nodes.py)，保留原 GGUF 模型及量化方式 |
| CLIPLoader、VAELoader、VAEEncode、VAEDecode、ConditioningZeroOut、KSamplerAdvanced | 直接调用 [ComfyUI nodes.py](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py) |
| TextEncodeQwenImageEditPlus | 直接调用 [Qwen 节点源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py)，保留视觉编码与缩放参考 latent |
| ReferenceLatent | 直接调用 [编辑节点源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_edit_model.py)，追加全尺寸 VAE latent，未去掉第二份参考 |
| ModelSamplingAuraFlow、CFGNorm | 直接调用 ComfyUI 的模型采样和 CFG 节点，保留 shift=3.1、strength=1、pre_cfg=False |
| 五个 UUID 子图 | 从 JSON 的 definitions.subgraphs 逐条解析边、输入边界及输出边界；实际执行各图全部六个内部节点 |
| PreviewImage、MarkdownNote | PreviewImage 保存图像用于审查；MarkdownNote 是无计算依赖的说明文字 |

原 `Compositor3` 后端读取浏览器上传的临时 PNG，文件名 `compositor/1789924689666.png [temp]` 不包含在输入中。本实验补上离线画布渲染，按层序、位置、缩放、翻转、MASK 合成；浮点位置取到最近像素，可能与浏览器产生亚像素差异。当前图没有旋转、倾斜、padding 或高度归一化；未实现的额外设置会报错。

SAM2 是点/框提示分割器。生成提示明确要求白色摄影背景。实测框提示及最高预测 IoU 可能选中背景，因此使用躯干正点、两侧负点，并用已知白背景条件对三张候选 SAM 遮罩排序。最终 alpha 是所选 SAM 遮罩，没有用颜色阈值替代分割模型；候选分数、选择和前景占比都记在 `sam2_masks.json`。

## 复现命令

当前实验已准备独立 `.venv`，基于已有 `uno_env` 的 PyTorch 2.12.0+cu130、diffusers 0.41.0.dev0、transformers 5.17.0。额外依赖记录于 `requirements-extra.txt`。以下从本实验目录执行：

```bash
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python src/generate_assets.py
.venv/bin/python src/compose_scene.py
.venv/bin/python src/run_pipeline.py --compare-no-lora \
  --layout runs/001-autumn-lounge/layout_new_scene.json --output final_no_lora_40step.png
.venv/bin/python src/report_results.py --select final_no_lora_8step.png
.venv/bin/python -m pytest -q tests
```

生成脚本默认跳过已存在素材。原生推理可通过 `--steps`、`--cfg`、`--layout`、`--output` 独立运行；不传 steps/cfg/layout 时使用原 JSON 的采样与布局设置。`--prepare-only` 可以先运行文本编码和 VAE，再缓存条件，避免文本编码器与采样器同时占显存。

新机器需先克隆 `vendor/ComfyUI` 和 `vendor/ComfyUI-GGUF`，按照 `provenance/manifest.json` 固定版本；用已有 GPU Python 创建 `--system-site-packages` 的 `.venv` 并安装额外依赖。`src/download_models.py` 下载原工作流的三份基础权重，目标为 `models/`。本机 `models` 是指向 `/home/hdd4/lpz/DraftDAG-experiments/02-comfy-pipeline/models` 的链接，以避开主分区空间限制；没有下载两份 LoRA。

## 参数与验证

素材随机种子为 202610110–202610115；Qwen 2.1 融合种子为 202610118。原生采样种子固定为 JSON 中的 603311763670135，忽略前端的 `randomize` 控件。原设置使用 8 步、CFG 1、euler_ancestral/simple、start=0、end=30、关闭 leftover noise。无 Lightning 的另一个实验只将 steps/end 改为 40、CFG 改为 4，并使用新座位布局，其余模型、条件和采样路径保留。

57 个计算节点（包含全部 30 个子图内部节点）已实际执行。6 项验证通过，覆盖两份 LoRA 删除后的连线完整性、seed 控件解析、缩放输出与真实上游 helper 的逐像素比较、前景遮罩方向和图层覆盖、五张实际生成素材的有效透明度，以及条件缓存对提示词和遮罩变化的失效检查。源码快照、版本、输入 workflow 的 SHA256、权重来源及 SHA256 均保存在 `provenance/`。三份基础模型的本地 SHA256 与 Hugging Face LFS 公布值一致。
