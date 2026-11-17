# 从一幅古画让苏东坡出道：用 MiniMax H3 全家桶在本地做音乐 MV

这一页是视频《古人说唱》下集的资料：模型下载地址、放进哪个文件夹、三个 ComfyUI 工作流的用法和参数、我们用的提示词。这几套 ComfyUI 工作流我们都跑通了，下载拖进 ComfyUI 就能用。这次拿苏东坡做例子，换成任何角色，用同一套流程都能做自己的音乐 MV。

- YouTube 频道：https://www.youtube.com/@builder2ai
- 视频里的人物、声音和画面都是 AI 生成的，王渔鱼是 AI 数字人主播。

## 三步

| 步骤 | 模型 | 做什么 | 工作流 |
|---|---|---|---|
| 1 | MiniMax H3-Image | 参考图 + 文字生图：角色的定妆照，每个镜头的第一帧 | [h3image_still.json](workflows/h3image_still.json) |
| 2 | MiniMax H3 / HyperFlow H3 Flash | 图 + 音频生视频，音画同步：人物跟着歌声开口唱 | [h3_singing.json](workflows/h3_singing.json) / [h3flash_singing.json](workflows/h3flash_singing.json) |
| 3 | Claude Code | 剪辑成 MV：原歌铺底，唱歌的镜头放回对应的歌词位置 | 见下面「剪辑」 |

## 用什么电脑

- **家用电脑**：用 H3 Flash。官方说 24G、32G 显卡能跑，16G 官方还没测。不用额外花钱，想改多少遍都行。
- **设备好一些**：也可以用 H3 完整版。我们用的是一台 NVIDIA DGX Spark（128G 内存，显卡和 CPU 共用），1344×768、10 秒左右的镜头一条约 12 分钟，内存最多用到 100G 左右。

## 安装

1. **ComfyUI**：到官网 https://www.comfy.org/download 下载安装，已经装过的更新到最新版。H3 需要 ComfyUI 0.43.0 以上，我们用的是 0.45.2。
2. **节点**：H3 和 H3-Image 用到的节点，新版 ComfyUI 都自带，不用另外装插件。
3. **Flash 的节点包**（只用完整版可以跳过）：在 ComfyUI 的节点管理器里搜 `HyperFlow`，安装官方的 **ComfyUI-HyperFlow-Flash**。
4. **模型文件**：都在 Hugging Face 上，下载后放进 `ComfyUI/models/` 下对应的文件夹。

| 文件 | 大小 | 放进 | 用途 |
|---|---|---|---|
| [minimax_h3_image_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) | 21.0G | diffusion_models | H3-Image 出图 |
| [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | 21.0G | diffusion_models | H3 视频，完整版 |
| [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) | 2.0G | loras | H3 加速 LoRA，20 步变 4 步 |
| [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15.7G | text_encoders | 读懂提示词，两个模型共用 |
| [minimax_h3_image_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors) | 1.2G | vae | H3-Image 的图片解码器 |
| [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 5.2G | vae | H3 视频的画面解码器 |
| [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | 0.6G | vae | H3 视频的声音解码器 |
| [minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash/resolve/main/ref2va_int8_convrot_comfyui/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors)（可选） | 16.4G | diffusion_models | H3 Flash，代替 H3 视频完整版 |

完整版七个文件合计约 67G。只用 Flash 的话，第二、三行换成 Flash 一个文件，合计约 60G。

模型页：[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) · [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) · [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) · [HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)

## 三个工作流

把 json 文件拖进 ComfyUI 就能用。每个工作流左上角有一段中文说明，节点标题也写了中文。参数都是我们实际用的，种子固定，照着跑能得到一样的结果。

**1. [h3image_still.json](workflows/h3image_still.json)：参考图 + 文字生图（MiniMax H3-Image）**

- 左边五个「加载图片」节点，按连线顺序就是提示词里的 `<Picture 1>` 到 `<Picture 5>`：正面、偏左、偏右三张脸，古画里的帽子，古画里的衣服。参考图最多能放 16 张。
- 提示词里写清楚每张图管什么，比如这张管脸、那张管衣服。脸最好多给几个角度，转过头的镜头也认得出是同一个人。
- 参数：2752×1536，6 步，CFG 1，euler / simple。在我们的机器上一张一分钟左右，每张多跑一两次，挑脸最像的那张。
- 舞台和外景每个镜头的第一帧也用它先画好，视频就从这张图开始动。

**2. [h3_singing.json](workflows/h3_singing.json)：图 + 音频生视频（MiniMax H3 完整版）**

- `Picture 1`：脸部特写，从定妆照上裁。`Picture 2`：这个镜头的第一帧。`Audio 1`：这几句的歌声。
- 歌声按这条镜头要唱的几句裁出来，一条十秒左右。帧数要是 17 的倍数加 5，243 帧约 10 秒（24 fps）。
- 参数：1344×768，加速 LoRA 打开，4 步，res_multistep / simple。
- 外景不唱歌：删掉 `Audio 1`，提示词里写他不说话。

**3. [h3flash_singing.json](workflows/h3flash_singing.json)：图 + 音频生视频（HyperFlow H3 Flash）**

- 输入和完整版一样。需要先装 ComfyUI-HyperFlow-Flash 节点包。
- 参数：6 步，CFG 1，euler，不挂加速 LoRA。

工作流里的输入文件名（`face_front.png`、`singer_face.png`、`first_frame.png`、`song_slice.wav` 等）只是占位：把你自己的图和歌声放进 `ComfyUI/input`，在节点里重新选上就行。

## 提示词

出图和唱歌的提示词模板在 [prompts.md](prompts.md)，用的是我们 MV 里实际用的原句，换上你自己的角色、歌词和动作就能用。

## 剪辑：交给 Claude Code

1. 列一张镜头表：每条镜头从歌的第几秒开始，配哪几句歌词。
2. 写个小脚本：把原歌铺在底下，H3 生成的声音全部关掉，唱歌的镜头放回生成时用的那几句的位置，嘴就对得上；外景放慢一点，镜头之间加个叠化。
3. 几分钟就能出一版。看完用大白话告诉它哪里要改，比如「这条重新生成一次」「这段外景再慢一点」，它改完马上再出一版。

## 许可证

H3-Image 可以商用，具体条件见模型页的许可证。H3 和 H3 Flash 用的是 MiniMax H3 的社区许可证，也请先看模型页。上传 AI 生成的视频时，记得按平台要求标注 AI 生成内容。
