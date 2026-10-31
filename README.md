# 王渔鱼是怎么做出来的：资料页

王渔鱼是 Builder² 的 AI 数字人主播。她的脸、声音和说话视频，全部用开源模型在一台电脑上本地生成，人通过 Claude Code 指挥整个过程。这一页放视频里提到的工具、模型、下载地址和硬件，还有我们打包好的 Claude Code skill。

- 视频：YouTube 频道 https://www.youtube.com/@builder2ai ，第 1 期 https://www.youtube.com/watch?v=jbGZmQ7aktI
- 王渔鱼是 AI 数字人，不是真人。视频里的人物、声音和画面都是 AI 生成的。

## 数字人制作 skill

skill 就是写给 Claude Code 的一份说明书，连着脚本和工作流一起。里面写的就是视频里那条路：用文字写脸、用文字设计声音、让它开口说话，每一步该量什么、量到多少算过关。里面没有任何真人的脸和声音，你的数字人要从文字开始写。

```
skills/digital-human/
├── SKILL.md            给 Claude Code 的说明书：步骤、检查标准、怎么跟人配合
├── reference/models.md 模型文件、插件、Python 环境、硬件
├── scripts/            改图、量脸、造声音、配音质检、做底稿、生成说话视频
└── workflows/          ComfyUI 画布版：形象编辑、说话视频
```

**安装**

1. 下载这个仓库（右上角 Code → Download ZIP，或者 `git clone`）。
2. 把 `skills/digital-human` 整个文件夹复制到 Claude Code 的 skills 文件夹：
   - 只给自己用：`~/.claude/skills/`（Windows 是 `C:\Users\你的用户名\.claude\skills\`）
   - 只给某个项目用：项目里的 `.claude/skills/`
3. 新开一个 Claude Code 会话，它自己就认得。

**前提**：电脑上已经装好 ComfyUI 和这些模型，需要英伟达显卡，显存越大越好。缺什么，skill 第一步会帮你查出来，下载地址在 [reference/models.md](skills/digital-human/reference/models.md)。

**用法**：直接跟 Claude Code 说你想要什么，比如「我想做一个三十岁左右、说话温和的男主播」。它会先查模型齐不齐，再一步一步做，每一步拼好对比图让你挑。

我们只在 NVIDIA DGX Spark 上跑通过，别的电脑没有实测。

## 用到的工具和模型

**Claude Code**（Anthropic 的 AI 编程助手）
- https://www.claude.com/product/claude-code
- VS Code 插件：https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code
- 文档：https://code.claude.com/docs/en/overview

**ComfyUI**（开源的节点式 AI 工作台）
- 下载：https://www.comfy.org/download
- GitHub：https://github.com/Comfy-Org/ComfyUI
- 插件管理器：https://github.com/Comfy-Org/ComfyUI-Manager
- WanVideoWrapper 插件（Kijai）：https://github.com/kijai/ComfyUI-WanVideoWrapper

**做脸：Qwen-Image-Edit-2511**（阿里通义千问）
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- 看图的模型 Qwen2.5-VL：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 4 步提速 LoRA（lightx2v）：https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- 人脸角度和相似度检查 InsightFace：https://github.com/deepinsight/insightface （代码是 MIT；预训练模型官方只许非商业研究使用，商用要另外授权）
- 2026 年 9 月新出的 Qwen-Image 2.1 换成了更新的 Qwen3-VL，许可证也改成了研究许可。我们用的是 Apache 2.0、可以商用的 2511。

**声音：Qwen3-TTS**（阿里通义千问）
- GitHub：https://github.com/QwenLM/Qwen3-TTS
- 在线试用：https://huggingface.co/spaces/Qwen/Qwen3-TTS
- 声音设计模型：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- 克隆模型：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- 声纹检查 CAM++（3D-Speaker）：https://github.com/modelscope/3D-Speaker
- 听写检查 Whisper：https://github.com/openai/whisper

**开口说话：InfiniteTalk（美团）+ Wan2.1（阿里通义万相）**
- InfiniteTalk：https://github.com/MeiGen-AI/InfiniteTalk
- Wan2.1：https://github.com/Wan-Video/Wan2.1
- 提速 LoRA：https://huggingface.co/lightx2v
- 中文语音特征 wav2vec2（腾讯中文版）：https://huggingface.co/TencentGameMate/chinese-wav2vec2-base

每个文件的准确文件名、大小、放哪个文件夹、从哪里下，见 [reference/models.md](skills/digital-human/reference/models.md)。模型文件做脸一套约 31 GB，说话一套约 36 GB；跑的时候要放进显卡能用的内存里。

## 硬件

**我们用的**：NVIDIA DGX Spark 官方版。GB10 芯片，20 核 CPU，128 GB 内存（CPU 和显卡共用），4 TB 硬盘，Ubuntu 系统。
https://www.nvidia.com/en-us/products/workstations/dgx-spark/

**其他能用的电脑**（我们没有实测，按官方和社区资料整理）：
- 同芯片的 GB10 机器，同样 128 GB 内存，流程可以照搬。华硕 Ascent GX10：https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/ 。戴尔 Pro Max with GB10、联想 ThinkStation PGX、惠普 ZGX Nano G1n 也是同一颗芯片，搜型号即可。
- 英伟达显卡台式机，系统内存 64 GB 以上。兼容最好，算得比 Spark 快；说话视频要开 WanVideoWrapper 的 block swap，或者用压缩版模型。
  - RTX 5090（32 GB）：https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
  - RTX 4090（24 GB）：https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- Mac：Mac Studio M5 Max（最高 128 GB）、M5 Ultra（96 GB 起）。配音能跑，做脸能跑但慢，说话视频目前没找到在 Mac 上跑通的可靠记录。
  - https://www.apple.com/mac-studio/
  - Mac 上跑 Qwen3-TTS 可以看社区项目 mlx-audio：https://github.com/Blaizzy/mlx-audio
- 手上没有合适的机器：可以先按小时租云端显卡试。

显存不够时，换压缩得更小的模型版本（fp8、GGUF），或者让 ComfyUI 把一部分模型放到内存里轮着用。能跑，就是更慢。

## 花多少钱

模型都在本地跑，生成多少张图、多少段视频都不按次付费。花钱的是电脑、电费和 Claude Code 的订阅。

## 许可证

这个仓库里的脚本、工作流和文档用 MIT 许可证，见 [LICENSE](LICENSE)。用到的每个模型有自己的许可证，用之前看清楚：比如 InsightFace 的预训练人脸模型只许非商业研究使用，商用要另外授权。

网址最后核对日期：2026 年 10 月 30 日。
