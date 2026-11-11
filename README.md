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

## AI 作曲 skill

> 这个 skill 在一首测试歌上从头到尾跑通过一次，样歌在 [examples/ai-song-dushu](examples/ai-song-dushu)。试跑里发现的问题已经改进 skill，改过的脚本大多重新验证过，要用显卡的两处（整版重新渲染、只分离一段）还没有重跑。目前只试过这一首歌，步骤和脚本以后可能还会改。

视频里苏东坡那首歌是我们一点点试出来的，这个 skill 把那些步骤和踩过的坑整理成一条固定的路：先扒一首你喜欢的歌的结构和风格，再填词、写风格描述，用开源模型作曲，挑出最好的那版，再补前奏、念白和结尾。每一步该量什么、量到多少算过关，都写在里面。

```
skills/ai-song/
├── SKILL.md            给 Claude Code 的说明书：扒曲、填词、风格描述、作曲、挑版、查漏补缺、交付检查
├── reference/          模型和许可证、文件格式、踩过的坑、你说一句话它该去量什么
└── scripts/            分离人声、听写、量速度和调、给乐器打标签、出歌、量每一版、拼接、打分、结尾渐慢
```

**试听**：[examples/ai-song-dushu](examples/ai-song-dushu) 是一次完整试跑。新开的 Claude Code 会话只拿着这个 skill，照一首西班牙风格的歌，写了李清照「赌书泼茶」的一段主歌和副歌，用 YuE2 作曲，再改字、补前奏和结尾。那里有 40 秒的样歌、歌词和出处、每一步做了什么，还有生成用的参数。

**安装**：和数字人 skill 一样，把 `skills/ai-song` 整个文件夹复制到 Claude Code 的 skills 文件夹，新开一个会话。

**前提**：英伟达显卡；一个能跑 YuE2 的 ComfyUI（要较新的版本，它自带 YuE2 的节点）；ACE-Step 1.5；几个分离人声和听写用的 Python 包；带 rubberband 滤镜的 ffmpeg。这些工具要装在不同的 Python 环境里。缺什么，skill 第一步会帮你查出来，下载地址和大小在 [reference/models.md](skills/ai-song/reference/models.md)。

**用法**：把你喜欢的那首歌的音频给 Claude Code，跟它说「照这首歌的风格，给杜甫写一首 40 秒的歌」。它会先把那首歌扒一遍，给你看一页风格卡，问你几个只有耳朵能答的问题（什么风格、什么乐器、男声还是女声），再和你一起写词。词你确认了，它才开始作曲。

两条规矩：
- **参考歌只用来分析。** 它的音频、分出来的人声和伴奏、听写稿，只在你自己的电脑上用来量速度、调、段落和句子，不交给任何生成模型，也不拿去翻唱。新歌只学它的结构和风格。
- **Claude Code 听不见声音。** 乐器和风格叫什么，只有两个来源：打标签的模型（AST、CLAP）给的猜测，和你的耳朵。模型会把吉他听成曼陀林，所以它会把猜测摆出来问你，你答了它才往下写。

我们只在 NVIDIA DGX Spark 上跑通过。

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

**作曲**
- YuE2（M-A-P 开源研究社区）：给歌词和一段风格描述，人声和伴奏一起出来。项目：https://github.com/multimodal-art-projection/YuE ，模型：https://huggingface.co/m-a-p/YuE2-3B 。我们在 ComfyUI 里跑，用的是 ComfyUI 官方打包版里的 `yue2_3b_int8_convrot.safetensors`：https://huggingface.co/Comfy-Org/YuE2
- ACE-Step 1.5（ACE Studio 和阶跃星辰）：字咬得更清楚，也能在一首现成的歌前后补几小节，我们歌里的前奏就是它补的。GitHub：https://github.com/ace-step/ACE-Step-1.5 ，模型：https://huggingface.co/ACE-Step/Ace-Step1.5 、https://huggingface.co/ACE-Step/acestep-v15-xl-sft
- 四川话念白：沿用上面的 Qwen3-TTS，用 CustomVoice 模型自带的成都男声 Eric：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- 分离人声和伴奏：python-audio-separator（https://github.com/nomadkaraoke/python-audio-separator ），用其中的 BS-RoFormer 模型 `model_bs_roformer_ep_317_sdr_12.9755`
- 分成人声、鼓、贝斯、其他四层：Demucs（Meta）https://github.com/facebookresearch/demucs 。原仓库已停止维护，作者的分支在 https://github.com/adefossez/demucs
- 听写和给咬字打分：沿用上面的 Whisper，用 faster-whisper 跑 large-v3-turbo：https://github.com/SYSTRAN/faster-whisper 、https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
- 给乐器和风格打标签：AST https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 ，CLAP https://huggingface.co/laion/clap-htsat-unfused

作曲这套每个文件的文件名、大小、放哪儿，见 [reference/models.md](skills/ai-song/reference/models.md)。YuE2 约 4 GB；ACE-Step 1.5 一套约 28 GB（其中 XL 模型约 19 GB，4B 的语言模型约 8 GB）；分离、听写和打标签的模型加起来约 3.5 GB。整套约 36 GB，两个作曲模型一次只用一个。

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

作曲用到的模型（2026 年 11 月 10 日在各自的官方页面上核对）：
- **YuE2**：代码是 Apache 2.0；模型权重是 CC BY-NC 4.0，也就是非商业许可，另外附了一份给个人创作者的许可。个人用户、内容创作者、以个人身份创作的音乐人，可以免费用这些权重，用它生成的歌可以发布、出售、授权、变现，生成的歌不受权重「非商业」的限制，署名也不是必须的。公司拿权重做商用，要另外找 M-A-P 谈授权。不管谁用，都不能用于违法、有害、欺骗的用途，比如诈骗、骚扰、冒充他人。ComfyUI 的打包版页面同样标的是 CC BY-NC 4.0。
- **ACE-Step 1.5**：代码和模型都是 MIT，模型页写明生成的音乐可以商用。它用到的 Qwen3-Embedding-0.6B 是 Apache 2.0。
- **Qwen3-TTS**：代码和 CustomVoice 模型都是 Apache 2.0。
- **Demucs**：代码是 MIT；预训练模型的许可，官方页面上没有单独写，以模型页为准。
- **python-audio-separator**：MIT。它下载的 BS-RoFormer 模型，我们没找到原作者写明的许可，以模型页为准。
- **Whisper large-v3-turbo、faster-whisper**：MIT。
- **AST**：BSD-3-Clause。**CLAP（clap-htsat-unfused）**：Apache 2.0。

网址最后核对日期：数字人部分 2026 年 10 月 30 日，作曲部分 2026 年 11 月 10 日。
