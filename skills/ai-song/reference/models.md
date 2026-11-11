# 模型、工具和环境

我们只在一台 NVIDIA DGX Spark（128 GB 内存，CPU 和显卡共用）上跑通过。下面的耗时都是在这台机器上、显卡同时还有别的任务时量的，换机器会不一样。

## 作曲模型

**YuE2**（香港科技大学、M-A-P 开源社区等机构）
- GitHub：https://github.com/multimodal-art-projection/YuE
- 给歌词和一段风格描述，人声和伴奏一起出来。我们在 ComfyUI 里跑。
- **节点从哪来：** YuE2 的节点（`EmptyYuE2LatentAudio`、`YuE2GenerateMusic` 等，代码在 `comfy_extras/nodes_yue2.py` 和 `comfy/text_encoders/yue2.py`）是 ComfyUI 主线自带的，不是插件。我们用的是 Comfy-Org/ComfyUI 的 master，提交 `5f415e31abfd837870bbc733ba1e158de71ed985`，另外要两个 Python 包：`comfy-kitchen` 0.2.40 和 `comfy-aimdo` 0.6.0。更早的 ComfyUI 没有这些节点。
- **模型文件：** `yue2_3b_int8_convrot.safetensors`，3.96 GB（3,960,938,800 字节），来自 https://huggingface.co/Comfy-Org/YuE2 ，放在 ComfyUI 模型目录的 `checkpoints/` 下。
- **可能要单独装一套 ComfyUI。** 大家共用的那个 ComfyUI 版本可能太旧，没有 YuE2 节点，也不好为它升级。我们的做法：另装一个这个版本的 ComfyUI，用同一个 Python 环境，加一层 Python 包覆盖（`PYTHONPATH` 指向装了上面两个包的文件夹；我们那台还要 `CPATH` 指向 Python 头文件，编译用），模型、输入、输出、临时、用户文件夹都单独放，单开一个端口。`scripts/yue2_service.sh` 用 `COMFY_ARGS` 把这些参数传给 `main.py`，例如 `--disable-all-custom-nodes --offline --models-directory … --input-directory … --output-directory … --temp-directory … --user-directory …`。`--disable-all-custom-nodes` 加上脚本自带的 `--whitelist-custom-nodes yue2_continue`，只加载接着唱那一个节点。
- 从任意时间接着唱：把 `scripts/yue2_continue` 复制到 ComfyUI 的 `custom_nodes/`。它保存每一版的音乐 token（每秒 25 个），下次可以保留前面若干秒，从那里接着生成。ComfyUI 自己的文件不改。
- 实测：四句说唱（56 个字）每版约 28 秒；`max_duration` 40 秒、显卡同时有别人在用时，每版 46–64 秒。
- 它不照风格描述里的调和速度唱（一次测试 16 版落在 8 个调上，98–118 BPM），每版都要用 `take_grid.py` 量。
- 许可证：代码 Apache 2.0；模型权重 CC BY-NC 4.0，附带创作者许可（2026 年 11 月在仓库的 MODEL_LICENSE 里读到）：个人、内容创作者和以个人身份创作的音乐人，可以免费发布生成的作品并变现，生成的作品不受权重「非商用」的限制；公司拿权重做商用，要找 M-A-P 另外授权。用之前再读一遍原文。

**ACE-Step 1.5**（ACE Studio 和阶跃星辰）
- GitHub：https://github.com/ace-step/ACE-Step-1.5
- 我们用 XL SFT 模型（`acestep-v15-xl-sft`），作曲时加 4B 的语言模型（`acestep-5Hz-lm-4B`）先规划。能整首生成、翻唱一段伴奏、在一段现成音频的前后或中间重写几小节（repaint）。
- 实测：同样四句每版约 59 秒（50 步），另加约 217 秒载入模型；显存占用约 20 GB，跑的时候显卡上别放别的大模型。repaint（补前奏、结尾）不载入语言模型，快得多：一次 12 版连载入共 257 秒，每版 9–22 秒。
- 许可证：MIT。

## 分离、听写和其他

| 工具 | 用来做什么 | 地址 | 许可证 |
|---|---|---|---|
| python-audio-separator | 统一调用下面两个分离模型 | https://github.com/nomadkaraoke/python-audio-separator | MIT |
| BS-RoFormer（`model_bs_roformer_ep_317_sdr_12.9755.ckpt`） | 分出人声和伴奏，用来打分、听写 | 随 audio-separator 下载 | 用前查清楚 |
| Demucs（`htdemucs_ft`，Meta 开源） | 分成人声、鼓、贝斯、其他四层，用来拆参考歌、调结尾各层音量 | https://github.com/facebookresearch/demucs | MIT |
| AST（`MIT/ast-finetuned-audioset-10-10-0.4593`） | 扒曲时给乐器和声音打标签（AudioSet 527 类），346 MB | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 | BSD-3-Clause（以模型页为准） |
| CLAP（`laion/clap-htsat-unfused`） | 扒曲时在候选清单里选乐器和风格，614 MB（Hugging Face 缓存里可能同时存一份 .bin 和一份 .safetensors，占 1.2 GB） | https://huggingface.co/laion/clap-htsat-unfused | Apache 2.0（以模型页为准） |
| diffq | audio-separator 里的 Demucs 一加载就要导入它（audio-separator 0.47.1 上遇到），不装 `--model parts` 直接报错 | `pip install diffq`（带编译的扩展） | 用前查清楚 |
| faster-whisper（Whisper large-v3-turbo） | 听写、按歌词强制对齐 | https://github.com/SYSTRAN/faster-whisper ，Whisper：https://github.com/openai/whisper | MIT |
| pypinyin | 按读音比较听写结果 | https://github.com/mozillazg/python-pinyin | MIT |
| librosa | 扒曲：速度、调、段落、音高 | https://github.com/librosa/librosa | ISC |
| ffmpeg + Rubber Band | 解码、变速、渐慢（ffmpeg 要带 rubberband 滤镜） | https://ffmpeg.org ，https://breakfastquay.com/rubberband/ | 看你用的构建 |
| Qwen3-TTS | 念白：用它自带的声音（我们用过四川话男声），或者用 VoiceDesign 模型（`Qwen3-TTS-12Hz-1.7B-VoiceDesign`）按文字设计一个声音；6 版一句话约 1.7 分钟 | https://github.com/QwenLM/Qwen3-TTS | 看仓库说明 |
| ComfyUI | 跑 YuE2 | https://github.com/Comfy-Org/ComfyUI | GPL-3.0 |

diffq 装不上时（它要编译），用 `scripts/stubs/diffq`：一个只供导入的空壳。`separate.py --model parts` 发现没有 diffq 会自己用它，`htdemucs_ft` 没有量化，用不到 diffq 的任何功能；真碰到量化模型，空壳会直接报错。

## 下载大小（我们机器上的文件）

| 模型 | 大小 |
|---|---|
| YuE2：`yue2_3b_int8_convrot.safetensors` | 3.96 GB |
| ACE-Step 1.5：`acestep-v15-xl-sft` | 19 GB |
| ACE-Step 1.5：`acestep-5Hz-lm-4B`（整首生成时用） | 7.9 GB |
| ACE-Step 1.5：`Qwen3-Embedding-0.6B`、`vae` | 1.2 GB、322 MB |
| BS-RoFormer（分人声） | 610 MB |
| Demucs `htdemucs_ft`（分四层） | 4 个文件，各 81 MB |
| Whisper large-v3-turbo（faster-whisper 版） | 约 1.5 GB |

下载前把大小告诉用户，等用户同意。

## Python 环境

几个模型的依赖互相冲突，我们分了几个环境。每个环境都跑一次 `check_setup.py`，看它对应的那几组：
- ComfyUI 自己的环境：YuE2（组 yue2）。
- ACE-Step 1.5 的环境：`ace_step_run.py`（组 ace）。一个以前跑通过的安装，往往还要它当时的那些环境变量，`ace_step_run.py` 在载入 ACE-Step 之前自己设好，并把设了什么打印出来：
  - 离线：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，跑的中途不下载（`--online` 关掉）。
  - 缓存放回安装自己的文件夹（`--runtime 目录`，或环境变量 `ACESTEP_RUNTIME`）：`HF_HOME=目录/hf_cache`、`HF_MODULES_CACHE=目录/hf_modules`、`TRITON_CACHE_DIR=目录/triton`、`TORCH_HOME=目录/torch`、`XDG_CACHE_HOME=目录/xdg`、`NUMBA_CACHE_DIR=目录/numba`、`TMPDIR=目录`。离线时，安装当初下过的文件只在它当时的 `HF_HOME` 里找得到。
  - `CPATH` 指向 Python 头文件：torch 有些矩阵乘法即使不编译模型也走 Triton，Triton 要现编一个 C 小程序，需要 `Python.h`。这个 Python 自带头文件就用它的；没有的话给 `--python-headers 目录`（或 `ACESTEP_PY_HEADERS`），目录下要有 `python3.X/Python.h` 和 `<架构>/python3.X/pyconfig.h`，比如把 python3.X-dev 包解到那里。缺了它，模型载入完、第一版生成时才报错。
  - 另外设 `ACESTEP_CHECKPOINTS_DIR`（等于 `--checkpoints`）、`TOKENIZERS_PARALLELISM=false`、`PYTHONDONTWRITEBYTECODE=1`。
  我们那台机器上：`--runtime <安装>/tools/runtime --python-headers <安装>/tools/python_headers/usr/include`。
- 装了 audio-separator 的环境，在显卡机器上：`separate.py`（组 separate）；librosa 也在里面的话，`analyze_ref.py` 可以在这里跑（组 analyse）。
- 装了 faster-whisper 和 pypinyin 的环境：`transcribe_ref.py`、`take_sections.py`、`score_lines.py`、`score_takes.py`、`forced_reading.py`（组 score）。显卡机器上的环境可能没装这两个，用笔记本的 CPU 跑也行：一首 3 分 43 秒的人声，large-v3-turbo int8 整首听写一遍约 80 秒。
- 装了 torch 和 transformers 的任意环境：`tag_ref.py`（AST 和 CLAP，第一次用时从 Hugging Face 下载；CPU 就行，一首 3 分多钟的歌约 1 分钟）。我们用的是 transformers 4.51.3。别用 `laion/larger_clap_music`：在这个版本上它对不同的音频给出一模一样的向量（音频那半边没在工作），`tag_ref.py` 会检查这一点。
- 装了 qwen-tts 的环境：念白（组 tts，数字人 skill 的 `voice_design.py`）。它常是单独的一个环境，模型放在本地文件夹：设 `QWEN_TTS_DESIGN` 指向它，再设 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，跑的时候不下载。`check_setup.py --for tts` 在别的环境里跑，也会找出装了它的环境和本地的模型文件夹。念白的检查 `spoken_qc.py` 在打分的环境里跑（组 score）。
- 一个普通环境（numpy）：拼接（`assemble.py`、`join_takes.py`、`beat_check.py`）、电平、渐慢、歌曲地图。

注意 librosa 1.0 以后只读 libsndfile 能读的格式，m4a、aac、mp4 读不了。脚本都先用 ffmpeg 解码，所以什么格式都能给。

## ffmpeg

每个脚本都按名字调用 `ffmpeg`，它要在 PATH 上，变速和渐慢还要带 rubberband 滤镜。有的机器上 ffmpeg 不在 PATH 上，只有某个 Python 包自带的一份（比如 imageio-ffmpeg 带的，文件名不叫 ffmpeg）。`check_setup.py` 会找到它并打印路径；在 PATH 上的某个文件夹里放一个叫 `ffmpeg` 的包装脚本或链接指向它就行。

## 硬件和共用

- 生成本身不费时间，费时间的是来回试听。我们那首用 YuE2 的一轮里生成了 971 版，渲染加起来 2.9 小时，这一轮前后跨了 33 小时。不要省生成，要省人试听的轮数。
- 显卡有别人在用时：单开一个 ComfyUI 实例出歌（`scripts/yue2_service.sh`），提交前看队列是空的，不取消别人的任务；ACE-Step 这类大模型等显卡空了再跑。
- 分离要几分钟：一首 3 分 43 秒的歌，在一台别人也在用的 GB10 上，分人声 325 秒，分四层 78 秒；16 版 40 秒的生成版分人声用了 19 分钟（每版约 72 秒），比生成还久，所以先用 `take_grid.py` 在混音上筛掉调和速度不对的，再分离。修字时也一样：一次修字用了 23 分钟显卡，分离整版占了 13.7 分钟，其实只要分出要贴的那几秒（`separate.py --start --end`）。长任务挂在后台跑（`setsid nohup … > 日志 2>&1 < /dev/null &`），SSH 断了也不会停。
- 在 Windows 上，librosa 依赖的 numba 可能被系统的应用控制策略拦下（「DLL load failed … An Application Control policy has blocked this file」），`analyze_ref.py` 就跑不了；放到 Linux 机器上跑。`take_grid.py`、`join_takes.py`、`cut_take.py`、`harmony_groove.py` 只要 numpy 和 ffmpeg，Windows 上也能跑。分四层（`--model parts`）在显卡忙的时候可以放 CPU 上跑（`CUDA_VISIBLE_DEVICES=""`）：一段 35 秒的样片在 GB10 的 CPU 上约 2.5 分钟。
- 脚本自己不写 `__pycache__`（入口里设了 `sys.dont_write_bytecode`）；从别的代码 import 它们时，设 `PYTHONDONTWRITEBYTECODE=1`。
- Whisper 打分批量的时候，放在有显卡、装了 faster-whisper 的机器上更快，只把结果传回来；在笔记本 CPU 上，24 版要十分钟以上。
