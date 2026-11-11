# 脚本

都要 ffmpeg 在 PATH 上（变速和渐慢要带 rubberband 滤镜）。用哪个 Python 环境见 `../reference/models.md`。每个脚本开头的说明里有完整用法。

## 可以直接用的

| 脚本 | 做什么 | 输入 | 说明 |
|---|---|---|---|
| `check_setup.py` | 开工前查这台机器、这个环境，按步骤分组 | `--for`；环境变量 `COMFY_URL`、`ACESTEP_REPO`、`ACESTEP_CHECKPOINTS`、`QWEN_TTS_DESIGN`、`FFMPEG` | 只查，不装；会试导入 Demucs 那条路径，会找出不在 PATH 上的 ffmpeg；`tts` 组在这个环境里缺语音模型时，会找出装了它的环境和本地的模型文件夹 |
| `songmap.py` | 歌曲地图（计划版和实测版）和共用的音频函数；直接运行时算段落时长 | `song_map.json`，`--target 秒数` | 别的脚本都 import 它；要用时间的脚本拒绝计划版 |
| `lyrics_tools.py` | 歌词文件：去掉 `[spoken]` 段给作曲模型，列出念白，数每段句数字数；可以两句并一行，也可以按段并 | 歌词文件，`--per-row 2` 或 `--per-row verse=2,chorus=1` | `render_yue2.py`、`ace_step_run.py` 自动调用（规格里的 `lines_per_row`，数字或按段的表） |
| `take_grid.py` | 量每一版：调、速度（跟拍后直线拟合，4:3、3:2 的错拍会换回来）、小节线、峰值、是否唱到 `max_duration`；有分离结果时再在混音上量一次速度对照；给了 `--sections` 时按段给出每句几小节；按调和速度筛 | 生成的版，可选它们的分离结果和 `sections.json` | 只要 numpy 和 ffmpeg；混音上只能粗筛调和速度，看不出被切的版 |
| `take_sections.py` | 按词找出每段在一版里的位置：听写人声，按拼音对歌词，给出每句、每段的起止，前面哼了多久，后面有没有即兴、重唱，是不是被切了；有人声却没听到词的地方自动用长窗口重听 | 各版的人声、歌词文件、`--sections verse,chorus` | 要 faster-whisper 和 pypinyin；笔记本 CPU 上每版约 50 秒；听写结果存在 `take_lines/`，再跑直接复用；`--no-fill` 不重听 |
| `cut_take.py` | 从一版里剪出样片或一段：第一句前一小节到最后一句后一小节，淡入淡出，真峰值 -1.2 dBTP；`--lufs` 对齐一组样片的响度 | 那一版、`takes.json`、`sections.json`，`--through 段名`，`--grid-from mix` | 只要 numpy 和 ffmpeg |
| `harmony_groove.py` | 按段量和弦（每小节，认出常见套路，没认出也列最像的）、人声的音阶色彩（标出 b2、#4 这类特色音）、鼓的律动（固定小节起点和转一圈两种读法，认出 3+3+2、二四拍反拍等）；写出风格卡用的句子；`--compare` 拿样片和参考歌比，结论分 MATCH、PARTIAL、REJECT、NOT MEASURED、NOT DECIDED | 伴奏、鼓层、「其他」层、人声，段落（风格卡、`sections.json` 或 `--span`）；`--pin`、`--ref-chords` 钉参考歌的和弦走向；`--grid-from mix` | 只要 numpy 和 ffmpeg；律动在鼓层上量，和弦在「其他」层上量；和弦最粗，没钉时只算半个权重 |
| `join_takes.py` | 两版在小节线上拼：调要相同，速度差了就整体拉伸，按伴奏对音量，查拼接处的拍子（`beat_check.py`：八分音符上的相位差、每小节重音错开几个八分音符），真峰值压到 -1 dBTP，打印响度 | 两版和它们的分离结果，`--shift-ms` | 只要 numpy 和 ffmpeg；默认做法是接着唱，这个是退路 |
| `beat_check.py` | 新的部分（拼上的一版、新前奏、新结尾）和旁边的音乐拍子对不对：八分音符上的相位差，每小节八个八分音符上的重音要错开几个才对得上，结尾新写的第一下离小节线多远 | 一个文件和换接的小节线（`--join`、`--bar-s`、`--new before`、`--ending`），或者 ACE-Step 规格（`--spec`） | 只要 numpy 和 ffmpeg；不在拍子上量（3+3+2 会拉偏最多半拍）；结尾只剩一个长和弦时只看第一下；`join_takes.py`、`assemble.py` 都用它 |
| `assemble.py` | 拼成品：前奏（自带的或 ACE-Step 写的）、这一版的伴奏和（贴过的）人声、结尾、念白和它下面压一点的伴奏，都在小节线上；查同一段音频、拍子、和原版不一样的小节、安静的半小节、跳变；响度按真峰值说了算，`--limiter` 可选；写实测版歌曲地图 | 那一版、分离结果、`takes.json`、`sections.json`、ACE-Step 规格和候选 id、念白；或者 `song_map.json` 的 `assembly` 段（`--map`） | 只要 numpy 和 ffmpeg；写 `成品.wav`、`_voice.wav`、`_bed.wav`、`.json`、`_map.json`，`--mp3` 时加 mp3；采样碰到满刻度就停 |
| `separate.py` | 分人声和伴奏，或者分成四层 | 任意格式的音频；`--start`、`--end` 只分其中一段 | 要 audio-separator；非 WAV/FLAC 先用 ffmpeg 解码；没有 diffq 时 `--model parts` 用 `stubs/diffq`；一首歌几分钟，挂后台跑；修字的候选只分要贴的那几秒，放进单独的 `--out` 文件夹 |
| `stubs/diffq/` | diffq 的导入空壳 | | 只给不量化的 Demucs 模型用，真用到会报错 |
| `transcribe_ref.py` | 扒曲：分窗口听写人声（窗口从换气处开始），只删编出来的分句（含「词曲」这类署名），接回在窗口边上被切开的句子，列出没听写到的人声；`take_sections.py` 也用它 | 人声，`--from/--to`，`--merge-into` | 要 faster-whisper，CPU 就行（3 分多钟的歌约 2 分钟）；参考歌的结果是别人的歌词，只放私人文件夹 |
| `tag_ref.py` | 扒曲：模型按段给乐器和风格打标签（AST、CLAP），写进风格卡的「模型听到的（待人听核）」；太短、太静的段落跳过 | 参考歌，伴奏层，风格卡，`--extra-instruments`、`--extra-genres`，`--plan` | 要 torch 和 transformers（`--plan` 不用）；只是猜测；会检查 CLAP 的音频向量是不是全一样 |
| `analyze_ref.py` | 扒曲：速度候选和拍子（8 对 12 个八分音符）、小节线、调、段落、各层音量和乐器角色、唱和说的句子（提示），分成「测量能定」和「待人听核」 | 参考歌，人声，四层，听写结果 | 要 librosa（Windows 上 numba 可能被拦，放 Linux 跑）；音高只在句子上算，`--no-pitch` 跳过；打印每步用时 |
| `yue2_continue/` | ComfyUI 节点：保存 token，从任意时间接着唱 | 复制进 ComfyUI 的 `custom_nodes/` | 要支持 YuE2 的 ComfyUI 版本；只改 YuE2 的编码，ComfyUI 的文件不动 |
| `yue2_service.sh` | 单开一个 ComfyUI 实例出歌，空了再停 | `COMFY_DIR`、`COMFY_PY`、`PORT`、`COMFY_ARGS`（传给 main.py 的其他参数） | 只停自己启动的那个进程；`PYTHONPATH`、`CPATH` 从调用的环境继承 |
| `render_yue2.py` | 按规格批量出 YuE2，一次一版 | 规格 JSON | 队列不空就不提交；已经出过的跳过；歌词里的 `[spoken]` 段自动去掉；`lines_per_row`；`max_duration` 的算法写在开头说明里；`"rerender": 那一版` 整版换渲染种子重渲染（保留除最后一个以外的全部 token），风格和歌词和原来不一样时会提醒 |
| `ace_step_run.py` | ACE-Step 1.5：整首、repaint、cover、lego | 规格 JSON，`--repo`、`--checkpoints`，`--runtime`、`--python-headers` | 自己设离线、缓存目录和 `CPATH`（见 `../reference/models.md`）；在一个测试里跑通了 4 版 text2music |
| `score_lines.py` | 按歌词强制对齐，每个字一个概率；也是共用的听写库 | 人声、起止秒数、歌词 | faster-whisper，`WHISPER_MODEL`、`WHISPER_DEVICE` |
| `score_takes.py` | 多版比咬字：几个窗口听写，按拼音对歌词；列出算错的字（`wrong`：三个窗口里至少两个没听到） | 多个人声文件、歌词；`--sections sections.json`（只听第一句到最后一句），或者每版自己找人声起点 | 要 pypinyin；编出来的段删掉，大半是编的窗口作废；只剩一个窗口时不判错；CPU 上每版 1–3 分钟，`WHISPER_DEVICE=cuda` 快很多 |
| `forced_reading.py` | 判多音字唱的是哪个读音 | 人声、起止秒数、带 `{}` 的那句词、候选字 | 那句词不带标点；换同音字重唱之前先跑 |
| `spoken_qc.py` | 念白：每版的听写错字率（按拼音）、每秒几个字、音高中位数；混进去以后在三个窗口里听写 | 念白的几版、那句话、`--voice male/female`；`--mix 成品 --at 起-止` | 要 faster-whisper 和 pypinyin；数字人 skill 的 `tts_qc.py` 要参考声音，不适合单独一句设计出来的声音 |
| `fit_check.py` | 字踩不踩拍、音在不在和弦里 | 同一时间轴的人声和伴奏、起止秒数 | 和弦判断比较粗 |
| `lead_notes.py` | 在 16 分音符格子上打印主奏的音 | 器乐段、小节范围 | 判断器乐段有没有动机 |
| `bar_levels.py` | 每小节电平、安静的小节、峰值、跳变；安静的小节和原版同一位置比，分出写好的停顿和洞 | 混音，可选人声；`--ref 原版 --ref-offset 秒数` | 交付前检查；`assemble.py` 也用它 |
| `comp_vocal.py` | 把重唱的几个字贴回原来的人声 | 贴片 JSON | 贴完要在混音里再判；整版重渲染的版时间不变，整句贴在共有的低谷上 |
| `smooth_lead.py` | 去掉 ACE-Step 新写的主奏的破音毛刺 | 那一版，和它分出来的四层，新小节的起止 | 处理完要人听 |
| `ritard.py` | 结尾渐慢：人声 WSOLA，伴奏 rubberband | 干人声、伴奏、开始和结束的小节 | 也能当库用 |

`--bpm`/`--downbeat` 和 `--map` 二选一：给了 `--bpm` 就用它，不然读实测版歌曲地图。

在 Windows 控制台上，要打印中文的脚本用 `python -X utf8` 跑。

脚本不往 skill 文件夹里写 `__pycache__`（入口设了 `sys.dont_write_bytecode`，skill 根目录也有 `.gitignore`）；从别处 import 它们时设 `PYTHONDONTWRITEBYTECODE=1`。

## 例子（我们那首歌的，照搬跑不起来）

在 `examples/` 里。它们读 `SONG_DIR`（默认当前文件夹）下我们那首歌的文件布局（`stems/`、`review/`、`audio/`、`ace_beds/`）和 `song_map.json`，但小节号、段落名、版本名、增益曲线都是那首歌的，换歌要改。拿来看做法，或者照着改。

| 脚本 | 做什么 | 还写死的东西 |
|---|---|---|
| `grow_ends.py` | 给整首接上 ACE-Step 写的前奏和结尾，可选尾声安静、加一点房间声、渐慢，最后统一响度（一般的拼法用 `../assemble.py`；这里多了按层压音量和渐慢） | 段落名（第三遍、末副歌）、各层增益的小节号、渐慢的起止小节、`review/mix_<名>.json` 记录格式 |
| `make_spec_ending.py` | 生成「结尾一口气写完」的 ACE-Step 规格和上下文音频 | 起止小节、caption、输入文件名 |
| `score_ending.py` | 结尾候选的筛选：上下文、撤鼓、根音、拍子（`beat_check.py`）、和人声撞不撞 | 小节号、输入文件名 |
| `verify_draft.py` | 整首交付检查：每段听写、保留的那版是否原样、安静小节、跳变 | 记录格式、保留那版的时间 |
| `check_section_in_mix.py` | 一段放回整首里判：每半小节电平、念白下面的伴奏、混音里听写 | 小节号、两句念白的位置 |
| `score_fit_then_words.py` | 两段式筛选：先节拍和和弦，再给前几名听写 | 和弦表、小节号、歌词 |
| `measure_takes.py` | 两个模型对比时的信号测量：速度、电平、高频、忙不忙 | 文件命名 |
