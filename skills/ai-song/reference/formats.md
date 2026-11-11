# 文件格式

## 风格卡（扒曲的结果）

`analyze_ref.py` 量出数字（`card.json` 里分成 settled 和 listen 两类），写成一页。量得准的直接写；要人听的写「待人听核」，没人能听时就一直留着这个标记，不要猜。风格卡只放在私人工作文件夹里。下面是格式，内容是编的：

```
参考歌：（歌名，只作学习）
速度：约 100 BPM，一小节约 2.4 秒；4/4（候选里的 67 是 3:2，拍子判断排除了 12/8）
小节线：约 12.3 秒处（从鼓点猜的）               调：A 小调，待人听核（和 C 大调只差 0.01）
段落：前奏 4 小节 → 主歌 8 → 副歌 16 → 间奏 4 → 主歌 8 → 副歌 16 → 结尾 4
和弦走向：副歌每小节 C G Am F，像流行四和弦 I–V–vi–IV（测量，待人听核）
音阶色彩：人声常落在 b7 上，大调里的混合利底亚色彩（测量，待人听核）
律动：鼓层每小节八个八分音符的重音 .90 .20 1.00 .20 .80 .20 1.00 .30，像二四拍的反拍（测量，鼓层上量的）
配器角色：铺底是长音（主歌里停）；旋律是一颗一颗的音；鼓从第一段副歌进；贝斯层录音里几乎没有，待人听核
乐器名字：待人听核（或者写人确认过的）
唱法：主歌一小节一句；副歌两小节一句；歌手：待人听核
说的部分：占全曲约 15%；开头 20 秒一段对白，底下没有音乐；结尾一句插话，叠在最后一句副歌上；是说话还是说唱、起什么作用：待人听核
句子：唱的句子音域约 11 个半音；最高音多在句子中间；句尾往下落的约三成
学什么：段落顺序和长短、副歌前的空拍、说的部分放在哪儿
不学什么：原曲的旋律、歌词、音色、说的内容

人答（扒曲后停下来问，答完才写风格描述）：
  风格：（人听了说的，比如「雷鬼」「布鲁斯味的流行」）
  标志性乐器：（人说的，或者你建议、人确认的）
  调：（比如「不影响这一轮：模型反正不照调唱」）    歌手：
  和弦和音阶色彩听着对不对：
  说的部分各起什么作用：
```

每一项「待人听核」都要在「人答」里有答案。人说「不影响这一轮」也算答了，原话和理由写上，以后用到这一项再问。风格描述里的风格、律动和乐器从这里来，不从歌词题材的年代和国别来；和弦、音阶色彩和律动按 SKILL.md 第 3 步的表翻成模型认得的风格词。

## 段落表（填词）

| 段落 | 谁 | 怎么唱 | 小节 | 情绪 | 备注 |
|---|---|---|---|---|---|
| 前奏 | 器乐 | 二胡 | 4 | 安静 | ACE-Step 补 |
| 开头 | 女声 | 唱，两小节一句 | 16 | 从容 | 唱到他推门出去，接上主旋律 |
| 主歌 | 男声 | 唱 | 16 | 带笑 | 核心画面那一句开头 |
| 念白 | 男声 | 说（语音模型） | 1 句 | | 放在间奏的长音上 |

多音字在备注里标读音，比如：宿 xiǔ。小节数按风格卡量出来的句长算（几句乘每句几小节），再折成秒和目标时长比，见下面的计划版歌曲地图。

## 歌词文件

整首歌的词都在这一个文件里，念白也在。作曲模型拿到的是去掉 `[spoken]` 段和 `@` 备注行以后的样子（`lyrics_tools.py` 可以打印出来看）。

```
[verse]
第一句写一个画面，
第二句接着写这个画面。

[chorus]
副歌第一句，
副歌第二句。

[spoken]
@ 男声，像隔着门叫朋友；放在间奏第 2 小节的长音上
你还没睡吧？出来看看月亮。

[Instrumental]
```

- 一句占一行，段和段之间空一行。
- `[spoken]`：念白。`@` 开头的是备注（谁说、什么声音、放在哪儿），下面是要说的话。发给作曲模型前会被去掉，第 6 步交给语音模型。
- `[Instrumental]`：一段纯器乐。它自己就是一个段落标记，下面不写字，也不放在别的段落标记下面。我们在 YuE2 里把它放在最后一段后面，让伴奏接着往下走。ACE-Step 要整首纯器乐时，整个歌词就写 `[Instrumental]`。
- 同音字只换在规格文件的歌词里，歌词文件保持原字。

## 听写结果 `lines.json`（`transcribe_ref.py`）

```json
[{"start": 33.0, "end": 35.1, "text": "（听写出的一句）", "no_speech": 0.02, "words": [[33.0, 33.4, "（字）"]]}]
```

时间是参考歌里的秒数。它是别人的歌词，只放在私人工作文件夹里。

## 风格描述

英文，按这个顺序写：风格，速度，调；人声（男女、高低、性格）和唱法；乐器和各自的角色；情绪；不要什么。

```
Mandarin Chinese rap-pop with Chinese traditional plucked instruments, 99 BPM, D minor.
One playful warm low male rapper, unhurried talk-rap with space between the lines, every word clear,
the easy voice of someone telling a joke, clear rhymes. One continuous beat and harmonic progression.
Dry repeating pipa and muted-guitar ostinato, punchy kick and snare, warm bass, tight hats, dizi fills.
No female voice.
```

不同段落各写一段：唱的段落写 sung, easy melody；说唱段写 talk-rap；纯器乐写主奏乐器和 no vocals。

## 歌曲地图 `song_map.json`

有两个阶段。

**计划版**（第 2 步写词时）：速度从风格卡来，还没有小节线。

```json
{"bpm": 100, "beats_per_bar": 4, "downbeat": null, "measured": false, "key": "A minor", "target_seconds": 30,
 "sections": [{"name": "intro", "bars": 4, "manner": "instrumental"},
              {"name": "verse", "lines": 8, "bars_per_line": 1, "who": "female", "manner": "sung"},
              {"name": "chorus", "lines": 4, "bars_per_line": 2, "who": "male", "manner": "sung"}]}
```

`python scripts/songmap.py song_map.json --target 30` 打印每段几小节、几秒，总长和差多少。段落可以直接写 `bars`，也可以写 `lines` 和 `bars_per_line`；不写 `start_bar` 就一段接一段排。要用时间的脚本（`fit_check.py` 这类）会拒绝计划版。

计划版的时长是初估：出了样片，用 `take_grid.py --lines` 量出每句实际唱几小节，改 `bars_per_line` 再算一次。

**实测版**（第 4 步拼好以后）：补上实测的速度和 `downbeat`（用 `take_grid.py` 量，不用 `analyze_ref.py` 的粗估），改成 `"measured": true`，每段写实际的 `start_bar`。

```json
{"bpm": 99, "beats_per_bar": 4, "downbeat": 0.303, "measured": true, "key": "D minor",
 "sections": [{"name": "intro", "start_bar": 0, "bars": 16, "who": "female", "manner": "sung"},
              {"name": "verse1", "start_bar": 16, "bars": 16, "who": "male", "manner": "sung"}]}
```

`downbeat` 是任意一个小节线在这首歌里的秒数，要量，不能按描述里的速度推，也不能用节拍检测给出的第一拍（那不一定是小节线）。脚本找地图的顺序：`--map` 参数，环境变量 `SONG_MAP`，当前文件夹的 `song_map.json`。

`assemble.py` 拼完会写一份实测版 `成品_map.json`：速度和小节长来自那一版的 `take_grid.py`，`downbeat` 是 0（第一小节前奏的开头），每段的起始小节按 `sections.json` 里每段第一个字和最后一个字算（不到半小节的弱起算下一小节），前奏和结尾各一段。

**拼接参数**（可选）：计划版里可以加一段 `assembly`，键就是 `assemble.py` 的参数名，路径相对这个文件，`assemble.py --map song_map.json` 读它，命令行上给的参数优先：

```json
"assembly": {"take": "takes/verse_s3.flac", "stems": "stems", "grid": "takes.json", "sections": "sections.json",
             "vocal": "comp/verse_s3_fixed_vocals.wav", "ace_spec": "ace/spec.json",
             "intro": "intro_s3", "ending": "end_s2", "ending_ctx_bar": 4,
             "spoken": "spoken/line_t3.wav", "spoken_at": 39.15, "duck": 4, "room": 0.12, "out": "final/song"}
```

## YuE2 规格（`render_yue2.py`）

```json
{"api": "http://127.0.0.1:8190", "checkpoint": "yue2_3b_int8_convrot.safetensors",
 "output_directory": "/path/to/ComfyUI/output", "output_prefix": "mysong/", "max_duration": 60,
 "shared": {"style": "英文风格描述", "lyrics": "[verse]\n...\n\n[chorus]\n..."},
 "variants": [
   {"id": "verse_s1", "seed": 1},
   {"id": "verse_s1_r2", "seed": 1, "render_seed": 2},
   {"id": "verse_fix_s7", "seed": 7, "keep_from": "verse_s1", "keep_seconds": 37.3,
    "lyrics": "[verse]\n...（整段歌词，要修的字换成同音字）"}]}
```

- `seed` 决定旋律和唱词，`render_seed` 只换咬字，默认等于 `seed`。
- `keep_from` + `keep_seconds`：保留那一版前 `keep_seconds` 秒，从那里接着唱。`keep_seconds` 要小于 `max_duration`。
- `rerender`：整版换渲染种子重渲染，修字的第一步。只写这几项：

  ```json
  {"id": "verse_s1_r9", "rerender": "verse_s1", "render_seed": 9}
  ```

  脚本从那一版的 token 文件读出时长（每秒 25 个 token），设 `keep_from` = 那一版，`keep_seconds` = 时长减 0.04 秒，`max_duration` = 时长，只采样最后一个 token。`seed` 默认用那一版记录里的音乐种子；读不到 token 文件时写 `"length": 秒数`。`shared` 里的风格描述和歌词要和那一版生成时一样，不一样它会提醒。那一版自己的渲染种子也跑一个，当对照。
- `lines_per_row`（规格或某一版里）：每行并几句歌词发给模型，默认 1。要短句一小节一句时写 2。按段写成 `{"verse": 2, "chorus": 1}`，键对段落标记或它开头的那个词（`verse` 也管 `[verse 2]`），`default` 管其余的段。
- `max_duration`：这一段计划秒数乘 2，再加 20 秒；量过每句几小节以后，乘 1.2 再加 20 秒。

副歌接着挑中的主歌唱（默认做法）：

```json
{"id": "chorus_from_v3_s1", "seed": 1, "keep_from": "verse_s3", "keep_seconds": 31.2,
 "max_duration": 75, "lyrics": "[verse]\n...（主歌，和那一版一样）\n\n[chorus]\n...（副歌）"}
```

`keep_seconds` 取主歌那版最后一句词之后的第一个小节线：`take_grid.py` 打印的 `downbeat` 加整数个 `bar_s`，刚过 `sung_to`（`take_sections.py` 量的最后一句词的结尾）的那一个。没有 `sung_to` 才用 `voice_to`，它会把词后面的哼唱也算进去。`max_duration` 要算上保留的那一段。

## 每版的测量 `takes.json`（`take_grid.py`）

每版一行：`key`、`key_next`、`key_gap`（小于 0.03 算没定）、`bpm`、`bar_s`、`beat_fit_sd_ms`（跟拍拟合的误差）、`downbeat`、`pulse`（跟拍原来锁在哪个脉冲上，1:1 以外说明换过）、`voice_from`、`voice_to`（人声能量，含哼唱和重唱）、`peak`、`clipped_samples`、`hit_cap`（唱到了 `max_duration`）、`notes`、`pass`（按 `--key`、`--bpm`、`--tol` 筛）。

给了分离结果时多出 `bpm_mix`、`grid_mix`（混音上量的格子：`bpm`、`beat_s`、`bar_s`、`downbeat`，`cut_take.py`、`harmony_groove.py` 加 `--grid-from mix` 时用它）、`tempo_agree`（伴奏和混音上的速度差在 2% 以内）。给了 `--sections` 时多出 `sung_from`、`sung_to`（第一句词的开头、最后一句词的结尾）、`sung_bars` 和按段的 `sections`：`[{"name": "verse", "lines": 8, "start": 12.0, "end": 30.2, "bars": 7.8, "bars_per_line": 0.98}]`。只给 `--lines` 时，`bars_per_line` 按人声能量算，只能当粗估。

## 每段在一版里的位置 `sections.json`（`take_sections.py`）

每版一行：

```json
{"id": "verse_s3", "seconds": 60.0, "sung_from": 12.0, "sung_to": 41.5, "heard_mean": 0.9,
 "sections": [{"name": "verse", "lines": 8, "heard_lines": 8, "start": 12.0, "end": 30.2, "line_period_s": 2.3,
               "line_detail": [{"text": "（这一句词）", "heard": 1.0, "start": 12.0, "end": 14.1, "got": "（听写出的字）"}]}],
 "lead_in": {"seconds": 8.0, "from": 2.5, "text": "嗯嗯嗯"},
 "after": {"seconds": 12.0, "to": 60.0, "text": "", "repeat_from": null},
 "cut_off": null, "notes": []}
```

- `heard`：这一句里拼音对上（或者韵母对上）的字占多少。
- `line_period_s`：每句起点对句号拟合的斜率，秒每句；`take_grid.py --sections` 把它换算成每句几小节。
- `lead_in`：第一句词之前有多少秒人声，听写出了什么。`after`：最后一句词之后，`repeat_from` 是从头再唱一遍的起点。
- `cut_off`：最后一句没唱完就到了这一版的结尾，或者最后几句没听到而人声一直到结尾。
- `notes`：哪一段没找到、哪几句没听写到（附补听的命令）、两段之间有多少秒人声。
- `gaps`：有人声却一句词也没听到的地方（起止、哪几句）；`filled`：自动重听了哪一段、多听到几句。重听多听到了词时，听写文件也会更新。

`take_grid.py --sections`、`score_takes.py --sections`、`cut_take.py --sections` 和 `harmony_groove.py --sections` 都读它。

## 和弦、音阶色彩、律动 `hg.json`（`harmony_groove.py`）

每段一行：`bpm`、`first_bar_line`、`bars`、`key`、`key_from`（给定的、从和弦套路来的，或者音高分布拟合的）、`chords`（每小节一个，`N` 是认不出）、`roman`、`progression`（认出的套路、在哪个调上、换和弦时有多少照它走）、`progression_candidates`（最像的三个常见套路，没认出时也列）、`chords_pinned`（`--pin` 钉上的、人核过的和弦走向）、`degrees`（人声落在各音阶级数上的时间占比）、`degrees_over_V`（小调时，相对属音再算一遍）、`colour`（标出的特色音）、`groove`（`profile_8ths` 每小节八个八分音符上的重音，`kick_band`、`snare_band`，`two_bars_8ths`，固定小节起点的 `scores`、`names`，转一圈找最像位置的 `scores_phase_free`、`names_phase_free`，鼓层比伴奏低多少 `drums_below_backing_db`，鼓层几乎没声时的 `not_measured`）、`words`（写风格卡用的句子）。没量的段落只有 `skipped` 和原因。加 `--compare` 时多出 `comparison`：每段三列对没对上（`None` 是不算）、和弦一列的权重和是否钉过、样片最像的套路、两种律动读法的相关系数，以及整体的 `overall`、`weights`、`score`、`not_measured` 和结论 `verdict`（MATCH、PARTIAL、REJECT、NOT MEASURED、NOT DECIDED）。参考歌的这份文件只放私人工作文件夹。

## 剪出来的样片 `样片.json`（`cut_take.py`）

`from_take_s`、`to_take_s`（在原来那版里的位置）、`seconds`、`sung_from_in_sample_s`、`sung_to_in_sample_s`、`bars`、`gain_db`、`integrated_lufs`、`true_peak_dbtp`、`lufs_target`（给了 `--lufs` 时）、`grid_from`、take_grid 那一行的 `grid` 和实际用的 `grid_used`，和 `warnings`（比如速度对不上、响度到不了目标）。
- 规格文件旁边会生成 `records/`（每版的记录）和 `tokens/`（每版的音乐 token）。路径要是 ComfyUI 那台机器能写的。

## ACE-Step 规格（`ace_step_run.py`）

```json
{"bpm": 99, "keyscale": "D minor", "duration": 30, "lm_negative_prompt": "",
 "candidates": [
   {"id": "verse_ace_s1", "seed": 1, "caption": "英文风格描述", "lyrics": "[verse]\n...", "vocal_language": "zh"},
   {"id": "intro_s1", "seed": 1, "task": "repaint", "src_audio": "src/backing_bars1-8.wav",
    "caption": "Chinese folk pop backing in D minor, 99 BPM, a solo erhu melody, no vocals.",
    "repainting_start": -9.7, "repainting_end": 0, "duration": 29.1},
   {"id": "ending_s1", "seed": 1, "task": "repaint", "src_audio": "src/backing_before_end.wav",
    "caption": "... then the drums and bass stop, a solo erhu plays a slow gentle melody, everything settles on a final D minor chord. No vocals.",
    "repainting_start": 19.4, "repainting_end": 36.4, "duration": 36.4}]}
```

- 不写 `task` 就是整首生成；`lyrics` 默认 `[Instrumental]`。
- repaint：`src_audio` 是截下来的原伴奏，从小节线截起。`repainting_start` 为负数时往前写这么多秒（四小节最合适），`repainting_end` 超过原长时往后写。写结尾时只给前面的上下文，从要变的小节写到底。
- 筛拍子：`beat_check.py --spec 这个规格 --bar-s 一小节的秒数`，按 `repainting_start` 找换接的小节线（负数是前奏，新小节在前；写到底的是结尾）。
- 拼进成品：`assemble.py --ace-spec 这个规格 --intro 候选id --ending 候选id`，再告诉它上下文是从原版第几小节截的：`--intro-ctx-bar`（默认 0）、`--ending-ctx-bar`。新小节有几个、结尾从第几小节开始写，它从 `repainting_start` 算。
- 还有 `cover`（给 `src_audio` 和 `audio_cover_strength`）和 `lego`（给 `track`，比如 `erhu`），后者会把整段重画，不适合做点缀。

## 贴片（`comp_vocal.py`）

```json
{"name": "verse_fixed", "base": "stems/verse_s1_vocals.flac",
 "patches": [["stems/verse_fix_s7_vocals.flac", 0.0, 37.40, 37.95]]}
```

每个贴片是：那一版的人声文件，它在原来那版时间轴上的起点，从哪一秒贴到哪一秒。贴片的起止点选在能量低谷，切完把两边都听写一遍。只分离了一段的（`separate.py --start 28 --end 42`），起点写 28。整版重渲染的版起点是 0，切点放在所有渲染共有的低谷上。

## 咬字打分 `score.json`（`score_takes.py`）

每版一行：`mean`、`per_line`（每句听对的比例，取几个窗口里最好的）、`windows`、`windows_dropped`、`segments_cut`、`heard`（第一个窗口听写的字）、`heard_all`（每个窗口的）、`wrong`：算错的字，`[{"line": 12, "char": "（字）", "at": 0, "missed": "3/3"}]`，在至少两个窗口里没听到（按拼音，不看声调）。

## 成品记录 `成品.json`（`assemble.py`）

- `grid`：小节长、那一版的小节线、成品第 0 小节是那一版的第几小节、`take_shift_s`（成品秒数 = 那一版秒数 + 它）。
- `loudness`：目标、拼完的响度和真峰值、加的增益、`limiter`（`used`，用了的话：限到多少、压了多久、最多压了几 dB）、wav（和 mp3）量出来的响度和真峰值、采样峰值。
- `checks`：`intro`、`ending`（文件、同一段音频的增益和相关、换接点在成品哪一秒、`beat`：`beat_check.py` 的结论）；`take_kept`（和原版不一样的几段，`differs`：成品的小节和秒数、最低相关）；`quiet_half_bars`（安静的半小节，和原版同一位置比的结论）；`sample_jumps`；`spoken`（念白在成品哪几秒、增益、压了几 dB、念白和它底下伴奏的电平）。
- `notes`：比如「真峰值说了算，差多少 LU，峰值在哪」。
