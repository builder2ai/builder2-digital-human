# 提示词模板

下面是我们 MV 里实际用的原句。照着结构写，把方括号里的内容换成你自己的角色、歌词、动作和镜头就能用。

## 1. 出图：参考图 + 文字生图（MiniMax H3-Image）

工作流里五张参考图按连线顺序是 `<Picture 1>` 到 `<Picture 5>`。提示词分四块：

1. **哪张图管什么**：前三张是同一个人的脸（正面、偏左、偏右），第四张管帽子，第五张管衣服。
2. **人物和造型**：年龄、身份、帽子和衣服的细节、表情。
3. **这个镜头**：站位、景别、光从哪里来、背景是什么。
4. **画面风格**：什么场合的照片、镜头、调色，画面里不要文字。

我们的原句（定妆照）：

```
<Picture 1>、<Picture 2>、<Picture 3>是同一个男人的脸，分别是正面、偏左和偏右。画面里的男人就是他。他的脸型、眉毛、眼睛、鼻子、嘴、颧骨、皱纹和肤色和这三张完全一样，是同一个人，不要把他画成别人。四十多岁，北宋文人苏东坡。他的东坡巾照<Picture 4>这幅古画里的帽子画成真实的帽子：用不透明的黑色硬布做的，硬挺，表面是哑光的布纹，不是网纱，不透光，看不到里面；帽子很高，约有一个半头那么高，方筒形，平顶；分里外两层：里层是一个高高的方筒，外层是一圈矮一截的硬挺围片，围片的上沿比里层的顶低一截，形成台阶一样的两层轮廓，正面中间正对着一道竖着的棱和折缝。他的衣服照<Picture 5>这幅古画：身穿浅米白色的交领宽袖长袍，领口和袖口有宽宽的深褐色镶边；腰间系黑色丝绦，打结后垂下两条带流苏的绦带。胡子是八字胡，下巴上一绺较长的胡须。只照前三张的脸，不照它们的帽子、衣服和话筒的位置。表情自然，不夸张，不露出一排牙齿的大笑。他右手拿着一支黑色无线话筒，话筒头在下巴下方，不挡住嘴。他站在舞台中央，正面，腰部以上，微笑着看镜头，没有在唱。暖金色的面光照在脸上，身后一束暖白色的逆光勾出东坡巾和肩膀的轮廓。背景是虚化的大屏，屏上是水墨山水和一轮明月。大型音乐综艺节目的官方舞台剧照，专业演唱会摄影，长焦镜头，浅景深，背景是大片柔和的光斑，电影感调色，层次丰富的舞台灯光，薄烟里看得见光束，皮肤质感真实，画面干净高级。画面里没有任何文字和标志。
```

换成你自己的角色：把「北宋文人苏东坡」、帽子、衣服、胡子的描述换掉，参考图换成你的角色的脸和服装照片。

## 2. 唱歌：图 + 音频生视频（MiniMax H3 / H3 Flash）

`<Picture 1>` 是脸部特写，`<Picture 2>` 是这个镜头的第一帧，`<Audio 1>` 是这几句的歌声。提示词分四块：

| 部分 | 写什么 |
|---|---|
| 哪张图管什么 | 用 `<Picture 1>` 的脸和造型；`<Picture 2>` 是第一帧，舞台、服装、景别和光保持一样 |
| 动作和情绪 | 这几句的情绪；每句歌词配一个手上的动作，一句一个，不要一直挥手 |
| 嘴 | 嘴唇跟着 `<Audio 1>` 的每个字，唱到 m、b、p 开头的字嘴要完全合上，写出这几句里有哪些这样的字，再写出歌词原文 |
| 镜头 | 一个连续镜头，镜头怎么动；不要推到脸部特写 |

模板（英文写给模型，中文是说明）：

```
Use <Picture 1> as the exact face, [脸和造型的特征] of the [人物].
<Picture 2> shows the first frame: [第一帧里的场景、服装、手里拿着什么]. Keep the same stage, costume, framing and light as <Picture 2>.
His free hand moves only now and then, each time a different small shape, never pointing, never waving above his shoulder.
He sings to the audience, [情绪]: on 「[歌词 1]」 [动作 1]; on 「[歌词 2]」 [动作 2]; on 「[歌词 3]」 [动作 3].
One continuous shot: the camera [镜头怎么动]. The shot never comes closer than it starts; no close-up of the face.
His lips follow every word of the vocal in <Audio 1>, press fully together on every M, B and P sound, on [这几句里 m/b/p 开头的字], and open on the vowels. The words are: [歌词原文]. Use <Audio 1> exactly as it is as the sound of the shot, no other sound, no subtitles or captions.
```

我们的原句（「怀民，开门！」那条）：

```
Use <Picture 1> as the exact face, moustache, long chin beard and tall black two-tier square scholar hat of the man. <Picture 2> shows the first frame: him on a concert stage in a light cream wide-sleeved robe with broad dark brown borders and a black silk sash, holding a black wireless handheld microphone just below his chin, never covering his mouth. Keep the same stage, costume, framing and light as <Picture 2>. His free hand moves only now and then, each time a different small shape, never pointing, never waving above his shoulder. His tall black hat keeps exactly the shape and look it has in the first frame, from the first frame to the last. He sings to the audience, bright and eager, like calling an old friend out to see the moon: on 「怀民，开门」 his free hand rises to chest height, fingers loosely curled with gaps between them, and he taps twice in the air with his bent knuckles, a small, light movement of the wrist, like knocking softly on a wooden door; on 「月亮都爬上墙」 he opens the hand, palm up, and shows the moon on the screen behind him; on 「再不出来」 he leans in with raised eyebrows and a grin and beckons with the open hand, fingers waving towards himself; on 「好景可不等人赏」 the open hand comes down and rests at his side. The big screen behind him keeps exactly the picture it shows in the first frame, the full moon over the river at night, from the first frame to the last. One continuous shot: the camera trucks slowly sideways at the same distance; it does not push in. The shot never comes closer than it starts; no close-up of the face.  His lips follow every word of the vocal in <Audio 1>, press fully together on every M, B and P sound, on 民, 门, 爬 and 不, and open on the vowels. The words are: 怀民，开门！月亮都爬上墙，再不出来，好景可不等人赏。 Use <Audio 1> exactly as it is as the sound of the shot, no other sound, no subtitles or captions.
```

## 3. 外景：不唱歌

外景不给歌声，删掉 `Audio 1`，提示词里写他不说话。我们的原句（江边）：

```
Use <Picture 1> as the exact face, moustache, long chin beard and tall black two-tier square scholar hat of the man. <Picture 2> shows the first frame: him on the muddy riverside path at dusk. Keep the same place, costume, framing and light as <Picture 2>. He never speaks or sings; his mouth stays closed. Realistic film, warm practical light, calm and lived-in, never eerie; nobody else appears. His tall black hat keeps exactly the shape and look it has in the first frame, from the first frame to the last. He walks slowly along the muddy riverside path at dusk, his robe hem flecked with mud, looking out over the river with a calm, faraway face. The camera tracks backward in front of him at the same pace, at the height of his chest, framing him from the knees up the whole time: his head and hat always stay in the frame and the camera never tilts down to his feet. No speech, no music score, no subtitles.
```
