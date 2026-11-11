"""扒曲: a model's first guess at the instruments and the genre of a reference, per section, before the person is asked.

The agent running this skill cannot hear. Instrument and genre names come only from these taggers and from the person's ear,
so the card shows the tags under 「模型听到的（待人听核）」 and the person decides.
- AST (MIT/ast-finetuned-audioset-10-10-0.4593): 527 AudioSet labels, sigmoid scores, averaged over the windows of a section.
  It has instrument labels (Guitar, Mandolin, Zither, Pizzicato ...) and a few genre labels, but no 'Spanish' or 'flamenco'.
- CLAP zero-shot (laion/clap-htsat-unfused): the audio is compared with text labels; softmax over each list. The lists are a
  broad default set plus --extra-instruments / --extra-genres "label|label" written for this song. It only picks among the
  labels it is given.
- Results are guesses. On a phone recording with no lows or highs, a guitar tends to come out as mandolin or banjo. A skit or
  speech in a window pulls the genre towards 'comedy sketch'. Use the tags to ask the person a better question, not as answers.
- Sanity check: two different windows must give different CLAP audio embeddings. laion/larger_clap_music returned the same
  embedding for every window with transformers 4.51.3, so the script stops if that happens.
- Instruments are tagged on the backing stem when --backing is given (on the mix, a high voice came out as 'bamboo flute' in
  every sung section); the genre is tagged on the mix.
Sections come from analyze_ref.py's card (--card card.json); without it, 10 s windows across the song.
Sections shorter than 5 s, quieter than -45 dBFS, or more than 35 dB below the loudest section are skipped and listed:
a near-silent 3 s opening came out as 'flute' 0.96 and 'R&B' 0.93, which means nothing.
--plan prints which sections would be tagged and which skipped, without loading any model (numpy and ffmpeg only).
Writes <out>.json and <out>.txt, and with --card adds "model_tags" to the card.
Needs torch and transformers (any venv; models download from Hugging Face on first use: AST 346 MB, CLAP 614 MB). CPU is fine.
usage: tag_ref.py <song> [--backing stems/<name>_instrumental.flac] [--card card.json] [--out tags]
                  [--extra-instruments "a|b"] [--extra-genres "a|b"]
                  [--windows-per-section 3] [--device cpu] [--min-seconds 5] [--min-db -45] [--plan]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

AST_MODEL = 'MIT/ast-finetuned-audioset-10-10-0.4593'
CLAP_MODEL = 'laion/clap-htsat-unfused'
WIN = 10.
INSTRUMENTS = ['Spanish flamenco guitar', 'nylon-string classical guitar', 'steel-string acoustic guitar strumming', 'electric guitar',
               'mandolin', 'banjo', 'ukulele', 'harp', 'piano', 'accordion', 'violin and string section', 'brass section', 'saxophone',
               'flute', 'synthesizer', 'guzheng zither', 'pipa lute', 'erhu', 'Chinese bamboo flute', 'drum kit', 'hand percussion and cajon']
GENRES = ['flamenco rumba', 'Latin pop', 'bossa nova', 'tango', 'mariachi', 'salsa', 'reggae', 'reggaeton', 'country', 'folk', 'blues',
          'jazz', 'rock', 'pop ballad', 'hip hop rap', 'R&B', 'electronic dance music', 'J-pop', 'K-pop', 'Chinese folk',
          'Chinese pop ballad', 'Chinese opera', "children's song", 'comedy sketch with music', 'classical orchestral']
GENERIC = {'Music', 'Speech', 'Musical instrument', 'Singing', 'Song', 'Inside, small room', 'Silence', 'Vehicle'}


def load(path, sr):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.float32).copy()


def windows(a, z, n):
    """Up to n windows of WIN seconds spread inside [a, z]."""
    if z - a <= WIN:
        return [(a, z)]
    starts = np.linspace(a, z - WIN, n) if n > 1 else [a + (z - a - WIN) / 2]
    return [(float(s), float(s) + WIN) for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('song')
    ap.add_argument('--backing', help='the instrumental stem: instruments are tagged on it (a voice in the mix gets tagged as a flute)')
    ap.add_argument('--card')
    ap.add_argument('--out', default='tags')
    ap.add_argument('--extra-instruments', default='')
    ap.add_argument('--extra-genres', default='')
    ap.add_argument('--windows-per-section', type=int, default=3)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--min-seconds', type=float, default=5.)
    ap.add_argument('--min-db', type=float, default=-45.)
    ap.add_argument('--plan', action='store_true', help='only print which sections would be tagged; no models')
    a = ap.parse_args()

    x16 = load(a.song, 16000)
    dur = len(x16) / 16000
    card = json.load(open(a.card, encoding='utf-8')) if a.card else None
    if card:
        all_sections = [(float(s['start']), float(s['start']) + float(s['seconds'])) for s in card['sections']]
    else:
        all_sections = [(float(t), float(min(dur, t + WIN))) for t in np.arange(0, dur, WIN)]
    levels = [float(20 * np.log10(np.sqrt(np.mean(x16[int(s * 16000):int(e * 16000)] ** 2)) + 1e-9)) for s, e in all_sections]
    loudest = max(levels) if levels else 0.
    sections, skipped = [], []
    for (s0, s1), lv in zip(all_sections, levels):
        why = (f'{s1 - s0:.1f} s, shorter than {a.min_seconds:g} s' if s1 - s0 < a.min_seconds else
               f'{lv:.1f} dBFS, quieter than {a.min_db:g}' if lv < a.min_db else
               f'{lv - loudest:.1f} dB below the loudest section' if lv < loudest - 35 else None)
        (skipped.append({'start': round(s0, 1), 'end': round(s1, 1), 'level_db': round(lv, 1), 'why': why}) if why
         else sections.append((s0, s1)))
    for s in skipped:
        print(f"  skip {s['start']:>6}-{s['end']:<6} s  ({s['why']})")
    if a.plan:
        for (s0, s1), lv in zip(all_sections, levels):
            if (s0, s1) in sections:
                print(f'  tag  {s0:>6.1f}-{s1:<6.1f} s  {lv:.1f} dBFS, windows ' + ', '.join(f'{w0:.1f}-{w1:.1f}' for w0, w1 in windows(s0, s1, a.windows_per_section)))
        return
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification, ClapModel, ClapProcessor

    x48 = load(a.song, 48000)
    b16, b48 = (load(a.backing, 16000), load(a.backing, 48000)) if a.backing else (x16, x48)
    instruments = INSTRUMENTS + [s.strip() for s in a.extra_instruments.split('|') if s.strip()]
    genres = GENRES + [s.strip() for s in a.extra_genres.split('|') if s.strip()]

    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    ast = ASTForAudioClassification.from_pretrained(AST_MODEL).to(a.device).eval()
    pr = ClapProcessor.from_pretrained(CLAP_MODEL)
    clap = ClapModel.from_pretrained(CLAP_MODEL).to(a.device).eval()
    with torch.no_grad():
        def text_emb(labels):
            t = pr.tokenizer(labels, padding=True, return_tensors='pt').to(a.device)
            e = clap.get_text_features(**t)
            return e / e.norm(dim=-1, keepdim=True)
        ti, tgn = text_emb(instruments), text_emb(genres)

        def audio_emb(s, e, x=x48):
            f = pr.feature_extractor([x[int(s * 48000):int(e * 48000)]], sampling_rate=48000, return_tensors='pt').to(a.device)
            v = clap.get_audio_features(**f)
            return v / v.norm(dim=-1, keepdim=True)

        # sanity check: two different stretches must not give the same embedding
        p, q = audio_emb(0, min(dur, WIN)), audio_emb(max(0, dur / 2), min(dur, dur / 2 + WIN))
        same = float((p * q).sum())
        if same > .999:
            raise SystemExit(f'CLAP gives the same audio embedding for different windows (cosine {same:.4f}): its audio branch is not '
                             f'working with this transformers version; use {CLAP_MODEL}')
        rows = []
        for s0, s1 in sections:
            ast_p, ci, cg = [], [], []
            for w0, w1 in windows(s0, s1, a.windows_per_section):
                f = fe(b16[int(w0 * 16000):int(w1 * 16000)], sampling_rate=16000, return_tensors='pt').to(a.device)
                ast_p.append(torch.sigmoid(ast(**f).logits)[0].cpu().numpy())
                vb = audio_emb(w0, w1, b48)
                ci.append(torch.softmax(100 * vb @ ti.T, dim=-1)[0].cpu().numpy())
                v = audio_emb(w0, w1)
                cg.append(torch.softmax(100 * v @ tgn.T, dim=-1)[0].cpu().numpy())
            ap_, ci_, cg_ = np.mean(ast_p, 0), np.mean(ci, 0), np.mean(cg, 0)
            labels = ast.config.id2label
            ast_top = [(labels[int(k)], round(float(ap_[k]), 2)) for k in np.argsort(ap_)[::-1] if labels[int(k)] not in GENERIC][:a.top]
            rows.append({'start': round(s0, 1), 'end': round(s1, 1),
                         'ast': ast_top,
                         'clap_instruments': [(instruments[k], round(float(ci_[k]), 2)) for k in np.argsort(ci_)[::-1][:3]],
                         'clap_genres': [(genres[k], round(float(cg_[k]), 2)) for k in np.argsort(cg_)[::-1][:3]]})
    out = {'models': {'ast': AST_MODEL, 'clap': CLAP_MODEL}, 'note': 'guesses for the person to check by ear',
           'instruments_tagged_on': 'backing stem' if a.backing else 'mix (a voice may be tagged as a flute)',
           'clap_window_cosine_check': round(same, 4), 'sections': rows, 'skipped': skipped}
    Path(a.out + '.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['模型听到的（待人听核）', f'  模型：AST {AST_MODEL}（AudioSet 标签，分数是概率）；CLAP {CLAP_MODEL}（只在给它的候选里选，分数是 softmax）',
             '  这些是猜测：手机录音、没有低音和高音时，吉他常被听成曼陀林或班卓琴；有对白的段落会被听成喜剧小品。',
             '  乐器在' + ('伴奏层上听（AST 和 CLAP 乐器）' if a.backing else '混音上听：人声常被当成笛子，最好给 --backing 伴奏层') + '；风格在混音上听。']
    for r in rows:
        lines.append(f"  {r['start']:>6}-{r['end']:<6} s  AST: " + ', '.join(f'{l} {s}' for l, s in r['ast'][:5]))
        lines.append('                   CLAP 乐器: ' + ', '.join(f'{l} {s}' for l, s in r['clap_instruments'])
                     + ' | 风格: ' + ', '.join(f'{l} {s}' for l, s in r['clap_genres']))
    for s in skipped:
        lines.append(f"  {s['start']:>6}-{s['end']:<6} s  没打标签（{s['why']}）")
    Path(a.out + '.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    if card is not None:
        card['model_tags'] = out
        card.setdefault('listen', []).append('instrument and genre names: the model tags under model_tags are guesses; ask the person')
        Path(a.card).write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
