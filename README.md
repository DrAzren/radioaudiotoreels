# SPACE atau AVOIDANCE? — radio interview → vertical reel

A 1080×1920 TikTok/Instagram Reel (~2:04) built from an audio-only Bahasa Melayu radio
interview (Kool FM) with **Dr Fakrul Azren bin Azhar** about whether to give a partner "space" during conflict, and when that space has
become avoidance, silent treatment or stonewalling.

**Central message:** *Dalam konflik, persoalannya bukan semata-mata berapa lama kita ambil space —
persoalannya: selepas kita tenang, adakah kita kembali untuk berbincang?*

## Deliverables (`output/`)

| File | What it is |
|---|---|
| `space_atau_avoidance_reel_1080x1920.mp4` | Final reel, H.264 + AAC, 30 fps, −14 LUFS / −1 dBTP |
| `captions_ms.srt` | Caption track (Bahasa Melayu, corrected) for platform upload |
| `EDL.md` | Edit decision list: every reel segment mapped to its source timestamp |
| `contact_sheet.jpg` | Frame grid for quick review |
| `preview_720p.mp4` | Small 720p preview (~20 MB) for sharing/review only |

## Story structure

| Reel time | Beat | Visual approach |
|---|---|---|
| 0:00–0:07 | Cold-open hook (doctor: *"…bukan berapa lama… apa yang berlaku selepas space"*) | Freeze → couple splits apart → **SPACE? / ATAU AVOIDANCE?**, unanswered message |
| 0:06–0:22 | Host frames the question; walking out to kedai mamak / jogging | Title card, radio mic + ON AIR + live waveform, Malaysian B-roll |
| 0:22–0:36 | Why people walk away (name lower-third: Dr Fakrul Azren bin Azhar): avoid conflict, discomfort with confrontation | Fight/Flight/Freeze/**Withdraw**; Conflict → Overload → Withdraw → Temporary relief ≠ problem solved |
| 0:36–0:47 | Shutdown / emotionally overwhelmed / regulate emotion | Muffled-world blur, heartbeat, EMOTIONAL LOAD meter → **SHUT DOWN**; arousal curve → **REGULATE EMOSI** |
| 0:47–1:02 | Learned behaviour from childhood | Door-close → sepia flashback; Speak up → Get scolded → Stay quiet → Feels safer; child → adult match; *Sometimes avoidance is LEARNED.* |
| 1:02–1:12 | "Kena beza dua benda" | Two cards: needs time to regulate vs. avoids everything |
| 1:12–1:31 | "Berapa jam, berapa hari?" → no specific number → the key line | Clock; 30 MIN / 2 JAM / 1 HARI struck out; music dip; **BUKAN SEKADAR BERAPA LAMA → APA JADI SELEPAS ITU?** |
| 1:32–1:54 | Healthy space vs. avoidance / stonewalling | Split screen: Space → Regulate → Return vs. disappear, unread, blocked, 3 days, returns as if nothing happened; brick wall → **STONEWALLING** |
| 1:54–2:04 | Climax + takeaway | Split closes on the reconciled couple; **SPACE YANG SIHAT · ADA TEMPOH · ADA JALAN UNTUK KEMBALI**, fade out (no end card) |

## Editorial rules followed

- Only original interview audio is used. Segments are selected and long pauses gently shortened; no words are
  generated, re-ordered within a sentence or voice-cloned. Every cut is placed in a gap between words (checked at
  5 ms resolution and by re-transcribing each segment), so no word is clipped. The cold-open line is a flash-forward of the doctor's own
  sentence, which plays again in context at 1:26.
- Removed: station/guest intro and men-vs-women generalisations (whole sentences only).
- Captions correct obvious ASR errors (e.g. *tone wall → stonewall*, *bintang → bincang*). Two short
  phrases that could not be transcribed with confidence are left uncaptioned rather than guessed.
- No talking-head avatar or fake studio footage of the real host/doctor. The radio visuals are generic.
  All people shown are AI-generated illustrations; tick the platform's "AI-generated content" label when posting.
- Childhood section shows a stern parent silhouette and a quiet child only. No physical abuse, no blame.
- Both men and women appear as the one who withdraws and the one who waits.

## Next episode

Use [`prompts/next_radio_reel_prompt.md`](prompts/next_radio_reel_prompt.md). Attach an MP4 or MP3; video files are
converted with `work/extract_mp3.sh <video> work/source.mp3` and only the MP3 is used.

## Pipeline (`work/`)

1. `transcribe.py`: faster-whisper large-v3 word-level transcript of the source.
2. `edit_audio.py`: edit list → `voice_edit_raw.wav` + `edit_map.json` (cuts placed between words, per-segment level match).
3. `build_captions.py`: remaps word timings through the edit, applies corrections, chunks to ≤2 lines.
4. B-roll: 17 photoreal stills generated with Canva AI image generation and exported at 1080×1920 (`work/broll/`).
5. `reel.py` + `gfx.py` + `warp.py`: frame renderer (OpenCV/PIL). Ken Burns camera, grading, grain, kinetic type,
   motion graphics, transitions and word-highlight captions. Scene cues are authored on the first-cut timeline
   (`captions_v1.json`) and `warp.py` re-maps them word-by-word onto the current edit. `render_all.sh` renders in 4 parallel chunks.
6. `audio_mix.py`: voice clean-up (HPF/LPF, EQ, gentle compression, limiter), synthesized score that
   follows the story arc, synthesized SFX, speech ducking and the music dip before the key line.
7. `master.sh`: two-pass loudness normalisation and mux. `make_docs.py` writes the SRT and EDL.

To re-run, put the original interview at `work/source.mp3` (it is not committed), then
`ffmpeg -i source.mp3 -ac 1 -ar 16000 src16k.wav` and `ffmpeg -i source.mp3 -ac 1 -ar 48000 src48m.wav`.

Requirements: Python 3.11, ffmpeg, `faster-whisper numpy scipy soundfile pillow opencv-python-headless fonttools brotli`.
Fonts (Anton, Montserrat, Inter, JetBrains Mono; SIL OFL) are in `work/fonts/`.

---

# Rejection & cemburu — second episode (`output/rejection_cemburu/`)

A ~1:48 reel from the radio interview `Rejection.mp4` (audio only; extracted with `work/extract_mp3.sh`) with
**Dr Fakrul Azren bin Azhar** on why rejection and jealousy hit so hard together, and how self-worth placed in a
relationship collapses when the partner leaves.

**Central message:** *Bukan semua benda kita boleh kawal dalam kehidupan kita. It's all about kita — fokus kepada
kita punya emosi, pemikiran, tingkah laku kita. Itu yang penting sebenarnya.*

| Reel time | Beat | Visual approach |
|---|---|---|
| 0:00–0:05 | Cold open (doctor): *"…bila pasangan tinggalkan kita, timbullah pemikiran apa yang kurang pada diri aku"* | Bed at night → mirror; **APA YANG KURANG / PADA DIRI AKU?** |
| 0:05–0:15 | Title, radio bridge, host: *"How to handle rejection?"* / *"kenapa rejection dan cemburu… kombo emosi yang sangat kuat?"* | Title card; mic + ON AIR + live waveform; **REJECTION + CEMBURU = KOMBO EMOSI** |
| 0:15–0:29 | Why (name lower-third): both touch something close to the self — diterima, disayangi, selamat | Rings around **DIRI KITA**; couple + need pills |
| 0:29–0:51 | Rejection → kehilangan/ditolak; cemburu → takut ditinggalkan, dibandingkan, tak cukup penting; both at once | Platform, café list, collision **SERENTAK**, **SEDIH · MARAH · MALU · TAKUT** + heartbeat |
| 0:51–1:13 | Host: *"Nak ajar anak-anak untuk menerima rejection."* Doctor: upbringing, **nilai diri**, most of it placed in the relationship | Mother/daughter, father/son; self-worth bar filling to 80 % **RELATIONSHIP** |
| 1:13–1:30 | Loved → *cukup baik, selamat*; left → *apa yang kurang pada diri aku* | Ticks over couple; desaturated mirror |
| 1:30–1:34 | Key insight (music dip + impact): **BUKAN SEMUA BENDA KITA BOLEH KAWAL.** | Kinetic type on black |
| 1:34–1:40 | Split screen: ✗ bergantung pada pasangan (expect them to stay forever) vs ✓ dalam kawalan kita | Phone face-down vs journaling; healthy side takes over |
| 1:40–1:48 | Takeaway: **IT'S ALL ABOUT KITA → EMOSI · PEMIKIRAN · TINGKAH LAKU → ITU YANG PENTING.** Hold, fade (no end card) | Woman walking in a park at sunrise |

Files: `rejection_cemburu_1080x1920.mp4`, `preview_720p.mp4`, `captions_ms.srt`, `EDL.md`, `contact_sheet.jpg`.
Pipeline: `work/rejection/` (`edit_audio.py` → `transcribe_edit.py` → `build_captions.py` → `reel.py`/`render_all.sh`
→ `audio_mix.py` → `master.sh`, `make_docs.py`; cut-planning helpers `words.py`, `contour.py`, `verify_segments.py`).
B-roll: 13 Canva AI stills in `work/rejection/broll/` (+ the generic studio mic). All people shown are AI-generated;
tick the platform's "AI-generated content" label when posting.
