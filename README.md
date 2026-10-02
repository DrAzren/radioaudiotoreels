# SPACE atau AVOIDANCE? — radio interview → vertical reel

A 1080×1920 TikTok/Instagram Reel (~116 s) built from an audio-only Bahasa Melayu radio
interview (Kool FM) about whether to give a partner "space" during conflict, and when that space has
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

## Story structure

| Reel time | Beat | Visual approach |
|---|---|---|
| 0:00–0:07 | Cold-open hook (doctor: *"…bukan berapa lama… apa yang berlaku selepas space"*) | Freeze → couple splits apart → **SPACE? / ATAU AVOIDANCE?**, unanswered message |
| 0:06–0:22 | Host frames the question; walking out to kedai mamak / jogging | Title card, radio mic + ON AIR + live waveform, Malaysian B-roll |
| 0:22–0:33 | Why people walk away: avoid conflict, discomfort with confrontation | Fight/Flight/Freeze/**Withdraw**; Conflict → Overload → Withdraw → Temporary relief ≠ problem solved |
| 0:33–0:44 | Shutdown / emotionally overwhelmed / regulate emotion | Muffled-world blur, heartbeat, EMOTIONAL LOAD meter → **SHUT DOWN**; arousal curve → **REGULATE EMOSI** |
| 0:44–0:59 | Learned behaviour from childhood | Door-close → sepia flashback; Speak up → Get scolded → Stay quiet → Feels safer; child → adult match; *Sometimes avoidance is LEARNED.* |
| 0:57–1:07 | "Kena beza dua benda" | Two cards: needs time to regulate vs. avoids everything |
| 1:07–1:22 | "Berapa jam, berapa hari?" → no specific number → the key line | Clock; 30 MIN / 2 JAM / 1 HARI struck out; music dip; **BUKAN SEKADAR BERAPA LAMA → APA JADI SELEPAS ITU?** |
| 1:22–1:44 | Healthy space vs. avoidance / stonewalling | Split screen: Space → Regulate → Return vs. disappear, unread, blocked, 3 days, returns as if nothing happened; brick wall → **STONEWALLING** |
| 1:44–1:56 | Climax + takeaway | Split closes on the reconciled couple; **ADA TEMPOH · ADA JALAN UNTUK KEMBALI**; REGULATE. RETURN. COMMUNICATE. · *Save untuk rujukan* |

## Editorial rules followed

- Only original interview audio is used. Segments are selected and pauses tightened; no words are generated,
  re-ordered within a sentence or voice-cloned. The cold-open line is a flash-forward of the doctor's own
  sentence, which plays again in context at 1:17.
- Removed: station/guest intro, men-vs-women generalisations, a few redundant clauses and fillers.
- Captions correct obvious ASR errors (e.g. *tone wall → stonewall*, *bintang → bincang*). Two short
  phrases that could not be transcribed with confidence are left uncaptioned rather than guessed.
- No talking-head avatar or fake studio footage of the real host/doctor. The radio visuals are generic.
  All people shown are AI-generated illustrations, as credited on the end card.
- Childhood section shows a stern parent silhouette and a quiet child only. No physical abuse, no blame.
- Both men and women appear as the one who withdraws and the one who waits.

## Pipeline (`work/`)

1. `transcribe.py`: faster-whisper large-v3 word-level transcript of the source.
2. `edit_audio.py`: edit list → `voice_edit_raw.wav` + `edit_map.json` (cuts snapped to silence).
3. `build_captions.py`: remaps word timings through the edit, applies corrections, chunks to ≤2 lines.
4. B-roll: 17 photoreal stills generated with Canva AI image generation and exported at 1080×1920 (`work/broll/`).
5. `reel.py` + `gfx.py`: frame renderer (OpenCV/PIL). Ken Burns camera, grading, grain, kinetic type,
   motion graphics, transitions and word-highlight captions. `render_all.sh` renders in 4 parallel chunks.
6. `audio_mix.py`: voice clean-up (HPF, denoise, EQ, levelling, compression), synthesized score that
   follows the story arc, synthesized SFX, speech ducking and the music dip before the key line.
7. `master.sh`: two-pass loudness normalisation and mux. `make_docs.py` writes the SRT and EDL.

To re-run, put the original interview at `work/source.mp3` (it is not committed), then
`ffmpeg -i source.mp3 -ac 1 -ar 16000 src16k.wav` and `ffmpeg -i source.mp3 -ac 1 -ar 48000 src48m.wav`.

Requirements: Python 3.11, ffmpeg, `faster-whisper numpy scipy soundfile pillow opencv-python-headless fonttools brotli`.
Fonts (Anton, Montserrat, Inter, JetBrains Mono; SIL OFL) are in `work/fonts/`.
