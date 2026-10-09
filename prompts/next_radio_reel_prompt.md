# Prompt: radio interview → vertical reel (next episode)

Copy everything inside the box below into a new session, attach the interview file (MP4 or MP3), and fill in
every `[…]` field. Fields marked *(optional)* can be deleted; I will work them out from the audio.

---

```text
Transform the attached RADIO INTERVIEW into a PREMIUM, HIGH-RETENTION vertical TikTok / Instagram Reel.
Repository: DrAzren/radioaudiotoreels. Reuse the existing pipeline in work/ (see README.md) — same style,
same quality bar as the "SPACE atau AVOIDANCE?" reel.

────────────────────────────
0. INPUT: EXTRACT MP3 — USE ONLY THE MP3
────────────────────────────
The attachment may be an MP4/MOV video recording of the interview, or an MP3.
- If it is a video: extract the audio first with
    work/extract_mp3.sh "<attached file>" work/source.mp3
  (first audio stream, video/subtitle/data streams dropped, LAME VBR -q:a 0, original sample rate).
  Confirm the MP3 duration matches the video and report the audio codec, sample rate and channels.
- If it is already an MP3: the same script copies it unchanged to work/source.mp3.
- From then on use ONLY work/source.mp3. Do NOT use, show, crop, screenshot or reference any frame of
  the original video (no faces of the host/doctor, no studio shots, no station logos from the footage).
  Treat the episode exactly as audio-only.
- Do NOT commit the source MP3/MP4 to the repository.

────────────────────────────
EPISODE DETAILS
────────────────────────────
Language: Bahasa Melayu (keep natural English psychology terms the speakers use).
Station: [Kool FM / nama stesen]
Guest (show on screen exactly): DR FAKRUL AZREN BIN AZHAR — subtitle "Tetamu · Temu bual radio"
Topic / host's question: "[soalan utama hos, contoh: Bila pasangan …, adakah …?]"
Doctor's main educational message: [1–3 ayat mesej utama doktor]
Preferred hook line (optional): "[ayat terkuat doktor untuk cold open, jika sudah tahu]"
Title card options (optional): "[TAJUK 1]" / "[TAJUK 2]"
Key words to emphasise in captions (optional): [contoh: SPACE, KONFLIK, EMOSI, KEMBALI]
Things to avoid in this episode (optional): [contoh: generalisasi lelaki vs perempuan, …]

────────────────────────────
1. AUDIO EDIT (most important — no choppy cuts)
────────────────────────────
Transcribe the whole interview first (faster-whisper large-v3, word timestamps), then build the
strongest coherent version.
- Target 75–110 s; up to ~120 s only if needed to keep the doctor's explanation intact.
- Prioritise the doctor. Keep just enough of the host to set up the question.
- Select WHOLE sentences/phrases only. Never generate, re-order or clone speech; never make the doctor
  appear to say something he did not say.
- Cut placement rules (lessons from the last reel):
  • every cut must sit in the quietest 5 ms frame BETWEEN two words, preferring real silence (< −45 dB);
    never snap blindly — check the words on both sides of each cut;
  • no mid-sentence trims unless BOTH sides land in real silence; if not, keep the phrase;
  • shorten only pauses longer than ~0.42 s, down to ~0.30 s; keep natural breathing room;
  • fades scale with how loud the cut point is (≈6 ms in silence, 18–35 ms otherwise);
  • level-match each segment (±5 dB); no dynaudnorm and no FFT denoiser on a clean source;
    voice chain = HPF/LPF, light EQ, gentle compression (≈2.5:1), limiter.
- Verify: re-transcribe EVERY edited segment on its own AND inside the final mix. Report each segment's
  text; any clipped first/last word or masked phrase must be fixed before rendering.
- Remove filler, false starts, repeated ideas and station/guest introductions, but only at clean gaps.

────────────────────────────
2. STRUCTURE & VISUALS (same style as the previous reel)
────────────────────────────
- Cold-open hook in the first 1–3 s with the doctor's strongest real line (may repeat later in context),
  big kinetic title, subtle bass hit / whoosh. Then a short title card and the radio bridge (generic mic,
  ON AIR, live waveform — never fake footage of the real host/doctor, no talking-head avatar).
- Story arc built from what is actually said: hook → why it happens → psychology → examples →
  healthy vs unhealthy comparison (split screen) → key insight (music dip + impact) → final takeaway.
- Photoreal Malaysian B-roll: generate stills with Canva AI image generation (natural adults, modern
  Malaysian homes/streets, soft cinematic light, no melodrama, both men and women), export at 1080×1920.
  Ask me before saving any Canva design. Children only in gentle, non-abusive scenes.
- Motion graphics only where they improve understanding; pattern interrupt every 5–8 s.
- Name lower-third "DR FAKRUL AZREN BIN AZHAR / Tetamu · Temu bual radio" the first time the doctor is
  heard in context; small name tag on his other key lines; "HOS RADIO" tag for the host.
- Captions: Bahasa Melayu, max 2 lines, word highlight, key words larger/coloured; correct obvious ASR
  errors; leave genuinely unclear words UNCAPTIONED rather than guessing.
- NO END CARD and no spoken/visual CTA screen: end on the doctor's final takeaway line with large
  kinetic text, hold ~1.5 s, fade out.

────────────────────────────
3. SOUND
────────────────────────────
- Voices always first. Synthesized score that follows the arc (tense → minimal pulse → warmer →
  resolution), ducked ≈ −20 dB under speech; dip before the key line.
- SFX sparingly and never on top of an important word (check the word timings before placing them).
- Master −14 LUFS integrated, −1 dBTP.

────────────────────────────
4. DELIVERABLES
────────────────────────────
Put everything in output/[slug-episod]/:
- [slug]_1080x1920.mp4 — H.264 2-pass ~5 Mbps (keep under ~90 MB), AAC 256 kb/s, 30 fps
- preview_720p.mp4 (≤ 20 MB, send this one to me in the chat)
- captions_ms.srt, EDL.md (every reel segment → source timestamp), contact_sheet.jpg
Run a final QC (frame grid + re-transcription of the final file), then commit and push to the working
branch. Do not open a pull request unless I ask. Remind me to tick the platform's "AI-generated content"
label when posting (the visuals are AI illustrations).
```

---

### Nota (Bahasa Melayu)

- **Fail MP4:** lampirkan sahaja video. Langkah 0 akan mengekstrak audio ke `work/source.mp3`, dan hanya
  MP3 itu digunakan. Tiada imej daripada video asal akan muncul dalam reel.
- **Ekstrak sendiri (pilihan):** jalankan
  `work/extract_mp3.sh rakaman.mp4 rakaman.mp3`, kemudian lampirkan MP3 itu.
- **Medan wajib:** topik/soalan hos dan mesej utama doktor. Semua medan lain boleh dipadam, dan Claude
  akan menentukannya daripada audio.
