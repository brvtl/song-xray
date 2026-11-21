// Pure mapping from the mixer state (mute / solo / synth / volume per row, stems ready or not) to the gain of every audio
// path. Kept in one place so the original mix and the stems can never play at the same time. DOM-free.
//
// Rows: "original" (the decoded file), one per track, and "residual" (what no track explains).
// With stems:     the mix is played as stems + residual (their sum is the original). Soloing "Original" plays the file
//                 itself instead (A/B); muting "Original" silences all recorded audio, leaving only synths.
// Without stems:  the file is played as one "Original" row; a track can only sound through its synth, so Synth or Solo
//                 on a track plays its synth (and Solo silences the original). Track mute only silences its synth.

/**
 * @typedef {{muted?: boolean, soloed?: boolean, volume?: number}} Row
 * @typedef {Row & {synth?: boolean}} TrackRow
 * @typedef {{original: number, residual: number, stems: number[], synths: number[], mode: string}} Gains
 */

/**
 * @param {{original: Row, residual: Row, tracks: TrackRow[], stemsReady: boolean}} state
 * @returns {Gains} linear gains per path; `mode` is a short label of what is audible
 */
export function mixGains({ original, residual, tracks, stemsReady }) {
  const vol = (r) => r.volume ?? 1;
  const anySolo = !!(original.soloed || residual.soloed || tracks.some((t) => t.soloed));
  const audible = (r) => !r.muted && (!anySolo || !!r.soloed);
  const out = { original: 0, residual: 0, stems: tracks.map(() => 0), synths: tracks.map(() => 0), mode: "" };

  if (stemsReady) {
    const abOriginal = !!original.soloed && !original.muted;
    const recorded = !original.muted && !abOriginal;
    if (abOriginal) out.original = vol(original);
    out.residual = recorded && audible(residual) ? vol(residual) * vol(original) : 0;
    tracks.forEach((t, k) => {
      out.stems[k] = recorded && !t.synth && audible(t) ? vol(t) * vol(original) : 0;
      out.synths[k] = t.synth && audible(t) ? vol(t) : 0;
    });
    const untouched =
      !anySolo &&
      !original.muted &&
      !residual.muted &&
      vol(residual) === 1 &&
      tracks.every((t) => !t.muted && !t.synth && vol(t) === 1);
    out.mode = abOriginal || untouched ? "Original mix" : "Stems";
  } else {
    out.original = audible(original) ? vol(original) : 0;
    tracks.forEach((t, k) => {
      out.synths[k] = (t.synth || t.soloed) && audible(t) ? vol(t) : 0;
    });
    out.mode = out.original > 0 ? "Original mix" : "Synth";
  }
  const synthOn = out.synths.some((g) => g > 0);
  if (synthOn && out.mode !== "Synth") out.mode += " + synth";
  if (!synthOn && out.original === 0 && out.residual === 0 && out.stems.every((g) => g === 0)) out.mode = "Silent";
  return out;
}
