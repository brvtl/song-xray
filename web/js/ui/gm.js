// General MIDI names used by the UI and the synths. DOM-free.

/** GM percussion key names (channel 10), short labels for the drum lane rows. */
export const GM_DRUMS = {
  27: "high Q",
  28: "slap",
  31: "sticks",
  33: "metronome",
  35: "kick 2",
  36: "kick",
  37: "side stick",
  38: "snare",
  39: "clap",
  40: "snare 2",
  41: "low floor tom",
  42: "closed hat",
  43: "floor tom",
  44: "pedal hat",
  45: "low tom",
  46: "open hat",
  47: "low-mid tom",
  48: "high-mid tom",
  49: "crash",
  50: "high tom",
  51: "ride",
  52: "china",
  53: "ride bell",
  54: "tambourine",
  55: "splash",
  56: "cowbell",
  57: "crash 2",
  58: "vibraslap",
  59: "ride 2",
  60: "high bongo",
  61: "low bongo",
  62: "mute conga",
  63: "open conga",
  64: "low conga",
  69: "cabasa",
  70: "maracas",
  75: "claves",
  76: "high block",
  77: "low block",
};

/** Name of a GM drum key ("kick", "closed hat", ...), or "drum N" for keys without a common name. */
export function drumName(key) {
  return GM_DRUMS[key] ?? `drum ${key}`;
}

/** Kind of drum sound for a GM key: kick, snare, clap, hat, openhat, cymbal, tom, or other. */
export function drumKind(key) {
  if (key === 35 || key === 36) return "kick";
  if (key === 38 || key === 40) return "snare";
  if (key === 39) return "clap";
  if (key === 42 || key === 44 || key === 54 || key === 69 || key === 70) return "hat";
  if (key === 46) return "openhat";
  if ([49, 51, 52, 53, 55, 57, 59].includes(key)) return "cymbal";
  if ([41, 43, 45, 47, 48, 50].includes(key)) return "tom";
  return "other";
}

/**
 * Synth family for a GM program (0-127).
 * @returns {"keys"|"organ"|"bass"|"strings"|"lead"|"default"}
 */
export function programFamily(program) {
  const p = program | 0;
  if (p < 16 || (p >= 24 && p < 32) || (p >= 104 && p < 112)) return "keys"; // pianos, chromatic perc, guitars, ethnic
  if (p < 24) return "organ";
  if (p < 40) return "bass";
  if (p < 56 || (p >= 88 && p < 96)) return "strings"; // strings, ensembles, synth pads
  if (p < 88) return "lead"; // brass, reeds, pipes, synth leads
  return "default";
}

/** "electric_bass" / "electric bass" -> "Electric bass". */
export function displayName(name) {
  const s = String(name ?? "")
    .replaceAll("_", " ")
    .trim();
  return s ? s[0].toUpperCase() + s.slice(1) : "Track";
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/** MIDI pitch -> scientific pitch name (60 -> "C4"). */
export function pitchName(p) {
  return `${NOTE_NAMES[((p % 12) + 12) % 12]}${Math.floor(p / 12) - 1}`;
}
