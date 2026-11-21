// Read a file's own sample rate from its header. decodeAudioData resamples to the context rate and hides it, but the
// drum model only runs for sources recorded at >= 40 kHz, and decoding at the native rate avoids a resample.

const MPEG_RATES = {
  3: [44100, 48000, 32000], // MPEG1
  2: [22050, 24000, 16000], // MPEG2
  0: [11025, 12000, 8000], // MPEG2.5
};
// kbps by [version family][layer][index]; family 1 = MPEG1, 2 = MPEG2/2.5. Layer index: 3 = I, 2 = II, 1 = III.
const BITRATES = {
  1: {
    3: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    2: [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  },
  2: {
    3: [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    1: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
  },
};
const MP3_SCAN_BYTES = 256 * 1024;

const tag = (u8, i) => String.fromCharCode(u8[i], u8[i + 1], u8[i + 2], u8[i + 3]);

/** Sample rate from the fmt chunk of a RIFF/RF64 WAVE file, or null. */
function wavRate(u8, view) {
  if (u8.length < 12 || !["RIFF", "RF64"].includes(tag(u8, 0)) || tag(u8, 8) !== "WAVE") return null;
  let p = 12;
  while (p + 8 <= u8.length) {
    const id = tag(u8, p);
    const size = view.getUint32(p + 4, true);
    if (id === "fmt ") return p + 16 <= u8.length && size >= 16 ? view.getUint32(p + 12, true) || null : null;
    // RF64 stores 0xFFFFFFFF for big chunk sizes in the ds64 chunk; fmt always precedes data, so stop at data.
    if (id === "data" || size === 0xffffffff) return null;
    p += 8 + size + (size & 1);
  }
  return null;
}

/** Parse a 4-byte MPEG audio frame header at i: {rate, length} or null for an invalid / free-format header. */
function mpegFrame(u8, i) {
  if (i + 4 > u8.length || u8[i] !== 0xff || (u8[i + 1] & 0xe0) !== 0xe0) return null;
  const version = (u8[i + 1] >> 3) & 3;
  const layer = (u8[i + 1] >> 1) & 3;
  const brIdx = u8[i + 2] >> 4;
  const srIdx = (u8[i + 2] >> 2) & 3;
  if (version === 1 || layer === 0 || brIdx === 0 || brIdx === 15 || srIdx === 3) return null;
  const rate = MPEG_RATES[version][srIdx];
  const kbps = BITRATES[version === 3 ? 1 : 2][layer][brIdx];
  const pad = (u8[i + 2] >> 1) & 1;
  let length;
  if (layer === 3) length = (Math.floor((12000 * kbps) / rate) + pad) * 4;
  else if (layer === 1 && version !== 3) length = Math.floor((72000 * kbps) / rate) + pad;
  else length = Math.floor((144000 * kbps) / rate) + pad;
  return { rate, length };
}

/** Offset of the first byte after an ID3v2 tag at the start of the file (0 if none). */
function id3v2End(u8) {
  if (u8.length < 10 || u8[0] !== 0x49 || u8[1] !== 0x44 || u8[2] !== 0x33) return 0;
  const size = ((u8[6] & 127) << 21) | ((u8[7] & 127) << 14) | ((u8[8] & 127) << 7) | (u8[9] & 127);
  const footer = u8[5] & 0x10 ? 10 : 0;
  return 10 + size + footer;
}

/** Sample rate of the first MPEG frame that is confirmed by a second frame header right after it, or null. */
function mp3Rate(u8) {
  let start = id3v2End(u8);
  // Some files carry several ID3v2 tags back to back.
  while (start + 10 <= u8.length && id3v2End(u8.subarray(start)) > 0) start += id3v2End(u8.subarray(start));
  const end = Math.min(u8.length - 3, start + MP3_SCAN_BYTES);
  for (let i = start; i < end; i++) {
    const f = mpegFrame(u8, i);
    if (!f) continue;
    const next = i + f.length;
    if (next + 4 > u8.length) return f.rate; // the file ends after this frame
    const g = mpegFrame(u8, next);
    if (g && g.rate === f.rate) return f.rate;
  }
  return null;
}

/**
 * The sample rate a file was recorded at, read from its header without decoding.
 * WAV (RIFF / RF64, any fmt variant): the fmt chunk. MP3 (MPEG 1 / 2 / 2.5, layers I-III): the first frame header
 * confirmed by the following one, after any ID3v2 tag. Other formats: null.
 * @param {ArrayBuffer} arrayBuffer the file's bytes
 * @param {string} [filename] used only to prefer the MP3 scan for *.mp3 files that lack a RIFF header
 * @returns {number | null} samples per second, or null when unknown
 */
export function sniffSampleRate(arrayBuffer, filename = "") {
  const u8 = new Uint8Array(arrayBuffer);
  const view = new DataView(arrayBuffer);
  const isRiff = u8.length >= 4 && ["RIFF", "RF64"].includes(tag(u8, 0));
  if (isRiff) return wavRate(u8, view);
  const looksMp3 = /\.(mp3|mp2|mpga)$/i.test(filename) || id3v2End(u8) > 0 || (u8[0] === 0xff && u8[1] >= 0xe0);
  return looksMp3 ? mp3Rate(u8) : null;
}
