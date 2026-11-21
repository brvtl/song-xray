// Drop zone: drag & drop anywhere on the page, or click to browse through the real <input type="file" id="file-input">.
// Both paths call the same handler.

/**
 * @param {(file: File) => void} onFile
 * @returns {{reset: () => void}}
 */
export function initDropZone(onFile) {
  const zone = document.getElementById("drop-zone");
  const input = document.getElementById("file-input");
  input.addEventListener("change", () => {
    const f = input.files?.[0];
    if (f) onFile(f);
  });
  let depth = 0;
  const isFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!isFiles(e)) return;
    e.preventDefault();
    depth++;
    zone.classList.add("over");
  });
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) zone.classList.remove("over");
  });
  window.addEventListener("dragover", (e) => {
    if (!isFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("drop", (e) => {
    if (!isFiles(e)) return;
    e.preventDefault();
    depth = 0;
    zone.classList.remove("over");
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  });
  return {
    reset() {
      input.value = "";
    },
  };
}
