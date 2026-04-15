// Handles CJS/ESM split: main files -> .cjs, preload stays .js, shared duplicated as both.
// Electron's sandbox loader only resolves .js for preload scripts.
import { readdirSync, renameSync, copyFileSync, readFileSync, writeFileSync, existsSync } from "fs";
import { join, basename, dirname } from "path";

const distMain = new URL("../dist/main", import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1");

function walk(dir) {
  const entries = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) entries.push(...walk(full));
    else entries.push(full);
  }
  return entries;
}

const allJsFiles = walk(distMain).filter((f) => f.endsWith(".js"));

const KEEP_JS = new Set(["preload.js"]);
const sharedDir = join(distMain, "shared");

const mainFiles = [];
const sharedFiles = [];
const preloadFiles = [];

for (const f of allJsFiles) {
  if (KEEP_JS.has(basename(f))) {
    preloadFiles.push(f);
  } else if (f.startsWith(sharedDir)) {
    sharedFiles.push(f);
  } else {
    mainFiles.push(f);
  }
}

for (const file of mainFiles) {
  let content = readFileSync(file, "utf8");
  content = content.replace(/require\("(\.[^"]+?)"\)/g, (match, p1) => {
    if (p1.endsWith(".json") || p1.endsWith(".html") || p1.endsWith(".node")) return match;
    const target = p1.split("/").pop();
    if (KEEP_JS.has(target + ".js")) return match;
    return `require("${p1}.cjs")`;
  });
  writeFileSync(file, content, "utf8");
}

for (const file of sharedFiles) {
  const cjsPath = file.replace(/\.js$/, ".cjs");
  copyFileSync(file, cjsPath);
}

for (const file of mainFiles) {
  renameSync(file, file.replace(/\.js$/, ".cjs"));
}

console.log(
  `Main: ${mainFiles.length} renamed to .cjs | ` +
  `Shared: ${sharedFiles.length} duplicated (.js + .cjs) | ` +
  `Preload: ${preloadFiles.length} kept as .js`
);
