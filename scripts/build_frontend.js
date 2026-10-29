/**
 * scripts/build_frontend.js
 * =========================
 * Precompiles JSX from src/app_source.jsx into static/app.js using Babel offline.
 * Replaces any need for in-browser @babel/standalone runtime.
 */

const fs = require("fs");
const path = require("path");

const BASE_DIR = path.resolve(__dirname, "..");
const BABEL_PATH = path.join(BASE_DIR, "babel_standalone.js");
const SOURCE_FILE = path.join(BASE_DIR, "src", "app_source.jsx");
const TARGET_FILE = path.join(BASE_DIR, "static", "app.js");

function build() {
  console.log("[*] Precompiling JSX to plain JavaScript (static/app.js)...");

  if (!fs.existsSync(BABEL_PATH)) {
    throw new Error(`Missing Babel standalone compiler at ${BABEL_PATH}. Run scripts/download_vendor_assets.py`);
  }

  const Babel = require(BABEL_PATH);
  const sourceCode = fs.readFileSync(SOURCE_FILE, "utf-8");

  const transformed = Babel.transform(sourceCode, {
    presets: [
      ["react", { runtime: "classic" }]
    ],
    comments: true,
    minified: false
  });

  fs.writeFileSync(TARGET_FILE, transformed.code, "utf-8");
  const size = fs.statSync(TARGET_FILE).size;
  console.log(`[SUCCESS] Precompiled static/app.js generated (${size} bytes).`);
}

if (require.main === module) {
  build();
}

module.exports = { build };
