import { readFile } from "node:fs/promises";

const files = ["package.json", "package-lock.json", "openclaw.plugin.json"];
const documents = await Promise.all(
  files.map(async (path) => JSON.parse(
    await readFile(new URL(`../${path}`, import.meta.url), "utf8"),
  )),
);
const versions = documents.map((document) => document.version);
const expected = versions[0];

if (!expected || versions.some((version) => version !== expected)) {
  throw new Error(
    `OpenClaw plugin version mismatch: ${files.map(
      (path, index) => `${path}=${versions[index]}`,
    ).join(", ")}`,
  );
}

const lockRootVersion = documents[1]?.packages?.[""]?.version;
if (lockRootVersion !== expected) {
  throw new Error(
    `package-lock root version mismatch: ${lockRootVersion} != ${expected}`,
  );
}

const packageJson = documents[0];
const packageLock = documents[1];
const minimumOpenClaw = ">=2026.8.2";
const metadataMinimums = [
  packageJson?.openclaw?.compat?.pluginApi,
  packageJson?.peerDependencies?.openclaw,
  packageLock?.packages?.[""]?.peerDependencies?.openclaw,
];
if (metadataMinimums.some((value) => value !== minimumOpenClaw)) {
  throw new Error(
    `OpenClaw minimum mismatch: expected ${minimumOpenClaw}, got ${metadataMinimums.join(", ")}`,
  );
}
if (packageJson?.openclaw?.compat?.minGatewayVersion !== "2026.8.2") {
  throw new Error("OpenClaw gateway minimum must be 2026.8.2");
}

const lockedOpenClaw = packageLock?.packages?.["node_modules/openclaw"]?.version;
const buildOpenClaw = packageJson?.openclaw?.build;
if (!lockedOpenClaw || buildOpenClaw?.openclawVersion !== lockedOpenClaw ||
    buildOpenClaw?.pluginSdkVersion !== lockedOpenClaw) {
  throw new Error(
    `OpenClaw build metadata must match lock: build=${JSON.stringify(buildOpenClaw)} lock=${lockedOpenClaw}`,
  );
}

console.log(
  `OpenClaw plugin versions consistent: ${expected}; SDK lock: ${lockedOpenClaw}`,
);
