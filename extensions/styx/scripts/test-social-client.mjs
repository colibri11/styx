import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { readFile } from "node:fs/promises";

import { createStyxClient } from "../dist/src/client.js";

const manifest = JSON.parse(await readFile(
  new URL("../openclaw.plugin.json", import.meta.url),
  "utf8",
));
assert.equal(manifest.configSchema.properties.socialToken.type, "string");
assert.equal(manifest.uiHints.socialToken.sensitive, true);

const calls = [];
const fetchImpl = async (url, init) => {
  calls.push({
    url: String(url),
    method: init?.method,
    headers: Object.fromEntries(new Headers(init?.headers).entries()),
    body: init?.body ? JSON.parse(String(init.body)) : null,
  });
  return new Response("{}", {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};

const client = createStyxClient({
  baseUrl: "http://styx.test/",
  httpToken: "ordinary-http-token",
  socialToken: "separate-social-token",
  fetchImpl,
});

// Creating the client and using an ordinary lifecycle method never infers or
// emits a social act.  Social writes remain explicit host calls.
assert.equal(calls.length, 0);
await client.agentInitialize({ agent_id: "agent-a" });
assert.equal(calls.length, 1);
assert.equal(new URL(calls[0].url).pathname, "/agent/initialize");
assert.equal(calls[0].headers.authorization, "Bearer ordinary-http-token");
assert.equal(calls[0].headers["x-styx-social-token"], undefined);
assert.equal("socialClassify" in client, false);
assert.equal("socialAutoAttest" in client, false);

const hash = "a".repeat(64);
const actorA = "00000000-0000-4000-8000-000000000001";
const actorB = "00000000-0000-4000-8000-000000000002";
const scope = "00000000-0000-4000-8000-000000000003";
const act = "00000000-0000-4000-8000-000000000004";
const evidence = "00000000-0000-4000-8000-000000000005";

const explicitCalls = [
  ["/social/actors", () => client.socialActor({
    agent_id: "agent-a",
    identity_namespace: "local",
    actor_key: "actor-a",
    actor_kind: "local_agent",
    identity_evidence_hash: hash,
  })],
  ["/social/scopes", () => client.socialScope({
    agent_id: "agent-a",
    scope_key: "scope-a",
    protocol_id: "protocol-a",
    protocol_version: "1",
    policy_hash: hash,
  })],
  ["/social/encounters", () => client.socialEncounter({
    agent_id: "agent-a",
    encounter_key: "encounter-a",
    scope_id: scope,
    observer_actor_id: actorA,
    encountered_actor_id: actorB,
    direction: "inbound",
    channel_kind: "openclaw",
    source_act_id: act,
    evidence_hash: hash,
    confidence: 0.8,
  })],
  ["/social/attestations", () => client.socialAttestation({
    agent_id: "agent-a",
    scope_id: scope,
    issuer_actor_id: actorA,
    subject_actor_id: actorB,
    attestation_key: "attestation-a",
    attestation_kind: "direct",
    verdict: "undetermined",
    protocol_id: "protocol-a",
    protocol_version: "1",
    source_act_id: act,
    evidence_refs: [],
    trust_level: "unverified",
  })],
  ["/social/attestations/revise", () => client.socialRevise({
    agent_id: "agent-a",
    scope_id: scope,
    issuer_actor_id: actorA,
    subject_actor_id: actorB,
    attestation_key: "attestation-revision-a",
    attestation_kind: "direct",
    verdict: "negative",
    protocol_id: "protocol-a",
    protocol_version: "1",
    source_act_id: act,
    evidence_refs: [],
    trust_level: "verified",
    supersedes_attestation_id: evidence,
  })],
  ["/social/scopes/dissolve", () => client.socialDissolve({
    agent_id: "agent-a",
    scope_id: scope,
  })],
  ["/social/grants", () => client.socialGrant({
    agent_id: "agent-a",
    grant_key: "grant-a",
    scope_id: scope,
    grantee_principal_id: "principal-b",
    capability: "social:read",
    evidence_class: "attestation",
    evidence_id: evidence,
  })],
  ["/social/grants/revoke", () => client.socialGrantRevoke({
    agent_id: "agent-a",
    revocation_key: "grant-a-revoke",
    grant_id: evidence,
  })],
  ["/social/query", () => client.socialQuery({
    agent_id: "agent-a",
    scope_id: scope,
    actor_a_id: actorA,
    actor_b_id: actorB,
  })],
  ["/social/explain", () => client.socialExplain({
    agent_id: "agent-a",
    scope_id: scope,
  })],
  ["/social/deliver", () => client.socialDeliver({
    agent_id: "agent-a",
    delivery_key: "delivery-a",
    scope_id: scope,
    evidence_class: "attestation",
    evidence_id: evidence,
    receiving_agent_id: "agent-b",
  })],
];

for (const [path, invoke] of explicitCalls) {
  const before = calls.length;
  await invoke();
  assert.equal(calls.length, before + 1);
  const call = calls.at(-1);
  assert.equal(new URL(call.url).pathname, path);
  assert.equal(call.method, "POST");
  assert.equal(call.headers.authorization, "Bearer ordinary-http-token");
  assert.equal(call.headers["x-styx-social-token"], "separate-social-token");
  if (
    (path === "/social/attestations" || path === "/social/attestations/revise")
    && call.body.trust_level === "verified"
  ) {
    assert.equal(
      call.headers["x-styx-social-signature"],
      createHmac("sha256", "separate-social-token")
        .update(JSON.stringify(call.body), "utf8")
        .digest("hex"),
    );
  } else {
    assert.equal(call.headers["x-styx-social-signature"], undefined);
  }
  assert.equal(call.headers["x-wrap-for-llm"], undefined);
  assert.equal(JSON.stringify(call.body).includes("separate-social-token"), false);
  assert.equal(JSON.stringify(call.body).includes("ordinary-http-token"), false);
}

assert.deepEqual(calls.at(-1).body, {
  agent_id: "agent-a",
  delivery_key: "delivery-a",
  scope_id: scope,
  evidence_class: "attestation",
  evidence_id: evidence,
  receiving_agent_id: "agent-b",
});

// Missing social config is not silently replaced by the ordinary bearer.
const headersWithoutSocial = [];
const noSocialClient = createStyxClient({
  baseUrl: "http://styx.test",
  httpToken: "ordinary-only",
  fetchImpl: async (_url, init) => {
    headersWithoutSocial.push(
      Object.fromEntries(new Headers(init?.headers).entries()),
    );
    return new Response("{}", { status: 200 });
  },
});
await noSocialClient.socialExplain({ agent_id: "agent-a", scope_id: scope });
assert.equal(headersWithoutSocial[0].authorization, "Bearer ordinary-only");
assert.equal(headersWithoutSocial[0]["x-styx-social-token"], undefined);
