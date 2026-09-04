// Langfuse instrumentation: this file is imported first (see index.js) so
// the OpenTelemetry SDK is registered before anything else runs. Every LLM
// call, tool call, and thought/reminder push in the app is manually
// instrumented (no LangChain/OpenAI auto-instrumentation) via the small
// helpers exported below, so call sites never touch the Langfuse SDK
// directly and never need to branch on whether tracing is configured.
import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { startObservation } from "@langfuse/tracing";
import { logger } from "./logger.js";

export const langfuseEnabled = Boolean(
  process.env.LANGFUSE_PUBLIC_KEY && process.env.LANGFUSE_SECRET_KEY
);

let sdk = null;

if (langfuseEnabled) {
  // Reads LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_BASE_URL from
  // the environment itself; no options needed here.
  const langfuseSpanProcessor = new LangfuseSpanProcessor();
  sdk = new NodeSDK({ spanProcessors: [langfuseSpanProcessor] });
  sdk.start();
  logger.info(
    { baseUrl: process.env.LANGFUSE_BASE_URL ?? "http://localhost:3000" },
    "langfuse tracing enabled"
  );
} else {
  logger.warn(
    "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set; Langfuse tracing disabled"
  );
}

// Stand-in for every observation type (span/generation/event/tool/...) when
// tracing is disabled, or as a defensive fallback if the SDK ever throws.
// Every call site chains .update()/.end()/.startObservation() on whatever
// startTrace/startChild hand back without caring whether tracing is
// actually on — the noop just swallows all of it.
const noop = {
  update: () => noop,
  end: () => {},
  startObservation: () => noop,
};

// Starts a brand-new Langfuse trace (a root observation with no parent).
// Every top-level entry point — a /chat request, a thought-loop tick, a
// reminder delivery — calls this exactly once; everything else nests under
// it via startChild.
export function startTrace(name, attributes = {}) {
  if (!langfuseEnabled) return noop;
  try {
    return startObservation(name, attributes, { asType: "span" });
  } catch (err) {
    logger.debug({ err }, "langfuse startTrace failed");
    return noop;
  }
}

// Starts a child observation under `parent`. If `parent` is nullish (a
// call site with no trace in scope, e.g. a detached background job), this
// starts a new root trace instead of dropping the observation — every LLM
// call should be visible in Langfuse even when it isn't part of a larger
// flow. asType is one of Langfuse's observation types: "span" (default),
// "generation" (LLM calls), "embedding" (embedding calls), "tool" (tool
// calls), or "event" (point-in-time, e.g. a push to the frontend).
export function startChild(parent, name, attributes = {}, asType = "span") {
  try {
    if (parent) return parent.startObservation(name, attributes, { asType });
    if (!langfuseEnabled) return noop;
    return startObservation(name, attributes, { asType });
  } catch (err) {
    logger.debug({ err }, "langfuse startChild failed");
    return noop;
  }
}

// Updates and ends an observation on success. Safe to call on anything
// startTrace/startChild returned, including the noop stand-in.
export function endOk(observation, updates = {}) {
  try {
    observation?.update(updates)?.end();
  } catch (err) {
    logger.debug({ err }, "langfuse endOk failed");
  }
}

// Same as endOk, but marks the observation as failed. err's message is
// attached as statusMessage; the observation itself never throws Langfuse
// errors back into application logic.
export function endError(observation, err, updates = {}) {
  try {
    observation
      ?.update({ ...updates, level: "ERROR", statusMessage: err?.message ?? String(err) })
      ?.end();
  } catch (loggingErr) {
    logger.debug({ err: loggingErr }, "langfuse endError failed");
  }
}

// Flushes and tears down the OTel SDK; called on process shutdown so the
// final batch of spans isn't lost.
export async function shutdownTracing() {
  if (!sdk) return;
  try {
    await sdk.shutdown();
    logger.info("langfuse tracing flushed and shut down");
  } catch (err) {
    logger.error({ err }, "langfuse shutdown failed");
  }
}
