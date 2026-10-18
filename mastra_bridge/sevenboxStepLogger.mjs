/**
 * sevenboxStepLogger -- the TypeScript/Node-side half of Sevenbox's Mastra
 * integration. Emits a structured JSONL event log from a real Mastra
 * `Agent` run; the Python side (`sevenbox_adapters.mastra_adapter
 * .import_mastra_log`) reads that log afterward and constructs the real
 * Warrants. See sevenbox_adapters/mastra_adapter.py's module docstring for
 * the full architecture decision and why this is a log-importer, not a
 * live in-process adapter like every other framework in this project.
 *
 * This module deliberately computes NO Warrant ids, NO hashes, and knows
 * nothing about the Warrant Calculus -- it only records raw facts
 * (what ran, with what arguments, what it returned). Python remains the
 * only place `compute_id()` is ever evaluated, so there is never a
 * second implementation of content-addressing to keep in sync.
 *
 * Usage (two additions to an existing Mastra agent):
 *
 *   import { createSevenboxStepLogger } from "./sevenboxStepLogger.mjs";
 *
 *   const sevenboxLog = createSevenboxStepLogger();
 *   sevenboxLog.recordRunStart(userInput);
 *   const result = await agent.generate(userInput, {
 *     onStepFinish: sevenboxLog.onStepFinish,
 *   });
 *   sevenboxLog.recordRunFinish(result.text);
 *
 * By default writes one JSON object per line to stdout; pass
 * `{ write: (line) => ... }` to redirect elsewhere (a file stream, an
 * array, etc).
 */

/**
 * Verified against a real `@mastra/core@1.57.0` `Agent.generate()` run
 * (no network, a hand-rolled `LanguageModelV2`, and real `createTool`
 * tools -- see mastra_bridge/examples/kyc_scenario.mjs): `onStepFinish`'s
 * `step.content` array is NOT a reliable complete record for a
 * multi-tool-call step (in a two-parallel-tool-call step, `step.content`
 * carried only the SECOND call's tool-call/tool-result pair, not the
 * first -- confirmed after ruling out a bug in the example's own tools).
 * `step.toolCalls` and `step.toolResults` are the reliable, complete
 * arrays: every entry wrapped in Mastra's own event envelope
 * (`{ type: 'tool-call' | 'tool-result', runId, from: 'AGENT', payload:
 * { toolCallId, toolName, args, result? } }`), matched here by
 * `toolCallId`. This module reads `toolCalls`/`toolResults`, never
 * `content`, because of this.
 */
export function createSevenboxStepLogger({ write } = {}) {
  const writeLine = write || ((line) => process.stdout.write(line + "\n"));
  let stepIndex = 0;

  function recordRunStart(input, meta = {}) {
    writeLine(JSON.stringify({ type: "run_start", input, ts: Date.now(), ...meta }));
  }

  function recordRunFinish(output, meta = {}) {
    writeLine(JSON.stringify({ type: "run_finish", output, ts: Date.now(), ...meta }));
  }

  /**
   * Pass directly as `onStepFinish` in `agent.generate()`/`agent.stream()`
   * options. A step with no tool calls (e.g. the final text-only step)
   * produces no log line -- there is nothing for Sevenbox to represent.
   */
  function onStepFinish(step) {
    const resultsByCallId = new Map();
    for (const entry of step.toolResults || []) {
      const payload = entry.payload || entry;
      resultsByCallId.set(payload.toolCallId, payload);
    }

    const calls = [];
    for (const entry of step.toolCalls || []) {
      const payload = entry.payload || entry;
      const resultPayload = resultsByCallId.get(payload.toolCallId);
      calls.push({
        call_id: payload.toolCallId,
        name: payload.toolName,
        args: payload.args,
        ok: resultPayload !== undefined,
        result: resultPayload ? resultPayload.result : undefined,
      });
    }
    if (calls.length === 0) return;

    stepIndex += 1;
    writeLine(JSON.stringify({ type: "step", step_index: stepIndex, calls, ts: Date.now() }));
  }

  return { recordRunStart, recordRunFinish, onStepFinish };
}
