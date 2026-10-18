/**
 * Real, network-free Mastra Agent run: one model turn requests TWO
 * genuinely independent tool calls (check_watchlist + verify_document),
 * exercising fan-out. Writes a Sevenbox JSONL log to stdout (or the path
 * given as argv[2]) via sevenboxStepLogger -- read it afterward with
 * `sevenbox_adapters.mastra_adapter.import_mastra_log()`.
 *
 * No API key, no network: the model is a hand-rolled object implementing
 * @ai-sdk/provider's LanguageModelV2 interface directly (the officially
 * exported `MockLanguageModelV2` from `ai/test` requires `vitest`/`msw`
 * as peer deps for its test harness, which this project has no other use
 * for -- verified by trying it first and hitting ERR_MODULE_NOT_FOUND;
 * a plain object satisfying the same interface avoids that dependency
 * entirely and is exactly what MockLanguageModelV2 wraps internally).
 */
import { writeFileSync } from "node:fs";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { createSevenboxStepLogger } from "../sevenboxStepLogger.mjs";

const checkWatchlist = createTool({
  id: "check_watchlist",
  description: "Check an applicant against the watchlist",
  inputSchema: z.object({ applicant_id: z.string() }),
  execute: async (inputData) => ({
    applicant_id: inputData.applicant_id,
    watchlist_match: false,
  }),
});

const verifyDocument = createTool({
  id: "verify_document",
  description: "Verify an applicant's submitted document",
  inputSchema: z.object({ applicant_id: z.string() }),
  execute: async (inputData) => ({
    applicant_id: inputData.applicant_id,
    document_verified: true,
  }),
});

let turn = 0;
const scriptedModel = {
  specificationVersion: "v2",
  provider: "sevenbox-example",
  modelId: "sevenbox-scripted-kyc-parallel",
  supportedUrls: {},
  async doGenerate(opts) {
    turn += 1;
    if (turn === 1) {
      return {
        finishReason: "tool-calls",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        content: [
          { type: "tool-call", toolCallId: "c1", toolName: "check_watchlist", input: JSON.stringify({ applicant_id: "A-991" }) },
          { type: "tool-call", toolCallId: "c2", toolName: "verify_document", input: JSON.stringify({ applicant_id: "A-991" }) },
        ],
        warnings: [],
      };
    }
    return {
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      content: [{ type: "text", text: "KYC complete." }],
      warnings: [],
    };
  },
};

const agent = new Agent({
  id: "kyc-agent",
  name: "kyc-agent",
  instructions: "Run KYC checks on an applicant.",
  model: scriptedModel,
  tools: { check_watchlist: checkWatchlist, verify_document: verifyDocument },
});

const lines = [];
const sevenboxLog = createSevenboxStepLogger({ write: (line) => lines.push(line) });

sevenboxLog.recordRunStart("Run KYC on A-991");
const result = await agent.generate("Run KYC on A-991", {
  onStepFinish: sevenboxLog.onStepFinish,
});
sevenboxLog.recordRunFinish(result.text);

const output = lines.join("\n") + "\n";
const outPath = process.argv[2];
if (outPath) {
  writeFileSync(outPath, output);
  console.error(`wrote ${lines.length} lines to ${outPath}`);
} else {
  process.stdout.write(output);
}
