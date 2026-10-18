/**
 * Real, network-free Mastra Agent run: two SEQUENTIAL model turns, each
 * requesting one tool call -- the companion scenario to
 * kyc_scenario_parallel.mjs, verifying that a later step correctly
 * chains onto the frontier the previous step left behind (not, as in
 * the parallel case, sharing one parent). See that file's docstring for
 * why the model here is a hand-rolled LanguageModelV2, not
 * `ai/test`'s `MockLanguageModelV2`.
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
  modelId: "sevenbox-scripted-kyc-sequential",
  supportedUrls: {},
  async doGenerate(opts) {
    turn += 1;
    if (turn === 1) {
      return {
        finishReason: "tool-calls",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        content: [
          { type: "tool-call", toolCallId: "c1", toolName: "check_watchlist", input: JSON.stringify({ applicant_id: "A-991" }) },
        ],
        warnings: [],
      };
    }
    if (turn === 2) {
      return {
        finishReason: "tool-calls",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        content: [
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
  id: "kyc-agent-sequential",
  name: "kyc-agent-sequential",
  instructions: "Run KYC checks on an applicant, one step at a time.",
  model: scriptedModel,
  tools: { check_watchlist: checkWatchlist, verify_document: verifyDocument },
});

const lines = [];
const sevenboxLog = createSevenboxStepLogger({ write: (line) => lines.push(line) });

sevenboxLog.recordRunStart("Run KYC on A-991, step by step");
const result = await agent.generate("Run KYC on A-991, step by step", {
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
