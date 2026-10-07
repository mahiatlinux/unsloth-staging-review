// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import {
  codexLocalToolRoundId,
  startsNewCodexToolRound,
} from "../src/features/chat/codex-reasoning.ts";
import { stripSearchImageTokens } from "../src/features/chat/search-images/search-images.ts";
import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import type { MessageRecord, ThreadRecord } from "../src/features/chat/types.ts";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  CONVERSATION_MARKDOWN_MIME_TYPE,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";
import {
  buildNamedConversationsMarkdown,
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";
import { csvDocument, csvEscape, CSV_MIME } from "../src/features/chat/utils/csv-export.ts";
import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import {
  conversationJsonlBody,
  exportFormatIncludesSiblings,
  ndjsonBody,
} from "../src/features/chat/utils/ndjson.ts";
import { unwrapPastedTextContent } from "../src/features/chat/utils/pasted-text.ts";
import { readSrc } from "./helpers/kit.ts";

type Exporters = {
  buildFineTuneJsonl: (format: string) => Promise<{
    lines: string[];
    conversations: number;
    skipped: number;
  }>;
  exportConversationRawJsonl: (threadId: string) => Promise<void>;
  exportConversationMessagesJsonl: (threadId: string) => Promise<void>;
  exportConversationShareGPT: (threadId: string) => Promise<void>;
  exportConversationCsv: (threadId: string) => Promise<void>;
  exportConversationMarkdown: (threadId: string) => Promise<void>;
  buildThreadContent: (threadId: string, format: string) => Promise<string | null>;
  saveConversationAsProjectSource: (
    threadId: string,
    projectId: string,
    title: string,
  ) => Promise<string>;
};

type ExportHarnessOptions = {
  threads?: ThreadRecord[];
  getStoredChatThread?: (id: string) => Promise<ThreadRecord | undefined>;
  listStoredChatMessages?: (id: string) => Promise<MessageRecord[]>;
  globalInferenceParams?: {
    systemPrompt?: string;
    systemVariables?: string;
  };
  getChatSettings?: () => Promise<{
    inferenceParams?: {
      systemPrompt?: string;
      systemVariables?: string;
    };
  }>;
  getStoredChatProject?: (id: string) => Promise<
    { instructions: string; archived: boolean } | null
  >;
  settleThreadScopedSettingsForCopy?: (id: string) => Promise<void>;
};

const DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

function slice(source: string, startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return source.slice(start, end);
}

const THREADS: ThreadRecord[] = [
  {
    id: "support",
    title: "Support",
    modelType: "base",
    projectId: "billing",
    archived: false,
    createdAt: 1,
    settings: {
      systemPrompt: "You answer for the {{team}} team. End with Ticket closed.",
      systemVariables: '{"team":"Billing"}',
    },
  },
  {
    id: "plain",
    title: "Plain",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 2,
    settings: { systemPrompt: "  " },
  },
];

const PROJECTS: Record<string, { instructions: string; archived: boolean }> = {
  billing: { instructions: "Cite the refund policy.", archived: false },
  openInComposer: { instructions: "Reply in French.", archived: false },
};

const SUPPORT_INSTRUCTIONS =
  "<project_instructions>\nCite the refund policy.\n</project_instructions>\n\n" +
  "You answer for the Billing team. End with Ticket closed.";

function turns(threadId: string) {
  return [
    ["u1", null, "user", "Where is my refund?"],
    ["a1", "u1", "assistant", "It went out today. Ticket closed."],
  ].map(([id, parentId, role, text], index) => ({
    id: `${threadId}-${id}`,
    threadId,
    parentId: parentId ? `${threadId}-${parentId}` : null,
    createdAt: index + 10,
    role,
    content: [{ type: "text", text }],
  }));
}

function loadExporters(
  threadIds: string[],
  downloads: string[],
  sources: string[] = [],
  options: ExportHarnessOptions = {},
) {
  const javascript = ts.transpileModule(
    [
      slice(ADAPTER, "function parseSystemVariablesMap(", "export const ThreadAutosaveHandle"),
      slice(ADAPTER, "async function resolveProjectInstructions(", "// Answered once per thread"),
      slice(ADAPTER, "export async function resolveProjectId(", "async function resolveSandboxSessionId("),
      slice(DIALOG, "function contentBlocksToText(", "/** A sidebar row as one markdown document."),
      slice(DIALOG, "async function buildThreadContent(", "function csvHeader("),
      slice(DIALOG, "// One JSONL line per conversation", "/** Download the fine-tuning JSONL"),
      "globalThis.__exporters = { buildFineTuneJsonl, exportConversationRawJsonl, exportConversationMessagesJsonl, exportConversationShareGPT, exportConversationCsv, exportConversationMarkdown, buildThreadContent, saveConversationAsProjectSource };",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const context = {
    exports: {},
    toast: { info: () => {}, success: () => {} },
    listStoredChatThreads: async () =>
      (options.threads ?? THREADS).filter((thread) =>
        threadIds.includes(thread.id),
      ),
    listStoredChatMessages:
      options.listStoredChatMessages ??
      (async (id: string) => turns(id) as MessageRecord[]),
    getStoredChatThread:
      options.getStoredChatThread ??
      (async (id: string) =>
        (options.threads ?? THREADS).find((thread) => thread.id === id)),
    getChatSettings:
      options.getChatSettings ??
      (async () => ({ inferenceParams: options.globalInferenceParams })),
    flushPendingChatSettings: async () => {},
    settleThreadScopedSettingsForCopy:
      options.settleThreadScopedSettingsForCopy ?? (async () => {}),
    getStoredChatProject:
      options.getStoredChatProject ??
      (async (id: string) => PROJECTS[id] ?? null),
    useChatRuntimeStore: { getState: () => ({ activeProjectId: "openInComposer" }) },
    isThreadIncognito: () => false,
    composerProjectByPendingThread: new Map(),
    ...liveThreadHead,
    orderByParentChain,
    unwrapPastedTextContent,
    toolResultModelText: (result: unknown) => result,
    toolCallReplayArguments,
    codexLocalToolRoundId,
    startsNewCodexToolRound,
    contentBlocksToMarkdownBlocks,
    renderConversationBlocks,
    buildConversationMarkdown,
    buildNamedConversationsMarkdown,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    CONVERSATION_MARKDOWN_MIME_TYPE,
    stripSearchImageTokens,
    conversationJsonlBody,
    exportFormatIncludesSiblings,
    ndjsonBody,
    csvDocument,
    csvEscape,
    CSV_MIME,
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
    saveMarkdownAsProjectSource: async (_projectId: string, body: string) => {
      sources.push(body);
      return true;
    },
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__exporters as Exporters;
}

test("chat training data starts with the system prompt the chat ran with", async () => {
  const exporters = loadExporters(["support", "plain"], []);

  const openai = (await exporters.buildFineTuneJsonl("openai")).lines.map(
    (line) => JSON.parse(line).messages,
  );
  assert.deepEqual(openai[0], [
    { role: "system", content: SUPPORT_INSTRUCTIONS },
    { role: "user", content: "Where is my refund?" },
    { role: "assistant", content: "It went out today. Ticket closed." },
  ]);
  assert.deepEqual(openai[1], [
    { role: "user", content: "Where is my refund?" },
    { role: "assistant", content: "It went out today. Ticket closed." },
  ]);

  const sharegpt = (await exporters.buildFineTuneJsonl("sharegpt")).lines.map(
    (line) => JSON.parse(line).conversations,
  );
  assert.deepEqual(sharegpt[0][0], { from: "system", value: SUPPORT_INSTRUCTIONS });
  assert.equal(sharegpt[1][0].from, "human");

  const alpaca = (await exporters.buildFineTuneJsonl("alpaca")).lines.map((line) =>
    JSON.parse(line),
  );
  assert.equal(alpaca[0].input, SUPPORT_INSTRUCTIONS);
  assert.equal(alpaca[1].input, "");
});

test("terminal stream yields retain the run's resolved instructions", () => {
  const audioProgress = slice(
    ADAPTER,
    'text: "Generating audio..."',
    "const result = await generateAudio(",
  );
  const success = slice(
    ADAPTER,
    "const finalIncompleteReason =",
    "} catch (err) {",
  );
  const failure = slice(
    ADAPTER,
    "if (!abortSignal.aborted || generationStopRequested) {",
    "throw err;",
  );

  assert.match(audioProgress, /resolvedInstructions: combinedSystemPrompt/);
  assert.match(success, /resolvedInstructions: combinedSystemPrompt/);
  assert.match(failure, /resolvedInstructions: combinedSystemPrompt/);
});

test("every chat export format starts with the chat's system prompt", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["support"], downloads);

  await exporters.exportConversationRawJsonl("support");
  assert.deepEqual(JSON.parse(downloads[0]).messages[0], {
    role: "system",
    content: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationMessagesJsonl("support");
  assert.deepEqual(JSON.parse(downloads[1].split("\n")[0]), {
    role: "system",
    content: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationShareGPT("support");
  assert.deepEqual(JSON.parse(downloads[2]).conversations[0], {
    from: "system",
    value: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationCsv("support");
  assert.match(downloads[3], /^\W*role,content\r?\n"system","<project_instructions>/);

  await exporters.exportConversationMarkdown("support");
  const markdown = downloads[4];
  assert.ok(markdown.includes(`## System\n\n${SUPPORT_INSTRUCTIONS}`));
  assert.ok(markdown.indexOf("## System") < markdown.indexOf("## User"));

  const bulk = await exporters.buildThreadContent("support", "jsonl-raw");
  assert.equal(JSON.parse(bulk ?? "").messages[0].role, "system");
});

test("a chat with no system prompt exports only its own turns", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["plain"], downloads);

  await exporters.exportConversationRawJsonl("plain");
  await exporters.exportConversationShareGPT("plain");
  await exporters.exportConversationCsv("plain");
  await exporters.exportConversationMarkdown("plain");

  assert.equal(JSON.parse(downloads[0]).messages[0].role, "user");
  assert.equal(JSON.parse(downloads[1]).conversations[0].from, "human");
  assert.ok(!downloads[2].includes('"system"'));
  assert.ok(!downloads[3].includes("## System"));
  assert.ok(!downloads.join("").includes("French"));
});

test("a chat without a settings snapshot inherits the global system prompt", async () => {
  const downloads: string[] = [];
  const inherited: ThreadRecord = {
    id: "inherited",
    title: "Inherited",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 3,
  };
  const exporters = loadExporters([inherited.id], downloads, [], {
    threads: [inherited],
    globalInferenceParams: {
      systemPrompt: "Answer for the {{team}} team.",
      systemVariables: '{"team":"Billing"}',
    },
  });

  await exporters.exportConversationRawJsonl(inherited.id);

  assert.deepEqual(JSON.parse(downloads[0]).messages[0], {
    role: "system",
    content: "Answer for the Billing team.",
  });
});

test("an imported system turn is not replaced or duplicated by current defaults", async () => {
  const downloads: string[] = [];
  let settingsReads = 0;
  const imported: ThreadRecord = {
    id: "imported",
    title: "Imported",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 4,
  };
  const messages = [
    {
      id: "imported-system",
      threadId: imported.id,
      parentId: null,
      role: "system",
      content: [{ type: "text", text: "Original imported prompt" }],
      createdAt: 10,
    },
    {
      id: "imported-user",
      threadId: imported.id,
      parentId: "imported-system",
      role: "user",
      content: [{ type: "text", text: "Imported question" }],
      createdAt: 11,
    },
    {
      id: "imported-assistant",
      threadId: imported.id,
      parentId: "imported-user",
      role: "assistant",
      content: [{ type: "text", text: "Imported answer" }],
      createdAt: 12,
    },
  ] as MessageRecord[];
  const exporters = loadExporters([imported.id], downloads, [], {
    threads: [imported],
    listStoredChatMessages: async () => messages,
    getChatSettings: async () => {
      settingsReads += 1;
      return { inferenceParams: { systemPrompt: "Current global prompt" } };
    },
  });

  await exporters.exportConversationRawJsonl(imported.id);
  const training = await exporters.buildFineTuneJsonl("openai");

  assert.deepEqual(JSON.parse(downloads[0]).messages, [
    { role: "system", content: "Original imported prompt" },
    { role: "user", content: "Imported question" },
    { role: "assistant", content: "Imported answer" },
  ]);
  assert.deepEqual(JSON.parse(training.lines[0]).messages, [
    { role: "system", content: "Original imported prompt" },
    { role: "user", content: "Imported question" },
    { role: "assistant", content: "Imported answer" },
  ]);
  assert.equal(settingsReads, 0);
});

test("later epochs retain the imported system turn alongside current instructions", async () => {
  const downloads: string[] = [];
  const imported: ThreadRecord = {
    id: "continued-import",
    title: "Continued import",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 4,
    settings: { systemPrompt: "Prompt B", systemVariables: "" },
  };
  const messages: MessageRecord[] = [
    {
      id: "continued-system",
      threadId: imported.id,
      parentId: null,
      role: "system",
      content: [{ type: "text", text: "Imported prompt A" }],
      createdAt: 10,
    },
    {
      id: "continued-u1",
      threadId: imported.id,
      parentId: "continued-system",
      role: "user",
      content: [{ type: "text", text: "Imported question" }],
      createdAt: 11,
    },
    {
      id: "continued-a1",
      threadId: imported.id,
      parentId: "continued-u1",
      role: "assistant",
      content: [{ type: "text", text: "Imported answer" }],
      createdAt: 12,
    },
    {
      id: "continued-u2",
      threadId: imported.id,
      parentId: "continued-a1",
      role: "user",
      content: [{ type: "text", text: "New question" }],
      createdAt: 13,
    },
    {
      id: "continued-a2",
      threadId: imported.id,
      parentId: "continued-u2",
      role: "assistant",
      content: [{ type: "text", text: "New answer" }],
      metadata: { resolvedInstructions: "Prompt B" },
      createdAt: 14,
    },
  ];
  const exporters = loadExporters([imported.id], downloads, [], {
    threads: [imported],
    listStoredChatMessages: async () => messages,
  });

  await exporters.exportConversationRawJsonl(imported.id);
  const records = downloads[0]
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line).messages);
  const fineTune = await exporters.buildFineTuneJsonl("openai");
  const fineTuneRecords = fineTune.lines.map(
    (line) => JSON.parse(line).messages,
  );

  assert.deepEqual(
    records[1]
      .filter((message: { role: string }) => message.role === "system")
      .map((message: { content: string }) => message.content),
    ["Prompt B", "Imported prompt A"],
  );
  assert.deepEqual(
    fineTuneRecords[1]
      .filter((message: { role: string }) => message.role === "system")
      .map((message: { content: string }) => message.content),
    ["Prompt B\n\nImported prompt A"],
  );
});

test("chat exports preserve instruction changes at their run boundaries", async () => {
  const downloads: string[] = [];
  const captured: ThreadRecord = {
    id: "captured",
    title: "Captured",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 3,
    settings: { systemPrompt: "Prompt B", systemVariables: "" },
  };
  const messages: MessageRecord[] = [
    {
      id: "captured-u1",
      threadId: captured.id,
      parentId: null,
      role: "user",
      content: [{ type: "text", text: "First question" }],
      createdAt: 10,
    },
    {
      id: "captured-a1",
      threadId: captured.id,
      parentId: "captured-u1",
      role: "assistant",
      content: [{ type: "text", text: "First answer" }],
      metadata: { resolvedInstructions: "Prompt A on 2026-10-06" },
      createdAt: 11,
    },
    {
      id: "captured-u2",
      threadId: captured.id,
      parentId: "captured-a1",
      role: "user",
      content: [{ type: "text", text: "Second question" }],
      createdAt: 12,
    },
    {
      id: "captured-a2",
      threadId: captured.id,
      parentId: "captured-u2",
      role: "assistant",
      content: [{ type: "text", text: "Second answer" }],
      metadata: { resolvedInstructions: "Prompt B on 2026-10-07" },
      createdAt: 13,
    },
  ];
  const exporters = loadExporters([captured.id], downloads, [], {
    threads: [captured],
    listStoredChatMessages: async () => messages,
  });

  await exporters.exportConversationRawJsonl(captured.id);

  assert.deepEqual(
    downloads[0]
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line).messages),
    [
      [
        { role: "system", content: "Prompt A on 2026-10-06" },
        { role: "user", content: "First question" },
        { role: "assistant", content: "First answer" },
      ],
      [
        { role: "system", content: "Prompt B on 2026-10-07" },
        {
          role: "user",
          content:
            "<conversation_context>\nUser: First question\n\nAssistant: First answer\n</conversation_context>",
        },
        { role: "user", content: "Second question" },
        { role: "assistant", content: "Second answer" },
      ],
    ],
  );
});

test("training exports split when a later run clears its instructions", async () => {
  const downloads: string[] = [];
  const cleared: ThreadRecord = {
    id: "cleared",
    title: "Cleared",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 3,
    settings: { systemPrompt: "", systemVariables: "" },
  };
  const messages = turns(cleared.id) as MessageRecord[];
  messages[1] = {
    ...messages[1],
    metadata: { resolvedInstructions: "Prompt A" },
  };
  messages.push(
    {
      id: "cleared-u2",
      threadId: cleared.id,
      parentId: `${cleared.id}-a1`,
      role: "user",
      content: [{ type: "text", text: "Question without instructions" }],
      createdAt: 12,
    },
    {
      id: "cleared-a2",
      threadId: cleared.id,
      parentId: "cleared-u2",
      role: "assistant",
      content: [{ type: "text", text: "Answer without instructions" }],
      metadata: { resolvedInstructions: "" },
      createdAt: 13,
    },
  );
  const exporters = loadExporters([cleared.id], downloads, [], {
    threads: [cleared],
    listStoredChatMessages: async () => messages,
  });

  await exporters.exportConversationRawJsonl(cleared.id);
  const records = downloads[0]
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line).messages);
  const fineTune = await exporters.buildFineTuneJsonl("openai");

  assert.equal(records.length, 2);
  assert.equal(records[0][0].content, "Prompt A");
  assert.ok(records[1].every((message: { role: string }) => message.role !== "system"));
  assert.equal(fineTune.lines.length, 2);
  assert.ok(
    JSON.parse(fineTune.lines[1]).messages.every(
      (message: { role: string }) => message.role !== "system",
    ),
  );
});

test("fine-tuning data keeps earlier turns as context across instruction changes", async () => {
  const thread: ThreadRecord = {
    id: "changing",
    title: "Changing",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 3,
    settings: { systemPrompt: "Prompt B", systemVariables: "" },
  };
  const messages: MessageRecord[] = [
    ...turns(thread.id),
    {
      id: `${thread.id}-u2`,
      threadId: thread.id,
      parentId: `${thread.id}-a1`,
      role: "user",
      content: [{ type: "text", text: "Second question" }],
      createdAt: 12,
    },
    {
      id: `${thread.id}-a2`,
      threadId: thread.id,
      parentId: `${thread.id}-u2`,
      role: "assistant",
      content: [{ type: "text", text: "Second answer" }],
      metadata: { resolvedInstructions: "Prompt B" },
      createdAt: 13,
    },
  ] as MessageRecord[];
  messages[1] = {
    ...messages[1],
    metadata: { resolvedInstructions: "Prompt A" },
  };
  const exporters = loadExporters([thread.id], [], [], {
    threads: [thread],
    listStoredChatMessages: async () => messages,
  });

  const result = await exporters.buildFineTuneJsonl("openai");

  const previousContext =
    "<conversation_context>\n" +
    "User: Where is my refund?\n\n" +
    "Assistant: It went out today. Ticket closed.\n" +
    "</conversation_context>\n\nSecond question";

  assert.equal(result.conversations, 2);
  assert.deepEqual(
    Array.from(result.lines, (line) => JSON.parse(line).messages),
    [
      [
        { role: "system", content: "Prompt A" },
        { role: "user", content: "Where is my refund?" },
        { role: "assistant", content: "It went out today. Ticket closed." },
      ],
      [
        { role: "system", content: "Prompt B" },
        { role: "user", content: previousContext },
        { role: "assistant", content: "Second answer" },
      ],
    ],
  );

  const sharegpt = await exporters.buildFineTuneJsonl("sharegpt");
  assert.equal(
    JSON.parse(sharegpt.lines[1]).conversations[1].value,
    previousContext,
  );

  const alpaca = await exporters.buildFineTuneJsonl("alpaca");
  assert.equal(JSON.parse(alpaca.lines[1]).instruction, "Second question");
  assert.equal(
    JSON.parse(alpaca.lines[1]).input,
    "Prompt B\n\nUser: Where is my refund?\n\nAssistant: It went out today. Ticket closed.",
  );
});

test("bulk training export reads inherited global prompt settings once", async () => {
  let settingsReads = 0;
  const inherited = ["first", "second"].map(
    (id, index): ThreadRecord => ({
      id,
      title: id,
      modelType: "base",
      projectId: null,
      archived: false,
      createdAt: index + 10,
    }),
  );
  const exporters = loadExporters(
    inherited.map((thread) => thread.id),
    [],
    [],
    {
      threads: inherited,
      getChatSettings: async () => {
        settingsReads += 1;
        return { inferenceParams: { systemPrompt: "One shared prompt" } };
      },
    },
  );

  const result = await exporters.buildFineTuneJsonl("openai");

  assert.equal(result.lines.length, 2);
  assert.equal(settingsReads, 1);
  for (const line of result.lines) {
    assert.equal(JSON.parse(line).messages[0].content, "One shared prompt");
  }
});

test("bulk training export skips global settings for complete snapshots", async () => {
  let settingsReads = 0;
  const complete: ThreadRecord = {
    id: "complete",
    title: "Complete",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 12,
    settings: { systemPrompt: "Saved prompt", systemVariables: "" },
  };
  const exporters = loadExporters([complete.id], [], [], {
    threads: [complete],
    getChatSettings: async () => {
      settingsReads += 1;
      throw new Error("global settings unavailable");
    },
  });

  const result = await exporters.buildFineTuneJsonl("openai");

  assert.equal(result.lines.length, 1);
  assert.equal(JSON.parse(result.lines[0]).messages[0].content, "Saved prompt");
  assert.equal(settingsReads, 0);
});

test("bulk training export reads each project's instructions once", async () => {
  let projectReads = 0;
  const threads = ["first", "second"].map(
    (id, index): ThreadRecord => ({
      id,
      title: id,
      modelType: "base",
      projectId: "billing",
      archived: false,
      createdAt: index + 20,
      settings: { systemPrompt: "Shared prompt", systemVariables: "" },
    }),
  );
  const exporters = loadExporters(
    threads.map((thread) => thread.id),
    [],
    [],
    {
      threads,
      getStoredChatProject: async () => {
        projectReads += 1;
        return PROJECTS.billing;
      },
    },
  );

  const result = await exporters.buildFineTuneJsonl("openai");

  assert.equal(result.lines.length, 2);
  assert.equal(projectReads, 1);
});

test("project instruction lookup failures stop the export", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["support"], downloads, [], {
    getStoredChatProject: async () => {
      throw new Error("project lookup unavailable");
    },
  });

  await assert.rejects(
    exporters.exportConversationRawJsonl("support"),
    /project lookup unavailable/,
  );
  assert.equal(downloads.length, 0);
});

test("export waits for a pending thread prompt write before reading settings", async () => {
  const downloads: string[] = [];
  let prompt = "Old prompt";
  let settled = false;
  const pending: Omit<ThreadRecord, "settings"> = {
    id: "pending",
    title: "Pending",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 4,
  };
  const exporters = loadExporters([pending.id], downloads, [], {
    settleThreadScopedSettingsForCopy: async (id) => {
      assert.equal(id, pending.id);
      prompt = "New prompt";
      settled = true;
    },
    getStoredChatThread: async (id) => {
      assert.equal(id, pending.id);
      assert.equal(settled, true);
      return { ...pending, settings: { systemPrompt: prompt, systemVariables: "" } };
    },
  });

  await exporters.exportConversationMarkdown(pending.id);

  assert.ok(downloads[0].includes("New prompt"));
  assert.ok(!downloads[0].includes("Old prompt"));
});

test("saving a chat to project sources leaves its system prompt out", async () => {
  const sources: string[] = [];
  const exporters = loadExporters(["support"], [], sources);

  await exporters.saveConversationAsProjectSource("support", "billing", "Support");

  assert.equal(sources.length, 1);
  assert.ok(!sources[0].includes("## System"));
  assert.ok(sources[0].includes("Where is my refund?"));
});
