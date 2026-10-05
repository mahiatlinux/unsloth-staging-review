// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ThreadScopeMaterialization } from "../src/features/rag/utils/materialize-thread-scope";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const FRESH = "__LOCALID_fresh0001";
const THREAD_CHANGED_ERROR =
  /Thread changed while preparing the document upload/;
const PERSISTENCE_UNCONFIRMED_ERROR =
  /persistence could not be confirmed/;

class ChatThreadDeletedErrorStub extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ChatThreadDeletedError";
  }
}

function load() {
  const { materializeThreadScope } = loadWithStubs<{
    materializeThreadScope: (m: ThreadScopeMaterialization) => Promise<string>;
  }>(
    new URL(
      "../src/features/rag/utils/materialize-thread-scope.ts",
      import.meta.url,
    ),
    {
      "@/features/chat/api/chat-api": {
        ChatThreadDeletedError: ChatThreadDeletedErrorStub,
      },
      "@/features/chat/utils/thread-ids": {
        isAssistantLocalThreadId: (id: string | null | undefined) =>
          typeof id === "string" && id.startsWith("__LOCALID_"),
      },
    },
  );
  return materializeThreadScope;
}

function deferredWriteStore() {
  const store = { resolvable: false };
  let reads = 0;
  const requireStoredThread = (threadId: string): Promise<boolean> => {
    reads += 1;
    return store.resolvable
      ? Promise.resolve(true)
      : Promise.reject(new Error(`Thread ${threadId} was not persisted`));
  };
  const initialize = () => {
    store.resolvable = true;
    return requireStoredThread(FRESH).then(() => FRESH);
  };
  return { requireStoredThread, initialize, readCount: () => reads };
}

function neverInitialize(): () => Promise<string> {
  const initialize = () => {
    throw new Error("initialize must not run");
  };
  return initialize;
}

test("a brand-new chat whose __LOCALID_ is missing materializes the row", async () => {
  const materialize = load();
  const { requireStoredThread, initialize, readCount } = deferredWriteStore();

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
    isThreadDeleted: () => false,
    requireStoredThread,
    initialize,
  });

  assert.equal(result, FRESH);
  assert.equal(
    readCount(),
    2,
    "the initialized row is confirmed after the miss",
  );
});

test("a brand-new chat still materializes when its stored-thread read is indeterminate", async () => {
  const materialize = load();
  let initialized = 0;

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
    isThreadDeleted: () => false,
    requireStoredThread: () => Promise.resolve(false),
    initialize: () => {
      initialized += 1;
      return Promise.resolve(FRESH);
    },
  });

  assert.equal(result, FRESH);
  assert.equal(initialized, 1);
});

test("an id-less composer materializes the thread immediately", async () => {
  const materialize = load();
  let initialized = 0;

  const result = await materialize({
    threadId: null,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
    isThreadDeleted: () => false,
    requireStoredThread: () => Promise.resolve(true),
    initialize: () => {
      initialized += 1;
      return Promise.resolve(FRESH);
    },
  });

  assert.equal(result, FRESH);
  assert.equal(initialized, 1);
});

test("a saved chat (remoteId set) that reads missing is an error, never re-initialized", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: FRESH }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a saved chat is reused when its stored-thread read is indeterminate", async () => {
  const materialize = load();

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({ id: FRESH, remoteId: FRESH }),
    isThreadDeleted: () => false,
    requireStoredThread: () => Promise.resolve(false),
    initialize: neverInitialize(),
  });

  assert.equal(result, FRESH);
});

test("a concurrent initializer gets a second stored-row check after a stale miss", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  let stateReads = 0;
  let storedReads = 0;

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({
      id: FRESH,
      remoteId: stateReads++ === 0 ? undefined : FRESH,
    }),
    isThreadDeleted: () => false,
    requireStoredThread: () => {
      storedReads += 1;
      return storedReads === 1
        ? Promise.reject(missing)
        : Promise.resolve(true);
    },
    initialize: neverInitialize(),
  });

  assert.equal(result, FRESH);
  assert.equal(storedReads, 2);
});

test("a concurrent initializer gets a confirmed recheck after an indeterminate read", async () => {
  const materialize = load();
  let stateReads = 0;
  let storedReads = 0;

  const result = await materialize({
    threadId: FRESH,
    readCurrentThreadItem: () => ({
      id: FRESH,
      remoteId: stateReads++ === 0 ? undefined : FRESH,
    }),
    isThreadDeleted: () => false,
    requireStoredThread: () => {
      storedReads += 1;
      return Promise.resolve(storedReads === 2);
    },
    initialize: neverInitialize(),
  });

  assert.equal(result, FRESH);
  assert.equal(storedReads, 2);
});

test("an indeterminate concurrent-initializer recheck blocks the upload", async () => {
  const materialize = load();
  let stateReads = 0;
  let storedReads = 0;

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({
        id: FRESH,
        remoteId: stateReads++ === 0 ? undefined : FRESH,
      }),
      isThreadDeleted: () => false,
      requireStoredThread: () => {
        storedReads += 1;
        return Promise.resolve(false);
      },
      initialize: neverInitialize(),
    }),
    PERSISTENCE_UNCONFIRMED_ERROR,
  );
  assert.equal(storedReads, 2);
});

test("an indeterminate concurrent-initializer recheck keeps the stale miss blocked", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  let stateReads = 0;
  let storedReads = 0;

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({
        id: FRESH,
        remoteId: stateReads++ === 0 ? undefined : FRESH,
      }),
      isThreadDeleted: () => false,
      requireStoredThread: () => {
        storedReads += 1;
        return storedReads === 1
          ? Promise.reject(missing)
          : Promise.resolve(false);
      },
      initialize: neverInitialize(),
    }),
    (error: unknown) => error === missing,
  );
  assert.equal(storedReads, 2);
});

test("a real (non-__LOCALID_) id that reads missing stays an error", async () => {
  const materialize = load();
  const missing = new Error("Thread chat_001 was not persisted");
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: "chat_001",
      readCurrentThreadItem: () => ({ id: "chat_001", remoteId: undefined }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a tombstoned thread stays an error instead of being resurrected", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
      isThreadDeleted: () => true,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error instanceof ChatThreadDeletedErrorStub,
  );
});

test("a tombstone landing during a successful read still blocks the upload", async () => {
  const materialize = load();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: FRESH }),
      isThreadDeleted: () => true,
      requireStoredThread: () => Promise.resolve(true),
      initialize: neverInitialize(),
    }),
    (error: unknown) => error instanceof ChatThreadDeletedErrorStub,
  );
});

test("a tombstone landing during an indeterminate read blocks the upload", async () => {
  const materialize = load();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
      isThreadDeleted: () => true,
      requireStoredThread: () => Promise.resolve(false),
      initialize: neverInitialize(),
    }),
    (error: unknown) => error instanceof ChatThreadDeletedErrorStub,
  );
});

test("a backend-tombstoned read stays an error instead of being resurrected", async () => {
  const materialize = load();
  const deleted = new ChatThreadDeletedErrorStub(`Thread ${FRESH} was deleted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({ id: FRESH, remoteId: undefined }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw deleted;
      },
      initialize,
    }),
    (error: unknown) => error === deleted,
  );
});

test("a thread switch while the stored check waits is an error, not a recovery", async () => {
  const materialize = load();
  const missing = new Error(`Thread ${FRESH} was not persisted`);
  const initialize = neverInitialize();

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({
        id: "__LOCALID_other",
        remoteId: undefined,
      }),
      isThreadDeleted: () => false,
      requireStoredThread: async () => {
        throw missing;
      },
      initialize,
    }),
    (error: unknown) => error === missing,
  );
});

test("a thread switch during an indeterminate read blocks the upload", async () => {
  const materialize = load();
  let stateReads = 0;

  await assert.rejects(
    materialize({
      threadId: FRESH,
      readCurrentThreadItem: () => ({
        id: stateReads++ === 0 ? FRESH : "__LOCALID_other",
        remoteId: undefined,
      }),
      isThreadDeleted: () => false,
      requireStoredThread: () => Promise.resolve(false),
      initialize: neverInitialize(),
    }),
    THREAD_CHANGED_ERROR,
  );
});
