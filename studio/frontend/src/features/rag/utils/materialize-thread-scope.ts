// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatThreadDeletedError } from "@/features/chat/api/chat-api";
import { isAssistantLocalThreadId } from "@/features/chat/utils/thread-ids";

export type ThreadScopeMaterialization = {
  threadId: string | null;
  readCurrentThreadItem: () => { id: string; remoteId: string | undefined };
  isThreadDeleted: (threadId: string) => boolean;
  requireStoredThread: (threadId: string) => Promise<boolean>;
  initialize: () => Promise<string>;
};

function assertThreadScopeActive(
  m: ThreadScopeMaterialization,
  threadId: string,
): void {
  if (m.isThreadDeleted(threadId)) {
    throw new ChatThreadDeletedError(`Thread ${threadId} was deleted`);
  }
}

async function initializeThreadScope(
  m: ThreadScopeMaterialization,
): Promise<string> {
  const threadId = await m.initialize();
  assertThreadScopeActive(m, threadId);
  return threadId;
}

async function confirmConcurrentInitialization(
  m: ThreadScopeMaterialization,
  threadId: string,
  missing: boolean,
  missingError: unknown,
): Promise<string> {
  let confirmed = false;
  try {
    confirmed = await m.requireStoredThread(threadId);
  } finally {
    assertThreadScopeActive(m, threadId);
  }
  if (confirmed) {
    return threadId;
  }
  if (missing) {
    throw missingError;
  }
  throw new Error(`Thread ${threadId} persistence could not be confirmed`);
}

export async function materializeThreadScope(
  m: ThreadScopeMaterialization,
): Promise<string> {
  if (!m.threadId) {
    return initializeThreadScope(m);
  }
  const initialState = m.readCurrentThreadItem();
  let confirmed = false;
  let missing = false;
  let missingError: unknown;
  try {
    confirmed = await m.requireStoredThread(m.threadId);
  } catch (error) {
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    missing = true;
    missingError = error;
  }
  assertThreadScopeActive(m, m.threadId);
  if (confirmed) {
    return m.threadId;
  }
  const state = m.readCurrentThreadItem();
  const initialStateWasFresh =
    initialState.id === m.threadId &&
    !initialState.remoteId &&
    isAssistantLocalThreadId(m.threadId);
  if (
    state.id === m.threadId &&
    !state.remoteId &&
    isAssistantLocalThreadId(m.threadId)
  ) {
    return initializeThreadScope(m);
  }
  if (initialStateWasFresh && state.id !== m.threadId) {
    throw new Error("Thread changed while preparing the document upload");
  }
  if (initialStateWasFresh && state.id === m.threadId && state.remoteId) {
    return confirmConcurrentInitialization(
      m,
      m.threadId,
      missing,
      missingError,
    );
  }
  if (missing) {
    throw missingError;
  }
  return m.threadId;
}
