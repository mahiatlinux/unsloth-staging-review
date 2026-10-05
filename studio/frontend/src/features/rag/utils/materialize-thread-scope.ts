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

export async function materializeThreadScope(
  m: ThreadScopeMaterialization,
): Promise<string> {
  if (!m.threadId) {
    return m.initialize();
  }
  let missing = false;
  let missingError: unknown;
  try {
    if (await m.requireStoredThread(m.threadId)) {
      return m.threadId;
    }
  } catch (error) {
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    missing = true;
    missingError = error;
  }
  const state = m.readCurrentThreadItem();
  const deleted = m.isThreadDeleted(m.threadId);
  if (
    !deleted &&
    state.id === m.threadId &&
    !state.remoteId &&
    isAssistantLocalThreadId(m.threadId)
  ) {
    return m.initialize();
  }
  if (missing) {
    throw missingError;
  }
  if (deleted) {
    throw new ChatThreadDeletedError(`Thread ${m.threadId} was deleted`);
  }
  return m.threadId;
}
