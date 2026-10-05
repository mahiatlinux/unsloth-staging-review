// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { renewTemporaryThreadDocumentLease } from "../api/rag-api";

const INITIAL_RETRY_AFTER_FAILURE_MS = 250;
const MAX_RETRY_AFTER_FAILURE_MS = 30_000;

/** Renew rowless document ownership until this Temporary Chat leaves the browser. */
export function keepTemporaryThreadDocumentLeaseAlive(
  threadId: string,
): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let request: AbortController | undefined;
  let conservativeExpiryMs: number | undefined;
  const delayWithinLease = (maximumMs: number) => {
    if (
      conservativeExpiryMs !== undefined &&
      conservativeExpiryMs <= Date.now()
    ) {
      conservativeExpiryMs = undefined;
    }
    if (conservativeExpiryMs === undefined) {
      return Math.min(maximumMs, INITIAL_RETRY_AFTER_FAILURE_MS);
    }
    return Math.min(
      maximumMs,
      Math.max(1, Math.floor((conservativeExpiryMs - Date.now()) / 2)),
    );
  };
  const renew = async () => {
    const startedAtMs = Date.now();
    const controller = new AbortController();
    request = controller;
    const requestTimeout = setTimeout(
      () => controller.abort(),
      delayWithinLease(MAX_RETRY_AFTER_FAILURE_MS),
    );
    // authFetch can outlive its caller's signal while it refreshes a session. Race
    // the signal ourselves so that path cannot consume the rest of this lease.
    const aborted = new Promise<never>((_resolve, reject) => {
      controller.signal.addEventListener(
        "abort",
        () => reject(new DOMException("Aborted", "AbortError")),
        { once: true },
      );
    });
    let delay: number;
    try {
      const lease = await Promise.race([
        renewTemporaryThreadDocumentLease(threadId, controller.signal),
        aborted,
      ]);
      if (!lease.active) {
        return;
      }
      if (Number.isFinite(lease.renewAfterMs) && lease.renewAfterMs > 0) {
        // The backend renews during this request for at least twice its suggested
        // interval. Starting the clock before the request is conservative and lets
        // failures account for both request duration and earlier failed attempts.
        conservativeExpiryMs = startedAtMs + lease.renewAfterMs * 2;
        delay = delayWithinLease(lease.renewAfterMs);
      } else {
        delay = delayWithinLease(INITIAL_RETRY_AFTER_FAILURE_MS);
      }
    } catch {
      // A transient backend outage is not proof that the Temporary Chat ended.
      delay = delayWithinLease(MAX_RETRY_AFTER_FAILURE_MS);
    } finally {
      clearTimeout(requestTimeout);
      if (request === controller) {
        request = undefined;
      }
    }
    if (!stopped) {
      timer = setTimeout(renew, delay);
    }
  };
  renew().catch(() => undefined);
  return () => {
    stopped = true;
    request?.abort();
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  };
}
