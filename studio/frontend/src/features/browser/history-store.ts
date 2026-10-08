// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { type StateStorage, createJSONStorage, persist } from "zustand/middleware";
import { useChatRuntimeStore } from "@/features/chat";
import { accountDatabaseName } from "@/lib/account-transition";
import { hostOf } from "./address";
import { forgetNativeDownloads } from "./native-downloads";
import { useBrowserPrefsStore } from "./prefs-store";

export type HistoryItem = { id: string; url: string; title: string; visitedAt: number };
export type DownloadItem = {
  id: string;
  name: string;
  url: string | null;
  size: number;
  contentType: string;
  downloadedAt: number;
  /** native saved-file id from native-downloads.ts. */
  nativeId?: string;
};

const MAX_HISTORY = 1000;
const MAX_DOWNLOADS = 200;
// cap page-provided URLs and titles so history cannot fill Studio storage.
export const MAX_URL_CHARS = 2048;
export const MAX_TITLE_CHARS = 200;
// sites usually declare icons in-page instead of at /favicon.ico; keep one per host.
const MAX_ICONS = 300;
const PERSIST_DELAY_MS = 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

/** keeps icons only for hosts with retained visits. */
function iconsFor(history: HistoryItem[], icons: Record<string, string>): Record<string, string> {
  const hosts = new Set(history.map((visit) => hostOf(visit.url)));
  return Object.fromEntries(Object.entries(icons).filter(([host]) => hosts.has(host)));
}

function savesHistory(temporary: boolean): boolean {
  return useBrowserPrefsStore.getState().saveHistory && !temporary && !useChatRuntimeStore.getState().incognito;
}

/** returns the retention cutoff; 0 keeps all visits. */
function retentionCutoff(): number {
  const days = useBrowserPrefsStore.getState().historyRetentionDays;
  return days > 0 ? Date.now() - days * DAY_MS : 0;
}

/** batches the single large history value and ignores storage exhaustion. */
function deferredLocalStorage(): StateStorage {
  const pending = new Map<string, string>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    for (const [name, value] of pending) {
      try {
        localStorage.setItem(name, value);
      } catch {
        // Quota or private mode: keep the in-memory history.
      }
    }
    pending.clear();
  };
  if (typeof window !== "undefined") window.addEventListener("pagehide", flush);
  return {
    getItem: (name) => pending.get(name) ?? localStorage.getItem(name),
    setItem: (name, value) => {
      pending.set(name, value);
      timer ??= setTimeout(flush, PERSIST_DELAY_MS);
    },
    removeItem: (name) => {
      pending.delete(name);
      localStorage.removeItem(name);
    },
  };
}

let nextId = 0;
const newId = () => `${Date.now().toString(36)}-${(nextId++).toString(36)}`;

interface BrowserHistoryState {
  history: HistoryItem[];
  downloads: DownloadItem[];
  /** host icons ordered oldest to newest. */
  icons: Record<string, string>;
  recordVisit: (url: string, title: string, temporary?: boolean) => void;
  recordIcon: (host: string, icon: string, temporary?: boolean) => void;
  recordDownload: (item: Omit<DownloadItem, "id" | "downloadedAt">, temporary?: boolean) => void;
  removeVisit: (id: string) => void;
  removeVisits: (ids: ReadonlySet<string>) => void;
  removeDownload: (id: string) => void;
  clearHistory: () => void;
  clearDownloads: () => void;
  /** drops visits outside the Browser retention setting. */
  pruneHistory: () => void;
}

export const useBrowserHistoryStore = create<BrowserHistoryState>()(
  persist(
    (set) => ({
      history: [],
      downloads: [],
      icons: {},
      recordIcon: (host, icon, temporary = false) =>
        set((state) => {
          if (!host || icon.length > MAX_URL_CHARS || state.icons[host] === icon) return state;
          // icons expose visited hosts, so they follow history privacy settings.
          if (!savesHistory(temporary)) return state;
          const { [host]: _replaced, ...rest } = state.icons;
          const hosts = Object.keys(rest);
          for (const old of hosts.slice(0, Math.max(0, hosts.length + 1 - MAX_ICONS))) delete rest[old];
          return { icons: { ...rest, [host]: icon } };
        }),
      recordVisit: (url, fullTitle, temporary = false) =>
        set((state) => {
          if (url.length > MAX_URL_CHARS || !savesHistory(temporary)) return state;
          const title = fullTitle.slice(0, MAX_TITLE_CHARS);
          const cutoff = retentionCutoff();
          const kept = cutoff ? state.history.filter((visit) => visit.visitedAt >= cutoff) : state.history;
          const [latest, ...rest] = kept;
          // reloads and title updates remain a single visit.
          const history =
            latest?.url === url
              ? [{ ...latest, title: title || latest.title, visitedAt: Date.now() }, ...rest]
              : [{ id: newId(), url, title, visitedAt: Date.now() }, ...kept].slice(0, MAX_HISTORY);
          // prune icons against the new visit because its icon was recorded first.
          const icons = kept.length < state.history.length ? iconsFor(history, state.icons) : state.icons;
          return { history, icons };
        }),
      recordDownload: (item, temporary = useChatRuntimeStore.getState().incognito) =>
        set((state) => {
          if (!useBrowserPrefsStore.getState().saveDownloadHistory || temporary) {
            if (item.nativeId) forgetNativeDownloads([item.nativeId]);
            return state;
          }
          // bound page-controlled fields while retaining downloads with oversized URLs.
          const entry = {
            ...item,
            name: item.name.slice(0, MAX_TITLE_CHARS),
            url: item.url !== null && item.url.length <= MAX_URL_CHARS ? item.url : null,
            contentType: item.contentType.slice(0, MAX_TITLE_CHARS),
            id: newId(),
            downloadedAt: Date.now(),
          };
          const downloads = [entry, ...state.downloads];
          const dropped = downloads.slice(MAX_DOWNLOADS).flatMap((item) => (item.nativeId ? [item.nativeId] : []));
          forgetNativeDownloads(dropped);
          return { downloads: downloads.slice(0, MAX_DOWNLOADS) };
        }),
      removeVisit: (id) => set((state) => ({ history: state.history.filter((item) => item.id !== id) })),
      removeVisits: (ids) => set((state) => ({ history: state.history.filter((item) => !ids.has(item.id)) })),
      removeDownload: (id) =>
        set((state) => {
          const nativeId = state.downloads.find((item) => item.id === id)?.nativeId;
          if (nativeId) forgetNativeDownloads([nativeId]);
          return { downloads: state.downloads.filter((item) => item.id !== id) };
        }),
      clearHistory: () => set({ history: [], icons: {} }),
      clearDownloads: () =>
        set((state) => {
          // The app's registry is shared by every account; forget only this one's.
          forgetNativeDownloads(state.downloads.flatMap((item) => (item.nativeId ? [item.nativeId] : [])));
          return { downloads: [] };
        }),
      pruneHistory: () =>
        set((state) => {
          const cutoff = retentionCutoff();
          if (!cutoff || state.history.every((visit) => visit.visitedAt >= cutoff)) return state;
          const history = state.history.filter((visit) => visit.visitedAt >= cutoff);
          return { history, icons: iconsFor(history, state.icons) };
        }),
    }),
    {
      // Per account: a write still deferred at a switch lands under the account that made it.
      name: accountDatabaseName("unsloth_browser_history"),
      version: 1,
      storage: createJSONStorage(deferredLocalStorage),
    },
  ),
);

const PRUNE_INTERVAL_MS = 60 * 60 * 1000;

// Prune at startup (storage loads synchronously), hourly while open, and when retention shortens.
useBrowserHistoryStore.getState().pruneHistory();
if (typeof window !== "undefined" && typeof window.setInterval === "function") {
  window.setInterval(() => useBrowserHistoryStore.getState().pruneHistory(), PRUNE_INTERVAL_MS);
}
useBrowserPrefsStore.subscribe((state, previous) => {
  if (state.historyRetentionDays !== previous.historyRetentionDays) useBrowserHistoryStore.getState().pruneHistory();
});
