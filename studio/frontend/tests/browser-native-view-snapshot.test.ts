// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// minimal DOM for native-view.ts with one page placeholder and test-controlled overlays
const frames: (() => void)[] = [];
const mutations: (() => void)[] = [];
const overlays: {
  getBoundingClientRect: () => DOMRect;
  closest: () => null;
  querySelector: () => null;
}[] = [];
const rect = (x: number, y: number, width: number, height: number) =>
  ({
    left: x,
    top: y,
    right: x + width,
    bottom: y + height,
    width,
    height,
  }) as DOMRect;
let pageBox = rect(500, 100, 500, 600);
const placeholder = {
  style: {} as Record<string, string> & {
    removeProperty: (name: string) => void;
  },
  offsetParent: {},
  isConnected: true,
  getBoundingClientRect: () => pageBox,
  closest: () => null,
};
const rootVars = new Map<string, string>();
const rootStyle = {
  getPropertyValue: (name: string) => rootVars.get(name) ?? "",
  setProperty: (name: string, value: string) => rootVars.set(name, value),
  removeProperty: (name: string) => rootVars.delete(name),
};
placeholder.style.removeProperty = (name) => {
  delete placeholder.style[
    name.replace(/-(\w)/g, (_, letter: string) => letter.toUpperCase())
  ];
};
const noop = () => undefined;
Object.assign(globalThis, {
  window: Object.assign(globalThis, {
    innerWidth: 1000,
    addEventListener: noop,
    removeEventListener: noop,
  }),
  document: {
    documentElement: { style: rootStyle },
    body: {},
    querySelector: (selector: string) =>
      selector.startsWith("[data-native-page") ? placeholder : null,
    querySelectorAll: (selector: string) =>
      selector.startsWith(".chat-full-view-dock") ? [] : overlays,
  },
  CSS: { escape: (value: string) => value },
  DOMRect: class {
    constructor(x: number, y: number, width: number, height: number) {
      // biome-ignore lint/correctness/noConstructorReturn: a plain rect stands in for DOMRect
      return rect(x, y, width, height);
    }
  },
  requestAnimationFrame: (callback: () => void) => frames.push(callback),
  cancelAnimationFrame: noop,
  ResizeObserver: class {
    observe() {}
    unobserve() {}
    disconnect() {}
  },
  MutationObserver: class {
    constructor(callback: () => void) {
      mutations.push(callback);
    }
    observe() {}
    disconnect() {}
  },
});

type Call = { command: string; args?: Record<string, unknown> };
const calls: Call[] = [];
let captureDone: ((bytes: ArrayBuffer) => void) | null = null;
(globalThis as { nativeViewCall?: unknown }).nativeViewCall = (
  command: string,
  args?: Record<string, unknown>,
) => {
  calls.push({ command, args });
  if (command === "browser_capture")
    return new Promise((resolve) => (captureDone = resolve));
  if (
    (command === "browser_view_navigate" || command === "browser_view_validate_url") &&
    (globalThis as { rejectNativeNavigation?: boolean }).rejectNativeNavigation
  ) {
    return Promise.reject(new Error("refused for test"));
  }
  return Promise.resolve();
};

register("./helpers/browser-store-resolver.mjs", import.meta.url);
register("./helpers/native-view-resolver.mjs", import.meta.url);
const { currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");
const { useChatRuntimeStore } = await import("@/features/chat");
const { nativePageTemporary, returnToNativePage, startNativeViews } = await import(
  "../src/features/browser/native-view.ts"
);

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
// apply overlay mutations before their scheduled frame
async function frame(): Promise<void> {
  for (const callback of mutations) callback();
  for (const callback of frames.splice(0)) callback();
  await settle();
}
const menu = {
  getBoundingClientRect: () => rect(900, 100, 100, 200),
  closest: () => null,
  querySelector: () => null,
};

test("a menu that closes and reopens while the page is captured keeps the snapshot", async () => {
  useBrowserStore.getState().openUrl("https://example.com/");
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId;
    assert.ok(
      calls.some(
        ({ command, args }) =>
          command === "browser_view_show" && args?.tabId === tabId,
      ),
    );

    overlays.push(menu);
    await frame();
    assert.ok(captureDone, "the covered page is captured before it hides");
    // close and reopen the menu before capture resolves
    overlays.length = 0;
    await frame();
    overlays.push(menu);
    await frame();
    captureDone?.(new Uint8Array([137, 80, 78, 71]).buffer);
    await settle();
    await frame();

    assert.match(placeholder.style.backgroundImage ?? "", /^url\(blob:/);
    const shows = calls.filter(
      ({ command }) => command === "browser_view_show",
    );
    assert.equal(
      shows.at(-1)?.args?.tabId,
      null,
      "the page stays hidden under the menu",
    );

    overlays.length = 0;
    await frame();
    assert.equal(
      placeholder.style.backgroundImage,
      undefined,
      "the snapshot goes once the page shows",
    );
  } finally {
    stop();
  }
});

test("toasts move left of a page that sits beside the Run settings panel", async () => {
  const stop = startNativeViews();
  try {
    pageBox = rect(500, 100, 500, 600);
    await frame();
    assert.equal(rootVars.get("--studio-browser-page-inset"), "500px");

    // ignore pages away from the window edge because they do not constrain toasts
    pageBox = rect(400, 100, 300, 600);
    await frame();
    assert.equal(rootVars.has("--studio-browser-page-inset"), false);

    // include the 300px Run settings panel between the page and window edge
    rootVars.set("--studio-chat-settings-inset", "300px");
    await frame();
    assert.equal(rootVars.get("--studio-browser-page-inset"), "600px");

    // omit the inset when no toast column fits; an overlapping toast will hide the page instead
    pageBox = rect(200, 100, 500, 600);
    await frame();
    assert.equal(rootVars.has("--studio-browser-page-inset"), false);
  } finally {
    stop();
    rootVars.clear();
    pageBox = rect(500, 100, 500, 600);
  }
});

test("a finished download is listed and reported after its tab closed, and warns when it isn't marked", async () => {
  const stop = startNativeViews();
  try {
    useBrowserStore.getState().openUrl("https://example.org/", { newTab: true });
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    useBrowserStore.getState().closeTab(tabId);
    await frame();
    const seen: { level: string; message: string }[] = [];
    const g = globalThis as {
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown;
    };
    g.nativeViewSeen = seen;
    const done = {
      kind: "download",
      tabId,
      url: "https://example.com/a.zip",
      name: "a.zip",
      path: null,
      size: 3,
      done: true,
      success: true,
    };
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d1", marked: true } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d2", marked: false } });
    g.nativeViewListener?.({ payload: { ...done, downloadId: "d3", marked: null } });
    // this page never opened the tab; a prior account started it before the last reload.
    g.nativeViewListener?.({ payload: { ...done, tabId: "tab-before-reload", downloadId: "d4", marked: true } });
    assert.deepEqual(seen, [
      { level: "history", message: "d1" },
      { level: "success", message: "browser.native.downloaded" },
      { level: "history", message: "d2" },
      { level: "warning", message: "browser.native.notMarked" },
      { level: "history", message: "d3" },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    stop();
  }
});

test("native navigation keeps temporary history private after the chat mode changes", async () => {
  useBrowserStore.getState().openUrl("https://normal.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown[];
    };
    g.nativeViewSeen = [];
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: { kind: "load", tabId, url: "https://private.example/", loading: true },
    });
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({
      payload: { kind: "load", tabId, url: "https://private.example/", loading: false },
    });
    g.nativeViewListener?.({ payload: { kind: "title", tabId, title: "Private" } });
    assert.deepEqual(g.nativeViewSeen, [
      { level: "visit", message: "https://private.example/", temporary: true },
      { level: "visit", message: "https://private.example/", temporary: true },
    ]);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("a native pushState change latches temporary provenance without a load event", async () => {
  useBrowserStore.getState().openUrl("https://spa.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown[];
    };
    g.nativeViewSeen = [];
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({ payload: { kind: "url", tabId, url: "https://spa.example/private" } });
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({ payload: { kind: "title", tabId, title: "Private route" } });
    assert.equal(nativePageTemporary(tabId), true);
    assert.deepEqual(g.nativeViewSeen, [
      { level: "visit", message: "https://spa.example/private", temporary: true },
    ]);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("a native privacy boundary starts a fresh history before showing a persistent page", async () => {
  useChatRuntimeStore.getState().setIncognito(true);
  useBrowserStore.getState().openUrl("https://private-history.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const afterPrivate = calls.length;
    useChatRuntimeStore.getState().setIncognito(false);
    useBrowserStore.getState().navigate(tabId, { url: "https://persistent-history.example/" });
    await frame();
    const boundary = calls.slice(afterPrivate);
    const closed = boundary.findIndex(({ command }) => command === "browser_view_close");
    const shown = boundary.findIndex(
      ({ command, args }) =>
        command === "browser_view_show" &&
        args?.tabId === tabId &&
        args?.url === "https://persistent-history.example/",
    );
    assert.ok(closed >= 0 && shown > closed);
    const validated = boundary.findIndex(({ command }) => command === "browser_view_validate_url");
    assert.ok(validated >= 0 && validated < closed);
    assert.equal(boundary.some(({ command }) => command === "browser_view_navigate"), false);
    assert.equal(nativePageTemporary(tabId), false);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("a native child tab inherits retained temporary page provenance", async () => {
  useBrowserStore.getState().openUrl("https://normal-parent.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as { nativeViewListener?: (event: { payload: unknown }) => void };
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: { kind: "load", tabId, url: "https://private-parent.example/", loading: true },
    });
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({
      payload: { kind: "newTab", tabId, url: "https://private-child.example/" },
    });
    const child = useBrowserStore
      .getState()
      .tabs.find((candidate) => candidate.id === useBrowserStore.getState().activeTabId);
    const childEntry = child ? currentEntry(child) : null;
    assert.equal(childEntry?.kind === "web" && childEntry.temporary, true);
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("a failed native navigation restores the displayed page's temporary provenance", async () => {
  useBrowserStore.getState().openUrl("https://normal-before-private.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeViewListener?: (event: { payload: unknown }) => void;
      rejectNativeNavigation?: boolean;
    };
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: { kind: "load", tabId, url: "https://private-shown.example/", loading: true },
    });
    useChatRuntimeStore.getState().setIncognito(false);
    g.rejectNativeNavigation = true;
    useBrowserStore.getState().navigate(tabId, { url: "https://refused.example/" });
    await frame();
    g.rejectNativeNavigation = false;
    assert.equal(returnToNativePage(tabId), true);
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
    const entry = tab ? currentEntry(tab) : null;
    assert.equal(entry?.kind === "web" && entry.temporary, true);
  } finally {
    (globalThis as { rejectNativeNavigation?: boolean }).rejectNativeNavigation = undefined;
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("native downloads inherit a temporary page navigation after chat mode changes", async () => {
  useBrowserStore.getState().openUrl("https://normal.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeDownloadAllowed?: boolean;
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown[];
    };
    g.nativeViewSeen = [];
    g.nativeDownloadAllowed = true;
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: { kind: "load", tabId, url: "https://private.example/", loading: true },
    });
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({
      payload: {
        kind: "downloadPrompt",
        tabId,
        url: "https://private.example/private.zip",
        site: "https://private.example/",
        name: "private.zip",
        id: "private-page-download",
      },
    });
    await settle();
    g.nativeViewListener?.({
      payload: {
        kind: "download",
        tabId,
        url: "https://private.example/private.zip",
        name: "private.zip",
        path: null,
        size: 3,
        done: true,
        success: true,
        requestId: "private-page-download",
        downloadId: "native-private-page",
        marked: true,
      },
    });
    assert.deepEqual(g.nativeViewSeen, [
      { level: "info", message: "browser.native.downloading" },
      { level: "history", message: "native-private-page", temporary: true },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    delete (globalThis as { nativeDownloadAllowed?: boolean }).nativeDownloadAllowed;
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("native download completion keeps the chat mode from its prompt", async () => {
  useBrowserStore.getState().openUrl("https://download.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeDownloadAllowed?: boolean;
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown[];
    };
    g.nativeViewSeen = [];
    g.nativeDownloadAllowed = true;
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: {
        kind: "downloadPrompt",
        tabId,
        url: "https://download.example/private.zip",
        site: "https://download.example/",
        name: "private.zip",
        id: "private-download",
      },
    });
    await settle();
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({
      payload: {
        kind: "download",
        tabId,
        url: "https://download.example/private.zip",
        name: "private.zip",
        path: null,
        size: 3,
        done: true,
        success: true,
        requestId: "private-download",
        downloadId: "native-private",
        marked: true,
      },
    });
    assert.deepEqual(g.nativeViewSeen, [
      { level: "info", message: "browser.native.downloading" },
      { level: "history", message: "native-private", temporary: true },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    delete (globalThis as { nativeDownloadAllowed?: boolean }).nativeDownloadAllowed;
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});

test("cancelling a native save retires its captured chat context", async () => {
  useBrowserStore.getState().openUrl("https://download.example/", { newTab: true });
  const stop = startNativeViews();
  try {
    await frame();
    const tabId = useBrowserStore.getState().activeTabId as string;
    const g = globalThis as {
      nativeDownloadAllowed?: boolean;
      nativeViewListener?: (event: { payload: unknown }) => void;
      nativeViewSeen?: unknown[];
    };
    g.nativeViewSeen = [];
    g.nativeDownloadAllowed = true;
    useChatRuntimeStore.getState().setIncognito(true);
    g.nativeViewListener?.({
      payload: {
        kind: "downloadPrompt",
        tabId,
        url: "https://download.example/cancelled.zip",
        site: "https://download.example/",
        name: "cancelled.zip",
        id: "cancelled-download",
      },
    });
    await settle();
    useChatRuntimeStore.getState().setIncognito(false);
    g.nativeViewListener?.({
      payload: { kind: "downloadCancelled", tabId, requestId: "cancelled-download" },
    });
    // a late terminal event is unexpected but exposes the retired context.
    g.nativeViewListener?.({
      payload: {
        kind: "download",
        tabId,
        url: "https://download.example/cancelled.zip",
        name: "cancelled.zip",
        path: null,
        size: 3,
        done: true,
        success: true,
        requestId: "cancelled-download",
        downloadId: "native-cancelled",
        marked: true,
      },
    });
    assert.deepEqual(g.nativeViewSeen, [
      { level: "info", message: "browser.native.downloading" },
      { level: "history", message: "native-cancelled" },
      { level: "success", message: "browser.native.downloaded" },
    ]);
  } finally {
    delete (globalThis as { nativeDownloadAllowed?: boolean }).nativeDownloadAllowed;
    useChatRuntimeStore.getState().setIncognito(false);
    stop();
  }
});
