// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat";
import { getLocale, translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { DownloadCancelledError, downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fileNameFromUrl, isWebUrl, safeDownloadName, withBaseUrl } from "./address";
import { type BrowserPage, fetchBrowserPage } from "./api";
import { approveDownload } from "./download-approval-queue";
import { isDangerousDownload } from "./download-safety";
import { useBrowserHistoryStore } from "./history-store";
import { type SavedNativeDownload, saveNativeDownload } from "./native-downloads";
import { useBrowserPrefsStore } from "./prefs-store";

export type BrowserDownload = {
  blob: Blob;
  name: string;
  contentType: string;
  url: string | null;
  site?: string;
  temporary?: boolean;
};

export function screenshotDownload(blob: Blob, name: string, temporary: boolean): BrowserDownload {
  return { blob, name, contentType: "image/png", url: null, temporary };
}

type SaveHandle = {
  name: string;
  createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }>;
};
type SaveFilePicker = (options: { suggestedName: string }) => Promise<SaveHandle>;

function saveFilePicker(): SaveFilePicker | null {
  const picker = (globalThis as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  return typeof picker === "function" ? picker : null;
}

export function canAskWhereToSave(): boolean {
  return isTauri || saveFilePicker() !== null;
}

function asksWhereToSave(): boolean {
  return !isTauri && saveFilePicker() !== null && useBrowserPrefsStore.getState().askWhereToSave;
}

/** True when Settings asks where to save but the click that started this has expired,
 *  so the dialog can't open without another one. */
export function saveNeedsClick(): boolean {
  return asksWhereToSave() && !navigator.userActivation?.isActive;
}

/** The browser's save dialog when Settings asks for it; null when off or it can't open
 *  (e.g. no recent click). Throws DownloadCancelledError if the reader cancels. */
async function pickSaveTarget(name: string): Promise<SaveHandle | null> {
  const picker = saveFilePicker();
  if (!picker || !asksWhereToSave()) return null;
  try {
    return await picker({ suggestedName: safeDownloadName(name) });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new DownloadCancelledError();
    return null;
  }
}

function approved(url: string | null, name: string, site?: string): Promise<boolean> {
  return url && isWebUrl(url) ? approveDownload(url, name, site ?? url) : Promise.resolve(true);
}

/** waits for website approval; target is preselected, null disables picking, and undefined follows Settings. */
export async function saveBrowserDownload(download: BrowserDownload, target?: SaveHandle | null): Promise<void> {
  const temporary = download.temporary ?? useChatRuntimeStore.getState().incognito;
  if (target === undefined) {
    if (!(await approved(download.url, download.name, download.site))) return;
    // late approval loses the user gesture, so Save supplies a fresh click.
    if (saveNeedsClick()) {
      const locale = getLocale();
      toast(translate("browser.downloadPrompt.ready", { name: safeDownloadName(download.name) }, locale), {
        action: {
          label: translate("browser.downloadPrompt.save", {}, locale),
          onClick: () => void writeDownload(download, undefined, temporary),
        },
      });
      return;
    }
  }
  await writeDownload(download, target, temporary);
}

async function writeDownload(
  download: BrowserDownload,
  target: SaveHandle | null | undefined,
  temporary: boolean,
): Promise<void> {
  const { blob, contentType, url } = download;
  const name = safeDownloadName(download.name);
  let saved: SavedNativeDownload | null = null;
  let picked: SaveHandle | null = null;
  try {
    if (isTauri) {
      // retain the native path so Download history can reveal the file.
      saved = await saveNativeDownload(blob, name, useBrowserPrefsStore.getState().askWhereToSave, url);
      if (!saved) return;
    } else {
      picked = target === undefined ? await pickSaveTarget(name) : target;
      if (picked) {
        const writable = await picked.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        await downloadFile(blob, name, contentType || undefined);
      }
    }
  } catch (error) {
    if (!isDownloadCancelled(error)) toast.error(error instanceof Error ? error.message : String(error));
    return;
  }
  useBrowserHistoryStore.getState().recordDownload(
    {
      name: saved?.name || picked?.name || name,
      url,
      size: blob.size,
      contentType,
      nativeId: saved?.id,
    },
    temporary,
  );
  if (saved?.marked === false) {
    toast.warning(translate("browser.native.notMarked", { name: saved.name || name }, getLocale()));
  }
}

// stays within the roughly five-second save-dialog gesture window.
const RESOLVE_BEFORE_ASK_MS = 1000;

/** uses the server filename for files and .html for pages. */
function linkDownload(page: BrowserPage, url: string, temporary: boolean): BrowserDownload {
  if (page.kind === "raw") {
    return {
      blob: page.blob,
      name: page.fileName ?? fileNameFromUrl(page.url),
      contentType: page.contentType,
      url,
      temporary,
    };
  }
  const name = fileNameFromUrl(page.url);
  return {
    blob: new Blob([withBaseUrl(page.html, page.base)], { type: "text/html" }),
    name: /\.html?$/i.test(name) ? name : `${name}.html`,
    contentType: "text/html",
    url,
    temporary,
  };
}

/** fetches through the panel proxy for cross-site support. */
export async function saveLinkAs(url: string): Promise<void> {
  const temporary = useChatRuntimeStore.getState().incognito;
  const controller = new AbortController();
  const pending = fetchBrowserPage({ url }, controller.signal).then((page) => linkDownload(page, url, temporary));
  // a slow fetch outlasts the menu gesture, so use its filename only when resolution is quick.
  let target: SaveHandle | null | undefined;
  let asked: string | undefined;
  if (asksWhereToSave()) {
    const quick = await Promise.race([
      pending.then(
        (download) => ({ download }),
        (error: unknown) => ({ error }),
      ),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), RESOLVE_BEFORE_ASK_MS)),
    ]);
    // avoid asking for a filename when the fetch has already failed.
    if (quick && "error" in quick) throw quick.error;
    const name = quick?.download.name ?? fileNameFromUrl(url);
    asked = name;
    try {
      if (!(await approved(url, name))) throw new DownloadCancelledError();
      target = await pickSaveTarget(name);
    } catch (error) {
      // No save after all: stop the fetch, which the backend drops on disconnect.
      controller.abort();
      pending.catch(() => undefined);
      if (isDownloadCancelled(error)) return;
      throw error;
    }
  }
  const download = await pending;
  // Approved under the address's name: a file that turns out to run code asks again, by its own name.
  if (asked !== undefined && isDangerousDownload(download.name) && !isDangerousDownload(asked)) {
    if (!(await approved(url, download.name))) return;
  }
  await saveBrowserDownload(download, target);
}
