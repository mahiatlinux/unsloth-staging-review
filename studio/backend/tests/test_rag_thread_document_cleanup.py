# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import multiprocessing
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.rag import folder_sync, ingestion, job_leases, store
from routes import chat_history, rag as rag_routes
from storage import rag_db, studio_db


def _hold_scope_lock_in_spawned_backend(scope, ready, acquired, release):
    from core.rag import folder_sync as child_folder_sync

    ready.set()
    with child_folder_sync.scope_lock(scope):
        acquired.set()
        if not release.wait(30):
            raise TimeoutError("parent did not release spawned scope-lock test")


@pytest.fixture
def client(rag_home, stub_embeddings):
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _create_thread(client, thread_id):
    response = client.post(
        "/api/chat/threads",
        json = {"id": thread_id, "title": "t", "modelType": "base", "createdAt": 1},
    )
    assert response.status_code == 200, response.text


def _upload(client, thread_id, name, text, *, temporary = False):
    response = client.post(
        f"/api/rag/threads/{thread_id}/documents",
        data = {"temporary": "true"} if temporary else None,
        files = {"file": (name, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    deadline = time.time() + 30
    while time.time() < deadline:
        status = ingestion.get_job_status(body["jobId"])
        if status and status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    return body["documentId"]


def _stored_path(document_id):
    conn = rag_db.get_connection()
    try:
        return store.get_document(conn, document_id)["stored_path"]
    finally:
        conn.close()


def _document_ids(client):
    response = client.get("/api/rag/documents")
    assert response.status_code == 200, response.text
    return {doc["id"] for doc in response.json()["documents"]}


def _temporary_scope_expiry(thread_id):
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT expires_at FROM temporary_thread_scopes WHERE scope=?",
            (store.thread_scope(thread_id),),
        ).fetchone()
        return row["expires_at"] if row is not None else None
    finally:
        conn.close()


def _expire_temporary_scope(thread_id):
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE temporary_thread_scopes SET expires_at=? WHERE scope=?",
            ("2000-01-01T00:00:00+00:00", store.thread_scope(thread_id)),
        )
        conn.commit()
    finally:
        conn.close()


def _chunk_count(thread_id):
    conn = rag_db.get_metadata_connection()
    try:
        scope = store.thread_scope(thread_id)
        chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE scope=?", (scope,)).fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE scope=?", (scope,)).fetchone()[0]
        return chunks + fts
    finally:
        conn.close()


def test_deleting_a_thread_removes_its_uploaded_documents(client):
    _create_thread(client, "doomed")
    _create_thread(client, "kept")
    doomed = [
        _upload(client, "doomed", f"doomed{i}.txt", f"alpha bravo charlie {i} " * 50)
        for i in range(2)
    ]
    kept = _upload(client, "kept", "kept.txt", "delta echo foxtrot " * 50)
    doomed_paths, kept_path = [_stored_path(d) for d in doomed], _stored_path(kept)
    assert all(os.path.isfile(path) for path in doomed_paths)

    response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["doomed"]})

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {kept}
    assert _chunk_count("doomed") == 0
    assert not any(os.path.exists(path) for path in doomed_paths)
    assert os.path.isfile(kept_path)
    assert _chunk_count("kept") > 0


def test_clearing_history_removes_every_threads_uploaded_documents(client):
    paths = []
    for thread_id in ("first", "second"):
        _create_thread(client, thread_id)
        paths.append(
            _stored_path(_upload(client, thread_id, f"{thread_id}.txt", f"{thread_id} words " * 50))
        )

    response = client.request("DELETE", "/api/chat")

    assert response.status_code == 200, response.text
    assert _document_ids(client) == set()
    assert not any(os.path.exists(path) for path in paths)


def test_upload_to_missing_thread_is_rejected(client):
    response = client.post(
        "/api/rag/threads/missing/documents",
        files = {"file": ("notes.txt", b"alpha bravo charlie", "text/plain")},
    )

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Thread not found"}
    assert _document_ids(client) == set()


def test_temporary_thread_upload_does_not_require_a_stored_row(client):
    document_id = _upload(
        client,
        "temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )

    assert _document_ids(client) == {document_id}
    assert _temporary_scope_expiry("temporary") is not None


def test_expired_temporary_thread_upload_is_reaped_after_session_loss(client):
    document_id = _upload(
        client,
        "abandoned-temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    path = _stored_path(document_id)
    _expire_temporary_scope("abandoned-temporary")

    folder_sync._enqueue_periodic()

    assert _document_ids(client) == set()
    assert _temporary_scope_expiry("abandoned-temporary") is None
    assert not os.path.exists(path)


def test_live_temporary_thread_heartbeat_renews_its_upload_lease(client):
    document_id = _upload(
        client,
        "live-temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    _expire_temporary_scope("live-temporary")

    response = client.post("/api/rag/threads/live-temporary/documents/lease")

    assert response.status_code == 200, response.text
    assert response.json()["active"] is True
    assert response.json()["renewAfterMs"] > 0
    assert _temporary_scope_expiry("live-temporary") > datetime.now(timezone.utc).isoformat()
    folder_sync._enqueue_periodic()
    assert _document_ids(client) == {document_id}


def test_temporary_thread_heartbeat_does_not_create_a_scope_lease(client):
    response = client.post("/api/rag/threads/never-uploaded/documents/lease")

    assert response.status_code == 200, response.text
    assert response.json()["active"] is False
    assert _temporary_scope_expiry("never-uploaded") is None


def test_saving_a_temporary_thread_promotes_its_uploads_before_expiry(client):
    document_id = _upload(
        client,
        "saved-temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    path = _stored_path(document_id)
    _create_thread(client, "saved-temporary")

    folder_sync._enqueue_periodic()

    assert _document_ids(client) == {document_id}
    assert _temporary_scope_expiry("saved-temporary") is None
    assert os.path.isfile(path)


def test_saving_a_temporary_thread_serializes_with_expiry(client, monkeypatch):
    document_id = _upload(
        client,
        "saving-temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    _expire_temporary_scope("saving-temporary")
    save_entered = threading.Event()
    allow_save = threading.Event()
    reap_attempted = threading.Event()
    reap_entered = threading.Event()
    actual_upsert = chat_history.upsert_chat_thread
    actual_scope_lock = folder_sync._scope_lock

    def paused_upsert(thread):
        save_entered.set()
        assert allow_save.wait(30)
        return actual_upsert(thread)

    def observed_scope_lock(scope):
        if threading.current_thread().name == "temporary-scope-reaper":
            reap_attempted.set()
        return actual_scope_lock(scope)

    def thread_exists(thread_id):
        reap_entered.set()
        return studio_db.get_chat_thread(thread_id) is not None

    monkeypatch.setattr(chat_history, "upsert_chat_thread", paused_upsert)
    monkeypatch.setattr(folder_sync, "_scope_lock", observed_scope_lock)
    responses = []
    saver = threading.Thread(
        target = lambda: responses.append(
            client.post(
                "/api/chat/threads",
                json = {
                    "id": "saving-temporary",
                    "title": "t",
                    "modelType": "base",
                    "createdAt": 1,
                },
            )
        )
    )
    saver.start()
    assert save_entered.wait(5)
    reaper = threading.Thread(
        target = lambda: folder_sync._reap_expired_temporary_thread_scopes(
            datetime.now(timezone.utc).isoformat(), thread_exists = thread_exists
        ),
        name = "temporary-scope-reaper",
    )
    reaper.start()
    try:
        assert reap_attempted.wait(10)
        assert not reap_entered.wait(0.1)
    finally:
        allow_save.set()
    saver.join(timeout = 10)
    reaper.join(timeout = 10)

    assert not saver.is_alive()
    assert not reaper.is_alive()
    assert responses[0].status_code == 200, responses[0].text
    assert _document_ids(client) == {document_id}
    assert _temporary_scope_expiry("saving-temporary") is None


def test_temporary_scope_lock_excludes_a_sibling_backend(rag_home):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    acquired = context.Event()
    release = context.Event()
    scope = store.thread_scope("cross-process-temporary")
    worker = context.Process(
        target = _hold_scope_lock_in_spawned_backend,
        args = (scope, ready, acquired, release),
    )

    try:
        with folder_sync.scope_lock(scope):
            worker.start()
            assert ready.wait(30)
            assert not acquired.wait(0.2)
        assert acquired.wait(30)
    finally:
        release.set()
        worker.join(timeout = 30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout = 5)

    assert worker.exitcode == 0


def test_expired_temporary_scope_waits_for_live_ingestion(client):
    document_id = _upload(
        client,
        "indexing-temporary",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    conn = rag_db.get_connection()
    try:
        job = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE document_id=?", (document_id,)
        ).fetchone()
        conn.execute("UPDATE ingestion_jobs SET status='running' WHERE id=?", (job["id"],))
        conn.execute(
            "INSERT OR REPLACE INTO rag_job_leases(kind, job_id, owner_id, expires_at) "
            "VALUES(?, ?, ?, ?)",
            (job_leases.INGESTION, job["id"], "live-worker", "2999-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "UPDATE temporary_thread_scopes SET expires_at=? WHERE scope=?",
            ("2000-01-01T00:00:00+00:00", store.thread_scope("indexing-temporary")),
        )
        conn.commit()
    finally:
        conn.close()

    folder_sync._enqueue_periodic()

    assert _document_ids(client) == {document_id}
    assert _temporary_scope_expiry("indexing-temporary") == "2000-01-01T00:00:00+00:00"


def test_temporary_thread_upload_is_rejected_after_deletion(client):
    studio_db.delete_chat_threads(["closed-temporary"])

    response = client.post(
        "/api/rag/threads/closed-temporary/documents",
        data = {"temporary": "true"},
        files = {"file": ("notes.txt", b"alpha bravo charlie", "text/plain")},
    )

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Thread not found"}
    assert _document_ids(client) == set()


def test_deleting_a_temporary_thread_removes_its_scope_lease(client):
    document_id = _upload(
        client,
        "closed-temporary-with-document",
        "notes.txt",
        "alpha bravo charlie " * 50,
        temporary = True,
    )
    path = _stored_path(document_id)

    response = client.request(
        "DELETE",
        "/api/chat/threads",
        json = {"ids": ["closed-temporary-with-document"]},
    )

    assert response.status_code == 200, response.text
    assert _document_ids(client) == set()
    assert _temporary_scope_expiry("closed-temporary-with-document") is None
    assert not os.path.exists(path)


def test_thread_cleanup_waits_for_an_upload_that_already_validated(
    client, monkeypatch, tmp_path
):
    from core.rag import conversation_archive

    _create_thread(client, "racing")
    cleanup_attempted = threading.Event()
    cleanup_entered = threading.Event()
    cleanup_threads = []

    monkeypatch.setattr(
        rag_routes,
        "_resolve_document_upload",
        lambda *_args: (str(tmp_path / "racing.txt"), "racing.txt", "0" * 64),
    )
    monkeypatch.setattr(conversation_archive, "delete_for_thread", lambda *_args, **_kw: 0)

    def delete_documents(*_args, **_kwargs):
        cleanup_entered.set()
        return 0

    monkeypatch.setattr(conversation_archive, "delete_thread_documents", delete_documents)

    def start_ingestion(*_args, **_kwargs):
        cutoff = datetime.now(timezone.utc).isoformat()
        studio_db.delete_chat_threads(["racing"])

        def cleanup():
            cleanup_attempted.set()
            chat_history._remove_thread_rag_data(["racing"], cutoff = cutoff)

        worker = threading.Thread(target = cleanup)
        cleanup_threads.append(worker)
        worker.start()
        assert cleanup_attempted.wait(1)
        assert not cleanup_entered.wait(0.1)
        return "document", "job"

    monkeypatch.setattr(ingestion, "start_ingestion", start_ingestion)

    response = client.post(
        "/api/rag/threads/racing/documents",
        files = {"file": ("notes.txt", b"ignored", "text/plain")},
    )

    assert response.status_code == 200, response.text
    cleanup_threads[0].join(timeout = 2)
    assert not cleanup_threads[0].is_alive()
    assert cleanup_entered.is_set()


def test_deleting_a_project_removes_its_member_threads_documents(client):
    response = client.post(
        "/api/chat/projects", json = {"id": "proj", "name": "p", "createdAt": 1, "updatedAt": 1}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/chat/threads",
        json = {
            "id": "member",
            "title": "t",
            "modelType": "base",
            "createdAt": 1,
            "projectId": "proj",
        },
    )
    assert response.status_code == 200, response.text
    path = _stored_path(_upload(client, "member", "member.txt", "golf hotel india " * 50))
    _create_thread(client, "outsider")
    outsider = _upload(client, "outsider", "outsider.txt", "sierra tango uniform " * 50)

    response = client.delete("/api/chat/projects/proj")

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {outsider}
    assert not os.path.exists(path)


def test_a_recreated_thread_keeps_documents_uploaded_after_the_cutoff(client):
    _create_thread(client, "recreated")
    old = _upload(client, "recreated", "old.txt", "juliet kilo lima " * 50)
    old_path = _stored_path(old)
    cutoff = datetime.now(timezone.utc).isoformat()
    fresh = _upload(client, "recreated", "fresh.txt", "mike november oscar " * 50)

    chat_history._remove_thread_rag_data(["recreated"])
    assert _document_ids(client) == {old, fresh}

    chat_history._remove_thread_rag_data(["recreated"], cutoff = cutoff)

    assert _document_ids(client) == {fresh}
    assert not os.path.exists(old_path)
    assert os.path.isfile(_stored_path(fresh))


def test_thread_documents_go_without_sqlite_vec(client, monkeypatch):
    from storage import studio_db

    _create_thread(client, "vecless")
    document_id = _upload(client, "vecless", "vecless.txt", "papa quebec romeo " * 50)
    path = _stored_path(document_id)
    studio_db.delete_chat_threads(["vecless"])

    def no_vec():
        raise rag_db.RagExtensionUnavailable("vec0 will not load")

    with monkeypatch.context() as patch:
        patch.setattr(rag_db, "get_connection", no_vec)
        chat_history._remove_thread_rag_data(["vecless"])

    assert _document_ids(client) == set()
    assert _chunk_count("vecless") == 0
    assert not os.path.exists(path)


def _add_message(client, thread_id, message_id):
    response = client.put(
        f"/api/chat/threads/{thread_id}/messages/{message_id}",
        json = {"id": message_id, "threadId": thread_id, "role": "user", "createdAt": 1},
    )
    assert response.status_code == 200, response.text


def _fork(client, thread_id, message_id, new_thread_id):
    response = client.post(
        f"/api/chat/threads/{thread_id}/fork",
        json = {"messageId": message_id, "newThreadId": new_thread_id, "createdAt": 2},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _thread_documents(client, thread_id):
    response = client.get(f"/api/rag/threads/{thread_id}/documents")
    assert response.status_code == 200, response.text
    return response.json()["documents"]


def _search(client, thread_id, query, mode):
    response = client.post(
        "/api/rag/search", json = {"query": query, "thread_id": thread_id, "mode": mode}
    )
    assert response.status_code == 200, response.text
    return response.json()["results"]


def test_forking_a_thread_copies_its_uploaded_documents(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    text = "victor whiskey xray " * 50
    source = _upload(client, "source", "source.txt", text)

    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None

    documents = _thread_documents(client, "fork")
    assert [(d["filename"], d["status"]) for d in documents] == [("source.txt", "completed")]
    copy = documents[0]["id"]
    assert copy != source
    assert documents[0]["numChunks"] == _thread_documents(client, "source")[0]["numChunks"]
    for mode in ("lexical", "dense"):
        hits = _search(client, "fork", "victor whiskey xray", mode)
        assert hits and {hit["documentId"] for hit in hits} == {copy}
    assert _stored_path(copy) != _stored_path(source)
    with open(_stored_path(copy), encoding = "utf-8") as copied:
        assert copied.read() == text


def _vectors(document_id):
    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT chunk_id, embedding FROM chunks_vec WHERE chunk_id LIKE ?",
            (f"{document_id}:%",),
        ).fetchall()
        return {row["chunk_id"].rsplit(":", 1)[1]: bytes(row["embedding"]) for row in rows}
    finally:
        conn.close()


def test_a_fork_copies_each_documents_own_vectors(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    sources = [
        _upload(client, "source", "short.txt", "kilo lima mike " * 50),
        _upload(client, "source", "long.txt", " ".join(f"november{i}" for i in range(3000))),
    ]
    _fork(client, "source", "m1", "fork")

    copies = {d["filename"]: d["id"] for d in _thread_documents(client, "fork")}
    assert len(_vectors(sources[1])) > 1
    for source, filename in zip(sources, ("short.txt", "long.txt")):
        assert _vectors(copies[filename]) == _vectors(source)


def test_a_forks_documents_are_independent_of_the_source_thread(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    _upload(client, "source", "source.txt", "yankee zulu alpha " * 50)
    _fork(client, "source", "m1", "fork")
    copy = _thread_documents(client, "fork")[0]["id"]
    copy_path = _stored_path(copy)

    _upload(client, "source", "later.txt", "bravo charlie delta " * 50)
    assert [d["id"] for d in _thread_documents(client, "fork")] == [copy]

    response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["source"]})

    assert response.status_code == 200, response.text
    assert _document_ids(client) == {copy}
    assert os.path.isfile(copy_path)
    assert _search(client, "fork", "yankee zulu alpha", "dense")


def test_a_fork_survives_documents_that_cannot_be_copied(client, monkeypatch):
    from core.rag import conversation_archive

    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)

    def broken(conn, *args, **kwargs):
        conn.execute("INSERT INTO chunks_fts(text, chunk_id, scope) VALUES('x', 'x', 'x')")
        raise RuntimeError("disk full")

    monkeypatch.setattr(conversation_archive.store, "copy_documents", broken)
    before = set(os.listdir(os.path.dirname(_stored_path(source))))

    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]

    assert "not copied" in (warning or "")

    assert _thread_documents(client, "fork") == []
    assert _document_ids(client) == {source}
    assert set(os.listdir(os.path.dirname(_stored_path(source)))) == before
    conn = rag_db.get_metadata_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE scope='x'").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_fork_warns_when_existing_documents_cannot_load_without_vec(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    with monkeypatch.context() as unavailable:
        unavailable.setattr(rag_db, "rag_available", lambda: False)
        warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")
    assert _thread_documents(client, "fork") == []
    assert _document_ids(client) == {source}


def test_a_fork_without_documents_needs_no_warning_without_vec(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None


def _add_document(thread_id, status):
    conn = rag_db.get_connection()
    try:
        store.create_document(
            conn,
            scope = store.thread_scope(thread_id),
            thread_id = thread_id,
            filename = f"{status}.txt",
            sha256 = f"{status}-upload",
            status = status,
        )
    finally:
        conn.close()


@pytest.mark.parametrize("status", ["pending", "running"])
def test_a_fork_warns_when_an_upload_is_still_indexing(client, status):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    _add_document("source", status)
    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")


@pytest.mark.parametrize("vec", [True, False])
def test_a_fork_does_not_warn_about_failed_uploads(client, monkeypatch, vec):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    _add_document("source", "failed")
    if not vec:
        monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None


def test_a_fork_copies_files_before_taking_the_rag_write_lock(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    for index in range(2):
        _upload(client, "source", f"source{index}.txt", f"echo foxtrot golf {index} " * 50)
    copy_upload = ingestion._copy_upload
    writable = []

    def copy_with_concurrent_writer(path):
        conn = rag_db.get_metadata_connection()
        try:
            conn.execute("PRAGMA busy_timeout = 20")
            conn.execute("BEGIN IMMEDIATE")
            writable.append(True)
        except sqlite3.OperationalError:
            writable.append(False)
        finally:
            conn.rollback()
            conn.close()
        return copy_upload(path)

    monkeypatch.setattr(ingestion, "_copy_upload", copy_with_concurrent_writer)
    assert _fork(client, "source", "m1", "fork")["containerSnapshotWarning"] is None
    assert writable == [True, True]


def test_a_fork_warns_if_a_source_document_is_deleted_during_file_copy(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    copy_upload = ingestion._copy_upload

    def copy_then_delete(path):
        copied = copy_upload(path)
        response = client.delete(f"/api/rag/documents/{source}")
        assert response.status_code == 200, response.text
        return copied

    monkeypatch.setattr(ingestion, "_copy_upload", copy_then_delete)
    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")
    assert _thread_documents(client, "fork") == []


def test_deleting_a_fork_during_file_copy_does_not_leave_orphan_documents(client, monkeypatch):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    upload_dir = os.path.dirname(_stored_path(source))
    before = set(os.listdir(upload_dir))
    copy_upload = ingestion._copy_upload

    def copy_then_delete_fork(path):
        copied = copy_upload(path)
        response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["fork"]})
        assert response.status_code == 200, response.text
        return copied

    monkeypatch.setattr(ingestion, "_copy_upload", copy_then_delete_fork)
    _fork(client, "source", "m1", "fork")
    assert _document_ids(client) == {source}
    assert set(os.listdir(upload_dir)) == before


def test_a_fork_warns_if_the_source_is_deleted_before_document_lookup(client, monkeypatch):
    from core.rag import conversation_archive

    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    _upload(client, "source", "source.txt", "echo foxtrot golf " * 50)
    copy_documents = conversation_archive.copy_thread_documents

    def delete_then_copy(source_thread_id, thread_id):
        response = client.request("DELETE", "/api/chat/threads", json = {"ids": ["source"]})
        assert response.status_code == 200, response.text
        return copy_documents(source_thread_id, thread_id)

    monkeypatch.setattr(conversation_archive, "copy_thread_documents", delete_then_copy)
    warning = _fork(client, "source", "m1", "fork")["containerSnapshotWarning"]
    assert "not copied" in (warning or "")
    assert _thread_documents(client, "fork") == []


def _cite(client, thread_id, message_id, document_id):
    sources = [
        {
            "citationId": 1,
            "chunkId": f"{document_id}:0",
            "documentId": document_id,
            "filename": "source.txt",
        }
    ]
    part = {
        "type": "tool-call",
        "toolCallId": "call-1",
        "toolName": "search_documents",
        "args": {"query": "hotel"},
        "result": "hotel india\n__RAG_SOURCES__:" + json.dumps(sources),
    }
    response = client.put(
        f"/api/chat/threads/{thread_id}/messages/{message_id}",
        json = {
            "id": message_id,
            "threadId": thread_id,
            "parentId": "m1",
            "role": "assistant",
            "content": [part],
            "metadata": {"custom": {"sources": sources}},
            "createdAt": 2,
        },
    )
    assert response.status_code == 200, response.text


def _cited(message):
    result = message["content"][0]["result"]
    return json.loads(result.split("__RAG_SOURCES__:", 1)[1])[0]


def test_a_forks_copied_messages_cite_the_forks_documents(client):
    _create_thread(client, "source")
    _add_message(client, "source", "m1")
    source = _upload(client, "source", "source.txt", "hotel india juliet " * 50)
    _cite(client, "source", "m2", source)

    forked = _fork(client, "source", "m2", "fork")

    copy = _thread_documents(client, "fork")[0]["id"]
    stored = client.get("/api/chat/threads/fork/messages").json()["messages"]
    for messages in (forked["messages"], stored):
        answer = [m for m in messages if m["role"] == "assistant"][0]
        cited = _cited(answer)
        assert (cited["documentId"], cited["chunkId"]) == (copy, f"{copy}:0")
        assert answer["metadata"]["custom"]["sources"][0]["documentId"] == copy
    parent = client.get("/api/chat/threads/source/messages/m2").json()
    assert (_cited(parent)["documentId"], _cited(parent)["chunkId"]) == (source, f"{source}:0")
    conn = studio_db.get_connection()
    try:
        assert conn.execute("SELECT dirty FROM chat_attachment_inventory_state").fetchone()[0] == 0
    finally:
        conn.close()
