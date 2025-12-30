from sentinel.rag.manifest import IngestManifest, file_digest


def test_unknown_file_is_not_current(tmp_path):
    log = tmp_path / "a.jsonl"
    log.write_text("one\n", encoding="utf-8")
    assert not IngestManifest(tmp_path / "manifest.json").is_current(log)


def test_recorded_file_is_current_until_it_changes(tmp_path):
    log = tmp_path / "a.jsonl"
    log.write_text("one\n", encoding="utf-8")
    manifest = IngestManifest(tmp_path / "manifest.json")
    manifest.record(log)
    assert manifest.is_current(log)
    log.write_text("one\ntwo\n", encoding="utf-8")
    assert not manifest.is_current(log)


def test_manifest_survives_a_reload(tmp_path):
    log = tmp_path / "a.jsonl"
    log.write_text("x", encoding="utf-8")
    first = IngestManifest(tmp_path / "state" / "manifest.json")
    first.record(log)
    first.save()
    second = IngestManifest(tmp_path / "state" / "manifest.json")
    assert second.is_current(log)
    assert len(second) == 1


def test_corrupt_manifest_is_ignored(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(IngestManifest(path)) == 0


def test_digest_is_content_based(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    assert file_digest(a) == file_digest(b)
