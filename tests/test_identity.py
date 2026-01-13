import numpy as np

from sentinel.tracking.identity import GalleryResolver, UnknownResolver, cosine_similarity, crop_box
from sentinel.tracking.tracker import Track


def make_track():
    return Track(1, (20.0, 20.0, 80.0, 80.0), 0.9, hits=3)


def test_unknown_resolver_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert UnknownResolver().resolve(frame, make_track()) is None


def test_crop_box_clips_to_frame():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop_box(frame, (-10.0, 90.0, 20.0, 130.0)).shape == (10, 20, 3)


def test_cosine_similarity_handles_zero_vectors():
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0


def test_gallery_resolver_matches_closest_identity():
    gallery = {"asha": np.array([1.0, 0.0]), "ravi": np.array([0.0, 1.0])}
    resolver = GalleryResolver(lambda face: np.array([0.9, 0.1]), gallery, threshold=0.8)
    assert resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), make_track()) == "asha"


def test_gallery_resolver_rejects_below_threshold():
    gallery = {"asha": np.array([1.0, 0.0])}
    resolver = GalleryResolver(lambda face: np.array([0.5, 0.9]), gallery, threshold=0.9)
    assert resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), make_track()) is None


def test_gallery_resolver_skips_empty_crop():
    resolver = GalleryResolver(lambda face: np.ones(2), {"a": np.ones(2)})
    far = Track(2, (500.0, 500.0, 560.0, 560.0), 0.9)
    assert resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), far) is None


def test_load_gallery_reads_named_embeddings(tmp_path):
    from sentinel.tracking.identity import load_gallery

    path = tmp_path / "gallery.npz"
    np.savez(path, asha=np.array([1.0, 0.0]), ravi=np.array([0.0, 1.0]))
    gallery = load_gallery(path)
    assert sorted(gallery) == ["asha", "ravi"]
    assert gallery["asha"].dtype == np.float32
