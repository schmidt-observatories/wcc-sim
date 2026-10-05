import numpy as np
import pytest

from tests.conftest import DEC0, RA0
from wcc_phot.geometry import geometry_from_r_ap
from wcc_phot.io import load_frame
from wcc_phot.select import pick_references, pick_target

GEOM = geometry_from_r_ap(6.0, centroid_box=13, fit_shape=13)


def test_fixture_saturation_pattern(phot_frames):
    """Guard: the fixture's saturation layout is what the tests assume."""
    sat = np.asarray(phot_frames[0].catalog["saturated"], dtype=bool)
    assert bool(sat[11])
    assert not sat[np.arange(len(sat)) != 11].any()


def test_pick_target_by_source_id(phot_frames):
    catalog = phot_frames[0].catalog
    assert pick_target(catalog, 0) == 0
    assert pick_target(catalog, np.int64(5)) == 5
    with pytest.raises(ValueError, match="source_id"):
        pick_target(catalog, 999)


def test_pick_target_by_radec(phot_frames):
    catalog = phot_frames[0].catalog
    assert pick_target(catalog, (RA0, DEC0)) == 0
    with pytest.raises(ValueError, match="no catalog source"):
        pick_target(catalog, (RA0 + 1.0, DEC0))


def test_pick_references_best_ten(phot_frames):
    frame = load_frame(phot_frames[0])
    refs = pick_references(
        frame.catalog, 0, frame.image_e.shape, GEOM, n_ref=10
    )
    # saturated (11), close pair (12, 13), and edge (14) stars rejected;
    # survivors ranked by |G - G_target| = 0.15 * index
    assert refs == list(range(1, 11))


def test_pick_references_warns_when_short(phot_frames):
    frame = load_frame(phot_frames[0])
    with pytest.warns(UserWarning, match="only 10 of 12"):
        refs = pick_references(
            frame.catalog, 0, frame.image_e.shape, GEOM, n_ref=12
        )
    assert len(refs) == 10


def test_pick_references_none_left_raises(phot_frames):
    frame = load_frame(phot_frames[0])
    # a huge annulus pushes every star inside the edge margin
    huge = geometry_from_r_ap(6.0, r_in=9.0, r_out=200.0, centroid_box=13)
    with pytest.raises(ValueError, match="no usable reference stars"):
        pick_references(frame.catalog, 0, frame.image_e.shape, huge)


def test_pick_target_uses_spherical_separation_across_the_ra_wrap():
    """(0.0001, 0) and (359.9999, 0) are 0.72 arcsec apart; a flat RA
    difference called it 360 degrees. A row without a position never wins."""
    from astropy.table import Table

    catalog = Table({
        "source_id": [1, 2, 3],
        "ra": [359.9999, np.nan, 10.0],
        "dec": [0.0, 0.0, 0.0],
    })
    assert pick_target(catalog, (0.0001, 0.0)) == 0
    with pytest.raises(ValueError, match="no catalog source within"):
        pick_target(catalog, (180.0, 0.0))
