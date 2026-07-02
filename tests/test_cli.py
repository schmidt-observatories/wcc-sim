from astropy.io import fits

from tests.conftest import DEC0, RA0


def test_cli_writes_fits(monkeypatch, tmp_path, canned_catalog):
    from wcc_sim import catalog as catmod
    from wcc_sim import cli

    monkeypatch.setattr(
        catmod, "query_gaia",
        lambda *a, **k: canned_catalog,
    )
    # pipeline imported query_gaia by name; patch it there too
    from wcc_sim import pipeline

    monkeypatch.setattr(pipeline, "query_gaia", lambda *a, **k: canned_catalog)

    out = tmp_path / "cli.fits"
    rc = cli.main(
        [
            "--ra", str(RA0), "--dec", str(DEC0),
            "--sensorfilter", "zwo:r", "--focus", "0",
            "--exptime", "60", "--seed", "3",
            "--shape", "256", "256", "--stamp-npix", "33",
            "-o", str(out),
        ]
    )
    assert rc == 0
    with fits.open(out) as hdul:
        assert hdul["SCI"].header["EXPTIME"] == 60.0
        assert len(hdul["CAT"].data) == 5
