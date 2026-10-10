"""Per-measurement quality flag bits (bitwise OR-able)."""

FLAG_CENTROID = 1  # centroid failed or ran away; WCS position used instead
FLAG_SATURATED = 2  # saturated pixel within the aperture radius
FLAG_EDGE = 4  # aperture/annulus clipped by the frame edge
FLAG_FIT = 8  # PSF fit flagged by photutils or returned a non-finite flux
FLAG_NOFLUX = 16  # relative flux undefined: non-finite target or ensemble <= 0
