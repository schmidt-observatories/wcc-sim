"""Per-measurement quality flag bits (bitwise OR-able)."""

FLAG_CENTROID = 1  # centroid failed or ran away; WCS position used instead
FLAG_SATURATED = 2  # saturated pixel within the aperture radius
FLAG_EDGE = 4  # aperture/annulus clipped by the frame edge
