"""The Gaussian KDE that replaced the last call into scipy.

scipy was 120 MB in the image for one optional density curve on one histogram,
so it is now eight lines of numpy. Eight lines of statistics nobody checks is
how a chart starts lying, hence this.
"""

import numpy as np
import pytest

from app.services.visualization_service import _gaussian_kde

pytestmark = pytest.mark.unit

# numpy renamed trapz to trapezoid in 2.0; requirements allow either.
_integrate = getattr(np, "trapezoid", None) or np.trapz


class TestGaussianKde:
    def test_too_few_points_is_no_curve(self):
        # None means "draw nothing"; a flat line would read as a real density.
        assert _gaussian_kde(np.array([1.0]), np.linspace(0, 2, 10)) is None
        assert _gaussian_kde(np.array([]), np.linspace(0, 2, 10)) is None

    def test_zero_spread_is_no_curve(self):
        assert _gaussian_kde(np.array([3.0, 3.0, 3.0]), np.linspace(0, 6, 10)) is None

    def test_nans_are_dropped_not_propagated(self):
        samples = np.array([1.0, 2.0, np.nan, 3.0])
        density = _gaussian_kde(samples, np.linspace(0, 4, 25))
        assert density is not None
        assert np.all(np.isfinite(density))

    def test_density_integrates_to_one(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=5.0, scale=2.0, size=500)
        grid = np.linspace(-10.0, 20.0, 3000)
        density = _gaussian_kde(samples, grid)
        assert _integrate(density, grid) == pytest.approx(1.0, abs=0.02)

    def test_peak_sits_at_the_mode(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(loc=5.0, scale=1.0, size=500)
        grid = np.linspace(0.0, 10.0, 1000)
        density = _gaussian_kde(samples, grid)
        assert grid[int(np.argmax(density))] == pytest.approx(5.0, abs=0.3)

    def test_bandwidth_follows_scotts_rule(self):
        # scipy's default: n ** (-1 / (d + 4)) scaled by the sample deviation.
        # Two samples that differ only in spread must produce curves whose
        # widths differ by the same factor.
        rng = np.random.default_rng(2)
        narrow = rng.normal(0.0, 1.0, 400)
        grid = np.linspace(-15.0, 15.0, 2000)
        d_narrow = _gaussian_kde(narrow, grid)
        d_wide = _gaussian_kde(narrow * 3.0, grid)
        width = lambda d: _integrate(d * grid**2, grid)  # noqa: E731
        assert width(d_wide) == pytest.approx(9.0 * width(d_narrow), rel=0.15)
