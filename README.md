# GPX Drawing Visualizer

Simple Python prototype for visualizing GPX routes as clean line drawings suitable for artistic tracing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the visualizer
python visualize_gpx.py
```

This will:
- Parse `route19668100.gpx`
- Compute distances using the Haversine formula
- Generate dummy elevation data (until real elevation service is integrated)
- Create two visualizations:
  - **3D view**: Interactive route with elevation in 3D space
  - **2D profile**: Clean elevation profile as a filled area chart
- Open both in your browser automatically
- Save HTML files you can share or reference

## Output Files

- `gpx_visualization_3d.html` - Interactive 3D route visualization
- `gpx_elevation_profile_2d.html` - 2D elevation profile

## Next Steps

- Replace dummy elevation data with real elevation API (e.g., Open-Elevation, Google Elevation API)
- Export as SVG for direct use in illustration software
- Add distance markers along the route
- Customize colors and styling for your artistic needs

## Why Python?

This prototype uses:
- **gpxpy**: Simple GPX parsing
- **numpy**: Efficient distance calculations
- **plotly**: Clean, interactive visualizations

Much simpler than the Three.js approach for creating traceable line drawings!

