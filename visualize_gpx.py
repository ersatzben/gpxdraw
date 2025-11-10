#!/usr/bin/env python3
"""
Simple GPX visualizer with 3D route and 2D elevation profile.
Creates a clean line diagram suitable for artistic tracing.
"""

import gpxpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def parse_gpx_file(filename):
    """Parse GPX file and extract coordinates and elevation."""
    with open(filename, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    points = []
    elevations = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
                # Use elevation if available, otherwise 0
                elevations.append(point.elevation if point.elevation is not None else 0)
    
    return np.array(points), np.array(elevations)


def compute_distance_array(coords):
    """
    Compute cumulative distance along the route.
    Returns array of distances in kilometers.
    """
    distances = [0]
    
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(distances[-1] + dist)
    
    return np.array(distances)


def generate_dummy_elevation(distances, num_points):
    """
    Generate dummy elevation data with realistic-looking terrain.
    Uses sine waves and noise for varied terrain.
    """
    # Create varied terrain with multiple frequencies
    x = distances / distances[-1] * 4 * np.pi  # Normalized to full route
    
    # Combine multiple sine waves for realistic hills
    elevation = (
        50 * np.sin(x * 0.5) +           # Long rolling hills
        30 * np.sin(x * 1.5 + 1) +       # Medium hills
        15 * np.sin(x * 3 + 2) +         # Small undulations
        np.random.normal(0, 5, num_points)  # Random noise
    )
    
    # Add a gentle overall slope
    elevation += distances / distances[-1] * 20
    
    # Offset to make all values positive
    elevation += 100
    
    return elevation


def find_closest_point_index(coords, target_lat, target_lon):
    """Find the index of the closest point on the route to a given lat/lon."""
    distances = []
    for i, (lat, lon) in enumerate(coords):
        dist = haversine_distance(lat, lon, target_lat, target_lon)
        distances.append(dist)
    return np.argmin(distances)


def create_visualization(coords, distances, elevations, stops=None):
    """
    Create 3D visualization with route line and 2D elevation profile.
    
    Args:
        coords: Array of (lat, lon) coordinates
        distances: Cumulative distances along route
        elevations: Elevation data
        stops: Optional list of (lat, lon) tuples for stop markers
    """
    # Normalize lat/lon to relative coordinates (in km from start)
    lat_start, lon_start = coords[0]
    
    # Approximate conversion (works for small distances)
    x = (coords[:, 1] - lon_start) * 111 * np.cos(np.radians(lat_start))  # km
    y = (coords[:, 0] - lat_start) * 111 * 1.5  # km, scaled up 1.5x for better proportions
    z = elevations / 1000 * 50  # Convert to km and scale up for visibility (50x exaggeration)
    
    # Create figure
    fig = go.Figure()
    
    # Create shaded "curtain" under the route line down to z=0
    # We'll create a mesh surface connecting the route to the ground
    n_points = len(x)
    
    # Create vertices for the mesh (each point on route + its projection on z=0)
    vertices_x = np.concatenate([x, x])  # Top points, then bottom points
    vertices_y = np.concatenate([y, y])
    vertices_z = np.concatenate([z, np.zeros_like(z)])
    
    # Create triangular faces connecting top and bottom
    # Each quad (between points i and i+1) becomes 2 triangles
    i_indices = []
    j_indices = []
    k_indices = []
    
    for i in range(n_points - 1):
        # Triangle 1: top[i], bottom[i], top[i+1]
        i_indices.append(i)
        j_indices.append(i + n_points)
        k_indices.append(i + 1)
        
        # Triangle 2: top[i+1], bottom[i], bottom[i+1]
        i_indices.append(i + 1)
        j_indices.append(i + n_points)
        k_indices.append(i + 1 + n_points)
    
    # Add the shaded curtain mesh
    fig.add_trace(go.Mesh3d(
        x=vertices_x,
        y=vertices_y,
        z=vertices_z,
        i=i_indices,
        j=j_indices,
        k=k_indices,
        color='rgba(150, 150, 150, 0.5)',
        opacity=0.5,
        name='Elevation curtain',
        showlegend=False,
        hoverinfo='skip',
        lighting=dict(ambient=0.8, diffuse=0.8, roughness=0.9, specular=0.1),
        flatshading=False
    ))
    
    # 3D route line (main path) - drawn on top
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color='rgb(0, 0, 0)',
            width=6
        ),
        name='Route',
        hovertemplate='<b>Distance:</b> %{customdata:.1f} km<br>' +
                      '<b>Elevation:</b> %{text:.0f} m<br>' +
                      '<extra></extra>',
        customdata=distances,
        text=elevations
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        marker=dict(size=10, color='rgb(0, 0, 0)', symbol='circle'),
        name='Start',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        marker=dict(size=10, color='rgb(0, 0, 0)', symbol='circle'),
        name='End',
        showlegend=False
    ))
    
    # Add stop markers if provided
    if stops is not None:
        stop_x = []
        stop_y = []
        stop_z = []
        
        for stop_lat, stop_lon in stops:
            # Find closest point on route
            idx = find_closest_point_index(coords, stop_lat, stop_lon)
            stop_x.append(x[idx])
            stop_y.append(y[idx])
            stop_z.append(z[idx])
        
        fig.add_trace(go.Scatter3d(
            x=stop_x, y=stop_y, z=stop_z,
            mode='markers',
            marker=dict(size=10, color='rgb(0, 0, 0)', symbol='circle'),
            name='Stops',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Project the route onto the ground plane (optional reference line)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=np.zeros_like(z),
        mode='lines',
        line=dict(
            color='rgba(100, 100, 100, 0.3)',
            width=2,
            dash='dot'
        ),
        name='Ground projection',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title=''
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='data'  # Use actual data proportions
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        width=1400,
        height=900
    )
    
    return fig


def create_elevation_profile_2d(distances, elevations):
    """
    Create a separate 2D elevation profile plot.
    """
    fig = go.Figure()
    
    # Filled area under the curve
    fig.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(70, 130, 255, 0.3)',
        line=dict(color='rgb(70, 130, 255)', width=3),
        name='Elevation Profile',
        hovertemplate='<b>Distance:</b> %{x:.1f} km<br>' +
                      '<b>Elevation:</b> %{y:.0f} m<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='2D Elevation Profile',
        xaxis_title='Distance along route (km)',
        yaxis_title='Elevation (m)',
        paper_bgcolor='rgb(15, 18, 25)',
        plot_bgcolor='rgb(20, 24, 30)',
        font=dict(color='rgb(200, 210, 220)'),
        xaxis=dict(
            gridcolor='rgb(50, 55, 65)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgb(50, 55, 65)',
            showgrid=True
        ),
        width=1400,
        height=400,
        hovermode='x unified'
    )
    
    return fig


def load_stops(filename):
    """Load stop coordinates from file."""
    stops = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        stops.append((lat, lon))
        return stops
    except FileNotFoundError:
        return None


def main():
    """Main execution function."""
    gpx_filename = '20251110013255-52079-data.gpx'
    
    print(f"📍 Loading GPX file: {gpx_filename}")
    coords, elevations = parse_gpx_file(gpx_filename)
    print(f"   Found {len(coords)} track points")
    
    print("📏 Computing distances...")
    distances = compute_distance_array(coords)
    total_distance = distances[-1]
    print(f"   Total route distance: {total_distance:.2f} km")
    
    print("⛰️  Using elevation data from GPX file...")
    elevation_gain = np.sum(np.maximum(np.diff(elevations), 0))
    print(f"   Elevation range: {elevations.min():.0f}m - {elevations.max():.0f}m")
    print(f"   Total ascent: {elevation_gain:.0f}m")
    
    # Load stops if available
    stops = load_stops('STOPS.md')
    if stops:
        print(f"🛑 Found {len(stops)} stops to mark on route")
    
    print("🎨 Creating 3D visualization...")
    fig_3d = create_visualization(coords, distances, elevations, stops=stops)
    
    print("📊 Creating 2D elevation profile...")
    fig_2d = create_elevation_profile_2d(distances, elevations)
    
    # Save and show
    output_3d = 'gpx_visualization_3d.html'
    output_2d = 'gpx_elevation_profile_2d.html'
    
    print(f"\n💾 Saving visualizations...")
    fig_3d.write_html(output_3d)
    print(f"   3D view: {output_3d}")
    
    fig_2d.write_html(output_2d)
    print(f"   2D profile: {output_2d}")
    
    print("\n✨ Opening in browser...")
    fig_3d.show()
    fig_2d.show()
    
    print("\n✅ Done! The visualizations use real elevation data and are clean line drawings suitable for tracing.")


if __name__ == '__main__':
    main()

