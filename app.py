"""
Flask application for geospatial map visualization.

This application provides a web interface for visualizing arbitrary
geospatial datasets with interactive controls for styling and filtering.
"""

import io
import os
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_file,
)
import geopandas as gpd
import matplotlib.pyplot as plt

from datasource import (
    DatasetConfig,
    FileDataSource,
    load_dataset,
)
from spatial_ops import (
    AggregationMethod,
    SpatialJoiner,
    spatial_join,
)
from visualizer import (
    ColorScale,
    MapStyle,
    MapVisualizer,
)
from config import (
    registry,
    get_dataset_config,
    register_dataset,
    list_datasets,
    WALKABILITY_INDEX,
    ZCTA_2020,
)


app = Flask(__name__)

# Cache for loaded datasets
_dataset_cache: Dict[str, gpd.GeoDataFrame] = {}


def get_available_colormaps() -> List[str]:
    """Get list of available colormaps."""
    return [cs.value for cs in ColorScale]


def get_available_aggregations() -> List[str]:
    """Get list of available aggregation methods."""
    return [am.value for am in AggregationMethod]


def load_cached_dataset(config: DatasetConfig) -> gpd.GeoDataFrame:
    """Load a dataset with caching."""
    cache_key = f"{config.path}:{config.layer}"
    if cache_key not in _dataset_cache:
        source = FileDataSource(config)
        _dataset_cache[cache_key] = source.load()
    return _dataset_cache[cache_key]


def clear_cache():
    """Clear the dataset cache."""
    _dataset_cache.clear()


@app.route("/")
def index():
    """Render the main page."""
    datasets = list_datasets()
    colormaps = get_available_colormaps()
    aggregations = get_available_aggregations()

    return render_template(
        "index.html",
        datasets=datasets,
        colormaps=colormaps,
        aggregations=aggregations,
    )


@app.route("/api/datasets")
def api_list_datasets():
    """API endpoint to list available datasets."""
    return jsonify(list_datasets())


@app.route("/api/datasets/<name>/columns")
def api_dataset_columns(name: str):
    """API endpoint to get columns for a dataset."""
    try:
        config = get_dataset_config(name)
        source = FileDataSource(config)
        columns = source.get_columns()
        return jsonify({
            "name": name,
            "columns": columns,
            "id_column": config.id_column,
            "value_column": config.value_column,
        })
    except KeyError:
        return jsonify({"error": f"Dataset '{name}' not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/register", methods=["POST"])
def api_register_dataset():
    """API endpoint to register a new dataset."""
    data = request.json

    required = ["name", "path", "id_column"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    config = DatasetConfig(
        path=data["path"],
        id_column=data["id_column"],
        value_column=data.get("value_column"),
        layer=data.get("layer"),
        name=data.get("display_name", data["name"]),
    )

    register_dataset(data["name"], config)
    return jsonify({"success": True, "name": data["name"]})


@app.route("/api/visualize", methods=["POST"])
def api_visualize():
    """API endpoint to generate a visualization."""
    data = request.json

    try:
        # Get visualization parameters
        dataset_name = data.get("dataset")
        value_column = data.get("value_column")
        colormap = data.get("colormap", "viridis")
        title = data.get("title", "")
        figsize_w = data.get("figsize_w", 12)
        figsize_h = data.get("figsize_h", 8)
        dpi = data.get("dpi", 100)
        format = data.get("format", "png")

        # Load dataset
        if dataset_name:
            config = get_dataset_config(dataset_name)
            gdf = load_cached_dataset(config)
            if not value_column:
                value_column = config.value_column
        else:
            return jsonify({"error": "No dataset specified"}), 400

        # Create visualization
        style = MapStyle(
            colormap=colormap,
            title=title or None,
            legend_label=value_column,
            figsize=(figsize_w, figsize_h),
        )

        viz = MapVisualizer(style)
        fig, ax = viz.plot(gdf, value_column, style)

        # Generate image
        image_data = viz.to_data_uri(fig, format=format, dpi=dpi)

        # Clean up
        plt.close(fig)

        return jsonify({
            "success": True,
            "image": image_data,
            "format": format,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualize/spatial-join", methods=["POST"])
def api_visualize_spatial_join():
    """API endpoint to visualize a spatial join result."""
    data = request.json

    try:
        # Get parameters
        boundary_dataset = data.get("boundary_dataset")
        value_dataset = data.get("value_dataset")
        filter_ids = data.get("filter_ids", [])
        aggregation = data.get("aggregation", "mean")
        colormap = data.get("colormap", "viridis")
        title = data.get("title", "")
        figsize_w = data.get("figsize_w", 12)
        figsize_h = data.get("figsize_h", 8)
        dpi = data.get("dpi", 100)
        format = data.get("format", "png")

        # Load datasets
        boundary_config = get_dataset_config(boundary_dataset)
        value_config = get_dataset_config(value_dataset)

        boundaries = load_cached_dataset(boundary_config)
        values = load_cached_dataset(value_config)

        # Perform spatial join
        result = spatial_join(
            boundaries=boundaries,
            boundary_id_column=boundary_config.id_column,
            source=values,
            source_id_column=value_config.id_column,
            value_column=value_config.value_column,
            filter_boundaries=filter_ids if filter_ids else None,
            aggregation=aggregation,
        )

        # Create visualization
        style = MapStyle(
            colormap=colormap,
            title=title or None,
            legend_label=result.value_column,
            figsize=(figsize_w, figsize_h),
        )

        viz = MapVisualizer(style)
        fig, ax = viz.plot(result.data, result.value_column, style)

        # Generate image
        image_data = viz.to_data_uri(fig, format=format, dpi=dpi)

        # Clean up
        plt.close(fig)

        return jsonify({
            "success": True,
            "image": image_data,
            "format": format,
            "num_features": len(result.data),
            "value_column": result.value_column,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualize/image.<format>", methods=["POST"])
def api_visualize_image(format: str):
    """API endpoint to generate an image file directly."""
    data = request.json

    try:
        dataset_name = data.get("dataset")
        value_column = data.get("value_column")
        colormap = data.get("colormap", "viridis")
        title = data.get("title", "")
        figsize_w = data.get("figsize_w", 12)
        figsize_h = data.get("figsize_h", 8)
        dpi = data.get("dpi", 150)

        # Load dataset
        config = get_dataset_config(dataset_name)
        gdf = load_cached_dataset(config)
        if not value_column:
            value_column = config.value_column

        # Create visualization
        style = MapStyle(
            colormap=colormap,
            title=title or None,
            legend_label=value_column,
            figsize=(figsize_w, figsize_h),
        )

        viz = MapVisualizer(style)
        fig, ax = viz.plot(gdf, value_column, style)

        # Generate image bytes
        image_bytes = viz.to_bytes(fig, format=format, dpi=dpi)
        plt.close(fig)

        mime_types = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "jpg": "image/jpeg",
        }

        return Response(
            image_bytes,
            mimetype=mime_types.get(format, f"image/{format}"),
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/examples")
def examples_page():
    """Render the examples page."""
    return render_template("examples.html")


@app.route("/api/examples/<example_name>")
def api_run_example(example_name: str):
    """API endpoint to run a specific example and return the visualization."""
    examples = {
        "walkability_basic": run_walkability_basic,
        "walkability_styled": run_walkability_styled,
        "spatial_join_mean": run_spatial_join_mean,
        "spatial_join_sum": run_spatial_join_sum,
        "comparison": run_comparison,
    }

    if example_name not in examples:
        return jsonify({
            "error": f"Unknown example: {example_name}",
            "available": list(examples.keys()),
        }), 404

    try:
        result = examples[example_name]()
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({
            "error": "Dataset files not found. Please ensure data files are in the ./data directory.",
            "details": str(e),
        }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_walkability_basic() -> Dict[str, Any]:
    """Run basic walkability visualization example."""
    config = WALKABILITY_INDEX
    gdf = load_cached_dataset(config)

    # Take a sample for faster rendering
    sample = gdf.head(1000)

    style = MapStyle(
        colormap=ColorScale.VIRIDIS,
        title="National Walkability Index (Sample)",
        legend_label="Walkability Score",
        figsize=(12, 8),
    )

    viz = MapVisualizer(style)
    fig, ax = viz.plot(sample, config.value_column, style)
    image = viz.to_data_uri(fig, dpi=100)
    plt.close(fig)

    return {
        "success": True,
        "image": image,
        "title": "Basic Walkability Visualization",
        "description": "Census block-level walkability scores using the viridis colormap.",
        "num_features": len(sample),
    }


def run_walkability_styled() -> Dict[str, Any]:
    """Run styled walkability visualization example."""
    config = WALKABILITY_INDEX
    gdf = load_cached_dataset(config)

    sample = gdf.head(1000)

    style = MapStyle(
        colormap=ColorScale.YELLOW_GREEN_BLUE,
        title="Walkability Index - Custom Style",
        legend_label="Walkability Score",
        edge_color="white",
        edge_width=0.2,
        alpha=0.9,
        figsize=(12, 8),
    )

    viz = MapVisualizer(style)
    fig, ax = viz.plot(sample, config.value_column, style)
    image = viz.to_data_uri(fig, dpi=100)
    plt.close(fig)

    return {
        "success": True,
        "image": image,
        "title": "Styled Walkability Visualization",
        "description": "Custom styling with YlGnBu colormap, white edges, and transparency.",
        "num_features": len(sample),
    }


def run_spatial_join_mean() -> Dict[str, Any]:
    """Run spatial join with mean aggregation example."""
    boundary_config = ZCTA_2020
    value_config = WALKABILITY_INDEX

    boundaries = load_cached_dataset(boundary_config)
    values = load_cached_dataset(value_config)

    # Filter to a specific area for demo (first 50 ZIP codes)
    sample_zips = boundaries[boundary_config.id_column].head(50).tolist()

    result = spatial_join(
        boundaries=boundaries,
        boundary_id_column=boundary_config.id_column,
        source=values,
        source_id_column=value_config.id_column,
        value_column=value_config.value_column,
        filter_boundaries=sample_zips,
        aggregation="mean",
    )

    style = MapStyle(
        colormap=ColorScale.SPECTRAL,
        title="Mean Walkability by ZIP Code",
        legend_label="Avg Walkability",
        figsize=(12, 8),
    )

    viz = MapVisualizer(style)
    fig, ax = viz.plot(result.data, result.value_column, style)
    image = viz.to_data_uri(fig, dpi=100)
    plt.close(fig)

    return {
        "success": True,
        "image": image,
        "title": "Spatial Join - Mean Aggregation",
        "description": f"Census block walkability aggregated to {len(result.data)} ZIP codes using mean.",
        "num_features": len(result.data),
        "value_column": result.value_column,
    }


def run_spatial_join_sum() -> Dict[str, Any]:
    """Run spatial join with sum aggregation example."""
    boundary_config = ZCTA_2020
    value_config = WALKABILITY_INDEX

    boundaries = load_cached_dataset(boundary_config)
    values = load_cached_dataset(value_config)

    sample_zips = boundaries[boundary_config.id_column].head(50).tolist()

    result = spatial_join(
        boundaries=boundaries,
        boundary_id_column=boundary_config.id_column,
        source=values,
        source_id_column=value_config.id_column,
        value_column=value_config.value_column,
        filter_boundaries=sample_zips,
        aggregation="sum",
    )

    style = MapStyle(
        colormap=ColorScale.REDS,
        title="Sum of Walkability Scores by ZIP Code",
        legend_label="Total Walkability",
        figsize=(12, 8),
    )

    viz = MapVisualizer(style)
    fig, ax = viz.plot(result.data, result.value_column, style)
    image = viz.to_data_uri(fig, dpi=100)
    plt.close(fig)

    return {
        "success": True,
        "image": image,
        "title": "Spatial Join - Sum Aggregation",
        "description": f"Census block walkability aggregated to {len(result.data)} ZIP codes using sum.",
        "num_features": len(result.data),
        "value_column": result.value_column,
    }


def run_comparison() -> Dict[str, Any]:
    """Run comparison visualization example."""
    config = WALKABILITY_INDEX
    gdf = load_cached_dataset(config)

    sample = gdf.head(500)

    # Create comparison with different colormaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colormaps = [
        (ColorScale.VIRIDIS, "Viridis"),
        (ColorScale.PLASMA, "Plasma"),
        (ColorScale.YELLOW_GREEN_BLUE, "YlGnBu"),
        (ColorScale.SPECTRAL, "Spectral"),
    ]

    for ax, (cmap, name) in zip(axes.flatten(), colormaps):
        style = MapStyle(
            colormap=cmap,
            title=name,
            legend=True,
            legend_label="Score",
        )
        sample.plot(
            column=config.value_column,
            cmap=cmap.value,
            ax=ax,
            legend=True,
            legend_kwds={"label": "Score", "shrink": 0.5},
        )
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_axis_off()

    plt.suptitle("Colormap Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    viz = MapVisualizer()
    image = viz.to_data_uri(fig, dpi=100)
    plt.close(fig)

    return {
        "success": True,
        "image": image,
        "title": "Colormap Comparison",
        "description": "Same data visualized with different colormaps for comparison.",
        "num_features": len(sample),
    }


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    """Handle dataset upload."""
    if request.method == "GET":
        return render_template("upload.html")

    # Handle file upload
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save file
    upload_dir = os.path.join(app.root_path, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    # Try to load and get columns
    try:
        gdf = gpd.read_file(filepath)
        columns = list(gdf.columns)

        return jsonify({
            "success": True,
            "filename": file.filename,
            "filepath": filepath,
            "columns": columns,
            "num_features": len(gdf),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
