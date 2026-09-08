"""
Utilities for reading and writing cryoEM data files.
"""
import numpy as np
import mrcfile
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import re


def load_mrc(mrc_path: str, permissive: bool = True) -> np.ndarray:
    """
    Load an MRC micrograph file.
    
    Args:
        mrc_path: Path to .mrc file
        permissive: If True (default), use mrcfile permissive mode so non-standard
            or legacy MRC headers (e.g. missing Map ID string) are tolerated.
        
    Returns:
        2D numpy array of the micrograph
    """
    with mrcfile.open(mrc_path, permissive=permissive) as mrc:
        data = mrc.data
    
    return data


def load_cs(cs_path: str) -> np.ndarray:
    """
    Load a CryoSPARC .cs particle pick file.
    
    Args:
        cs_path: Path to .cs file
        
    Returns:
        Structured numpy array with particle pick data
    """
    return np.load(cs_path)


def get_cs_field_groups(cs_data: np.ndarray) -> List[str]:
    """
    Return unique top-level CryoSPARC field groups present in a structured .cs array.

    Example:
      'location/micrograph_path' -> 'location'
      'alignments2D/pose' -> 'alignments2D'
    """
    groups = []
    seen = set()
    for name in (cs_data.dtype.names or ()):
        if "/" not in name:
            continue
        group = str(name).split("/", 1)[0]
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def get_picks_for_micrograph(cs_data: np.ndarray, micrograph_path: str) -> np.ndarray:
    """
    Extract particle picks for a specific micrograph.
    
    Args:
        cs_data: Loaded .cs file data
        micrograph_path: Path or filename of the micrograph
        
    Returns:
        Subset of cs_data containing picks for this micrograph
    """
    # Extract just the filename for matching
    micrograph_name = os.path.basename(micrograph_path)
    
    # Match by micrograph path (bytes comparison)
    matches = []
    for pick in cs_data:
        pick_path = pick['location/micrograph_path'].decode('utf-8') if isinstance(pick['location/micrograph_path'], bytes) else pick['location/micrograph_path']
        pick_name = os.path.basename(pick_path)
        if pick_name == micrograph_name:
            matches.append(pick)
    
    if matches:
        return np.array(matches, dtype=cs_data.dtype)
    return np.array([], dtype=cs_data.dtype)


def picks_to_coordinates(picks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert fractional pick coordinates to pixel coordinates.
    
    Args:
        picks: Array of particle picks
        image_shape: (height, width) of the micrograph
        
    Returns:
        Array of (x, y) pixel coordinates
    """
    if len(picks) == 0:
        return np.array([]).reshape(0, 2)
    
    coords = np.zeros((len(picks), 2))
    coords[:, 0] = picks['location/center_x_frac'] * image_shape[1]  # x coordinate
    coords[:, 1] = picks['location/center_y_frac'] * image_shape[0]  # y coordinate
    
    return coords


def create_blob_cs_data(cs_data: np.ndarray) -> np.ndarray:
    """
    Create a blob-only CS data array containing only uid and blob/* fields.
    
    CryoSPARC expects blob files to contain uid + blob/* fields only.
    
    Args:
        cs_data: Structured numpy array with particle pick data (must include blob fields)
        
    Returns:
        New structured numpy array with only uid and blob/* fields
    """
    if not cs_data.dtype.names:
        return cs_data
    
    # Extract blob fields (uid + blob/*)
    blob_fields = []
    if 'uid' in cs_data.dtype.names:
        blob_fields.append(('uid', cs_data.dtype.fields['uid'][0]))
    
    blob_fields.extend([(name, cs_data.dtype.fields[name][0]) 
                       for name in cs_data.dtype.names 
                       if name.startswith('blob/')])
    
    if len(blob_fields) == 0:
        # No blob fields, return empty array with same length
        empty_dtype = np.dtype([('uid', '<u8')])
        return np.empty(len(cs_data), dtype=empty_dtype)
    
    # Create new dtype with only blob fields
    blob_dtype = np.dtype(blob_fields)
    
    # Create new array with only these fields
    blob_data = np.empty(len(cs_data), dtype=blob_dtype)
    for field in blob_dtype.names:
        blob_data[field] = cs_data[field]
    
    return blob_data


def create_passthrough_cs_data(cs_data: np.ndarray) -> np.ndarray:
    """
    Create a passthrough CS data array by removing blob/* fields.
    
    CryoSPARC expects passthrough files to contain location/*, ctf/*, pick_stats/*
    fields but NOT blob/* fields. Blob data should be in a separate file.
    
    Args:
        cs_data: Structured numpy array with particle pick data (may include blob fields)
        
    Returns:
        New structured numpy array with blob/* fields removed, preserving all other fields
    """
    if not cs_data.dtype.names:
        return cs_data
    
    # Filter out blob/* fields, keep all other fields
    passthrough_fields = [(name, cs_data.dtype.fields[name][0]) 
                         for name in cs_data.dtype.names 
                         if not name.startswith('blob/')]
    
    if len(passthrough_fields) == len(cs_data.dtype.names):
        # No blob fields to remove, return as-is
        return cs_data
    
    # Create new dtype with only passthrough fields
    passthrough_dtype = np.dtype(passthrough_fields)
    
    # Create new array with only these fields
    passthrough_data = np.empty(len(cs_data), dtype=passthrough_dtype)
    for field in passthrough_dtype.names:
        passthrough_data[field] = cs_data[field]
    
    return passthrough_data


def save_cs(cs_data: np.ndarray, output_path: str):
    """
    Save particle picks to a .cs file.
    
    Args:
        cs_data: Structured numpy array with particle pick data
        output_path: Path where to save the .cs file (should end with .cs)
    
    Note:
        Uses the method from CryoSPARC: open file handle in binary mode and pass to np.save()
        This ensures the file is saved with the exact .cs extension without .npy being added.
    """
    # Ensure .cs extension
    if not output_path.endswith('.cs'):
        output_path += '.cs'
    
    # Use file handle method (as shown in CryoSPARC examples)
    # This ensures the file is saved with .cs extension, not .npy
    with open(output_path, 'wb') as outfile:
        np.save(outfile, cs_data, allow_pickle=False)


def create_csg_file(cs_file_path: str, original_csg_path: Optional[str] = None, 
                    output_csg_path: Optional[str] = None,
                    blob_file_path: Optional[str] = None,
                    passthrough_file_path: Optional[str] = None,
                    use_combined_format: bool = False,
                    cs_data: Optional[np.ndarray] = None):
    """
    Create or modify a .csg file to match filtered .cs files.
    
    Two formats are supported:
    1. Split format (default): blob group points to blob file, location/ctf/pick_stats point to passthrough file
    2. Combined format (use_combined_format=True): ALL groups point to the same file (preserves all fields)
    
    This function REQUIRES an original_csg_path to preserve the full structure.
    It updates metafile paths and num_items counts while preserving all other metadata.
    
    Args:
        cs_file_path: Path to the .cs file (used if passthrough_file_path not provided)
        original_csg_path: REQUIRED path to original .csg file to use as template
        output_csg_path: Optional path for output .csg file (defaults to cs_file_path with .csg extension)
        blob_file_path: Optional path to blob .cs file (only used in split format)
        passthrough_file_path: Optional path to passthrough .cs file (only used in split format)
        use_combined_format: If True, creates a combined format where all groups point to the same file
    
    Returns:
        Path to the created .csg file
    
    Raises:
        ValueError: If original_csg_path is not provided or doesn't exist
    """
    # Require original .csg file
    if not original_csg_path or not os.path.exists(original_csg_path):
        raise ValueError(
            f"original_csg_path is required and must exist. "
            f"Please provide the original .csg file from CryoSPARC export. "
            f"Provided path: {original_csg_path}"
        )
    
    # Determine which file to use
    if use_combined_format:
        # Combined format: use cs_file_path (single file with all fields)
        file_to_use = cs_file_path
    else:
        # Split format: use passthrough file (or cs_file_path as fallback)
        if passthrough_file_path is None:
            file_to_use = cs_file_path
        else:
            file_to_use = passthrough_file_path
    
    # Get number of particles and check which field groups exist
    if cs_data is None:
        data = np.load(file_to_use)
    else:
        data = cs_data
    num_particles = len(data)
    
    # Determine which field groups actually exist in the file
    existing_field_groups = set()
    if data.dtype.names:
        for field_name in data.dtype.names:
            if field_name.startswith('blob/'):
                existing_field_groups.add('blob')
            elif field_name.startswith('location/'):
                existing_field_groups.add('location')
            elif field_name.startswith('ctf/'):
                existing_field_groups.add('ctf')
            elif field_name.startswith('pick_stats/'):
                existing_field_groups.add('pick_stats')
    
    # Determine output path
    if output_csg_path is None:
        output_csg_path = file_to_use.replace('.cs', '.csg')
    
    output_csg_dir = os.path.dirname(os.path.abspath(output_csg_path))
    
    # Get relative path for the file (relative to output CSG file directory)
    def get_relative_path(file_path: str) -> str:
        file_abs_path = os.path.abspath(file_path)
        try:
            file_rel_path = os.path.relpath(file_abs_path, output_csg_dir)
        except ValueError:
            file_rel_path = file_abs_path
        # Use just filename if in same directory
        if os.path.dirname(file_abs_path) == output_csg_dir:
            return os.path.basename(file_path)
        return file_rel_path
    
    if use_combined_format:
        # Combined format: all groups point to the same file
        file_path_to_use = get_relative_path(file_to_use)
    else:
        # Split format: blob points to blob file, others point to passthrough file
        passthrough_path_to_use = get_relative_path(file_to_use)
        if blob_file_path:
            blob_path_to_use = get_relative_path(blob_file_path)
        else:
            blob_path_to_use = passthrough_path_to_use  # Fallback
    
    # Use YAML parsing to only include groups that have fields in the file
    with open(original_csg_path, 'r') as f:
        csg_data = yaml.safe_load(f)
    
    # Remove groups that don't exist in the file
    if 'results' in csg_data:
        groups_to_remove = [g for g in list(csg_data['results'].keys()) 
                           if g not in existing_field_groups]
        for g in groups_to_remove:
            del csg_data['results'][g]
    
    # Update metafile paths and num_items for remaining groups
    if 'results' in csg_data:
        for group_name, group_data in csg_data['results'].items():
            if use_combined_format:
                # All groups point to the same file
                group_data['metafile'] = '>' + get_relative_path(file_to_use)
            else:
                # Split format: blob -> blob file, others -> passthrough file
                if group_name == 'blob' and blob_file_path:
                    group_data['metafile'] = '>' + get_relative_path(blob_file_path)
                else:
                    group_data['metafile'] = '>' + get_relative_path(passthrough_file_path or file_to_use)
            group_data['num_items'] = num_particles
    
    # Write the modified CSG file
    with open(output_csg_path, 'w') as f:
        yaml.dump(csg_data, f, default_flow_style=False, sort_keys=False)
    
    return output_csg_path


def write_csg_with_group_map(
    *,
    original_csg_path: str,
    output_csg_path: str,
    group_to_file_path: Dict[str, str],
    num_items: Optional[int] = None,
) -> str:
    """
    Write a new .csg file by updating only selected result-group `metafile:` lines.

    This helper preserves the existing file formatting and ordering by editing the
    original text line-by-line instead of re-emitting YAML. It intentionally leaves
    all other content untouched, including comments, spacing, and `num_items`.
    """
    if not original_csg_path or not os.path.exists(original_csg_path):
        raise ValueError(
            f"original_csg_path is required and must exist. Provided path: {original_csg_path}"
        )
    if num_items is not None:
        raise ValueError(
            "write_csg_with_group_map preserves formatting by editing only metafile paths; "
            "num_items updates are intentionally not supported here."
        )

    output_csg_dir = os.path.dirname(os.path.abspath(output_csg_path))

    def _relativize(file_path: str) -> str:
        file_abs_path = os.path.abspath(file_path)
        try:
            file_rel_path = os.path.relpath(file_abs_path, output_csg_dir)
        except ValueError:
            file_rel_path = file_abs_path
        if os.path.dirname(file_abs_path) == output_csg_dir:
            return os.path.basename(file_abs_path)
        return file_rel_path

    group_to_relpath = {
        str(group_name): ">" + _relativize(file_path)
        for group_name, file_path in group_to_file_path.items()
    }

    lines = Path(original_csg_path).read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines: List[str] = []
    in_results = False
    current_group: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped == "results:":
            in_results = True
            current_group = None
            out_lines.append(line)
            continue

        if in_results and indent <= 0 and stripped:
            in_results = False
            current_group = None

        if in_results:
            if indent == 2 and stripped.endswith(":") and not stripped.startswith("metafile:"):
                current_group = stripped[:-1]
                out_lines.append(line)
                continue

            if (
                current_group is not None
                and current_group in group_to_relpath
                and indent >= 4
                and stripped.startswith("metafile:")
            ):
                prefix = line[: line.index("metafile:") + len("metafile:")]
                existing_value = line[line.index("metafile:") + len("metafile:") :].rstrip("\n")
                value_stripped = existing_value.strip()
                new_value = group_to_relpath[current_group]
                if value_stripped.startswith("'") and value_stripped.endswith("'"):
                    new_value = f"'{new_value}'"
                elif value_stripped.startswith('\"') and value_stripped.endswith('\"'):
                    new_value = f'\"{new_value}\"'
                newline = "\n" if line.endswith("\n") else ""
                out_lines.append(f"{prefix} {new_value}{newline}")
                continue

        out_lines.append(line)

    Path(output_csg_path).write_text("".join(out_lines), encoding="utf-8")
    return output_csg_path


def filter_picks_by_polygons(picks: np.ndarray, image_shape: Tuple[int, int], 
                             bad_polygons: List[np.ndarray]) -> np.ndarray:
    """
    Filter out particle picks that fall within any of the bad region polygons.
    
    Args:
        picks: Array of particle picks
        image_shape: (height, width) of the micrograph
        bad_polygons: List of polygons, each as Nx2 array of (x, y) coordinates
        
    Returns:
        Filtered array of picks (picks not in bad regions)
    """
    from matplotlib.path import Path as MPLPath
    
    if len(picks) == 0 or len(bad_polygons) == 0:
        return picks
    
    coords = picks_to_coordinates(picks, image_shape)
    
    # Create matplotlib paths for each polygon
    polygon_paths = [MPLPath(poly) for poly in bad_polygons]
    
    # Check which picks are inside any bad polygon
    keep_mask = np.ones(len(picks), dtype=bool)
    for path in polygon_paths:
        inside = path.contains_points(coords)
        keep_mask &= ~inside
    
    return picks[keep_mask]
