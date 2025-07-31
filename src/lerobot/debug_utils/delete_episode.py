#!/usr/bin/env python

"""
Utility script to delete episodes from a LeRobot dataset.

This script removes:
- Episode parquet data file
- All episode video files (for each camera)
- Episode metadata from episodes.jsonl and episodes_stats.jsonl

Usage:
    python -m lerobot.debug_utils.delete_episode --repo_id adungus/PP-v2 --episode_index 30
    python -m lerobot.debug_utils.delete_episode --repo_id adungus/PP-v2 --episode_index 30 --root /path/to/dataset
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


def get_episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    """Calculate which chunk an episode belongs to."""
    return episode_index // chunk_size


def find_episode_files(dataset_root: Path, episode_index: int) -> dict:
    """Find all files associated with an episode."""
    files = {
        "parquet": [],
        "videos": [],
        "missing": []
    }
    
    episode_str = f"episode_{episode_index:06d}"
    chunk = get_episode_chunk(episode_index)
    chunk_str = f"chunk-{chunk:03d}"
    
    # Find parquet file
    parquet_file = dataset_root / "data" / chunk_str / f"{episode_str}.parquet"
    if parquet_file.exists():
        files["parquet"].append(parquet_file)
    else:
        files["missing"].append(f"parquet: {parquet_file}")
    
    # Find video files
    videos_dir = dataset_root / "videos" / chunk_str
    if videos_dir.exists():
        for camera_dir in videos_dir.iterdir():
            if camera_dir.is_dir():
                video_file = camera_dir / f"{episode_str}.mp4"
                if video_file.exists():
                    files["videos"].append(video_file)
                else:
                    files["missing"].append(f"video: {video_file}")
    else:
        files["missing"].append(f"videos directory: {videos_dir}")
    
    return files


def remove_episode_from_jsonl(jsonl_path: Path, episode_index: int) -> bool:
    """Remove episode entry from a JSONL file."""
    if not jsonl_path.exists():
        logging.warning(f"JSONL file not found: {jsonl_path}")
        return False
    
    lines = []
    found = False
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get("episode_index") == episode_index:
                        found = True
                        logging.info(f"Found episode {episode_index} in {jsonl_path.name}")
                        continue  # Skip this line (delete it)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON line in {jsonl_path}: {line}")
                lines.append(line)
    
    if found:
        # Write back the file without the deleted episode
        with open(jsonl_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        logging.info(f"Removed episode {episode_index} from {jsonl_path.name}")
    else:
        logging.warning(f"Episode {episode_index} not found in {jsonl_path.name}")
    
    return found


def delete_episode(repo_id: str, episode_index: int, root: Path = None, dry_run: bool = False) -> bool:
    """
    Delete an episode from a LeRobot dataset.
    
    Args:
        repo_id: Repository ID (e.g., "adungus/PP-v2")
        episode_index: Episode index to delete
        root: Root directory for dataset (defaults to HF_LEROBOT_HOME/repo_id)
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        bool: True if episode was successfully deleted, False otherwise
    """
    if root is None:
        dataset_root = HF_LEROBOT_HOME / repo_id
    else:
        dataset_root = Path(root)
    
    if not dataset_root.exists():
        logging.error(f"Dataset directory not found: {dataset_root}")
        return False
    
    logging.info(f"Processing dataset: {repo_id}")
    logging.info(f"Dataset root: {dataset_root}")
    logging.info(f"Episode to delete: {episode_index}")
    
    # Find all files for this episode
    files = find_episode_files(dataset_root, episode_index)
    
    if files["missing"]:
        logging.warning("Some expected files are missing:")
        for missing in files["missing"]:
            logging.warning(f"  - {missing}")
    
    total_files = len(files["parquet"]) + len(files["videos"])
    if total_files == 0:
        logging.error(f"No files found for episode {episode_index}")
        return False
    
    if dry_run:
        logging.info("DRY RUN - Would delete the following files:")
        for file_path in files["parquet"] + files["videos"]:
            logging.info(f"  - {file_path}")
        
        # Check metadata files
        episodes_file = dataset_root / "meta" / "episodes.jsonl"
        episodes_stats_file = dataset_root / "meta" / "episodes_stats.jsonl"
        logging.info("Would update metadata files:")
        logging.info(f"  - {episodes_file}")
        logging.info(f"  - {episodes_stats_file}")
        return True
    
    # Delete parquet files
    deleted_count = 0
    for file_path in files["parquet"]:
        try:
            file_path.unlink()
            logging.info(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")
    
    # Delete video files
    for file_path in files["videos"]:
        try:
            file_path.unlink()
            logging.info(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")
    
    # Update metadata files
    episodes_file = dataset_root / "meta" / "episodes.jsonl"
    episodes_stats_file = dataset_root / "meta" / "episodes_stats.jsonl"
    
    metadata_updated = 0
    if remove_episode_from_jsonl(episodes_file, episode_index):
        metadata_updated += 1
    if remove_episode_from_jsonl(episodes_stats_file, episode_index):
        metadata_updated += 1
    
    success = deleted_count > 0 and metadata_updated > 0
    if success:
        logging.info(f"Successfully deleted episode {episode_index} from {repo_id}")
        logging.info(f"Deleted {deleted_count} files and updated {metadata_updated} metadata files")
    else:
        logging.error(f"Failed to completely delete episode {episode_index}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Delete an episode from a LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lerobot.debug_utils.delete_episode --repo_id adungus/PP-v2 --episode_index 30
  python -m lerobot.debug_utils.delete_episode --repo_id adungus/PP-v2 --episode_index 30 --dry-run
  python -m lerobot.debug_utils.delete_episode --repo_id adungus/PP-v2 --episode_index 30 --root /path/to/dataset
        """
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (e.g., 'adungus/PP-v2')"
    )
    
    parser.add_argument(
        "--episode_index",
        type=int,
        required=True,
        help="Episode index to delete"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory for dataset (defaults to ~/.cache/huggingface/lerobot/repo_id)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Enable info level for our messages even if not verbose
    logging.getLogger().setLevel(logging.INFO)
    
    root_path = Path(args.root) if args.root else None
    
    success = delete_episode(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        root=root_path,
        dry_run=args.dry_run
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 