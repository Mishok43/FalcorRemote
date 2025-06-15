#!/usr/bin/env python3
"""
Create a database of images, scene descriptions, and object lists from a folder.
Uses AsyncSceneDescriptionAPI from async_parallel_inference.py
"""

import asyncio
import sqlite3
import hashlib
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

# Import the scene description API
from async_parallel_inference import AsyncSceneDescriptionAPI #, SceneDescriptionResult

@dataclass
class DatabaseEntry:
    """Database entry for image analysis."""
    image_hash: str
    image_path: str
    scene_description: str
    objects_list: str
    processing_time: float
    sample_id: Optional[int]
    timestamp: str
    image_size: Optional[int] = None

class ImageAnalysisDatabase:
    """SQLite database for storing image analysis results."""
    
    def __init__(self, db_path: str = "image_analysis.db"):
        """Initialize the database."""
        self.db_path = db_path
        self._create_tables()
        print(f"Database initialized: {db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main table for image analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT UNIQUE NOT NULL,
                    image_path TEXT NOT NULL,
                    scene_description TEXT,
                    objects_list TEXT,
                    processing_time REAL,
                    sample_id INTEGER,
                    timestamp TEXT NOT NULL,
                    image_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_hash ON image_analysis(image_hash)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_path ON image_analysis(image_path)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sample_id ON image_analysis(sample_id)
            ''')
            
            conn.commit()
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate SHA-256 hash of image file."""
        hash_sha256 = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_image_size(self, image_path: str) -> Optional[int]:
        """Get image file size in bytes."""
        try:
            return os.path.getsize(image_path)
        except:
            return None
    
    def insert_entry(self, entry: DatabaseEntry) -> bool:
        """Insert a new database entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO image_analysis 
                    (image_hash, image_path, scene_description, objects_list, 
                     processing_time, sample_id, timestamp, image_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.image_hash,
                    entry.image_path,
                    entry.scene_description,
                    entry.objects_list,
                    entry.processing_time,
                    entry.sample_id,
                    entry.timestamp,
                    entry.image_size
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error inserting into database: {e}")
            return False
    
    def get_entry_by_hash(self, image_hash: str) -> Optional[DatabaseEntry]:
        """Get entry by image hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_hash, image_path, scene_description, objects_list,
                       processing_time, sample_id, timestamp, image_size
                FROM image_analysis WHERE image_hash = ?
            ''', (image_hash,))
            
            row = cursor.fetchone()
            if row:
                return DatabaseEntry(*row)
            return None
    
    def get_entry_by_path(self, image_path: str) -> Optional[DatabaseEntry]:
        """Get entry by image path."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_hash, image_path, scene_description, objects_list,
                       processing_time, sample_id, timestamp, image_size
                FROM image_analysis WHERE image_path = ?
            ''', (image_path,))
            
            row = cursor.fetchone()
            if row:
                return DatabaseEntry(*row)
            return None
    
    def get_all_entries(self) -> List[DatabaseEntry]:
        """Get all entries from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_hash, image_path, scene_description, objects_list,
                       processing_time, sample_id, timestamp, image_size
                FROM image_analysis ORDER BY created_at DESC
            ''')
            
            return [DatabaseEntry(*row) for row in cursor.fetchall()]
    
    def search_by_description(self, keyword: str, limit: int = 4) -> List[DatabaseEntry]:
        """Search entries by description keyword, return first N matches."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_hash, image_path, scene_description, objects_list,
                       processing_time, sample_id, timestamp, image_size
                FROM image_analysis 
                WHERE scene_description LIKE ? OR objects_list LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (f'%{keyword}%', f'%{keyword}%', limit))
            
            return [DatabaseEntry(*row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) FROM image_analysis')
            total_entries = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute('SELECT AVG(processing_time) FROM image_analysis')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Total image size
            cursor.execute('SELECT SUM(image_size) FROM image_analysis')
            total_size = cursor.fetchone()[0] or 0
            
            return {
                'total_entries': total_entries,
                'avg_processing_time': avg_processing_time,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024) if total_size > 0 else 0
            }

def visualize_images(entries: List[DatabaseEntry]):
    """Simple visualization of images."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    if not entries:
        print("No images to visualize")
        return
    
    print(f"\n🖼️  Visualizing {len(entries)} images:")
    
    # Create subplot grid
    fig, axes = plt.subplots(1, len(entries), figsize=(4*len(entries), 4))
    if len(entries) == 1:
        axes = [axes]
    
    for i, entry in enumerate(entries):
        try:
            # Load and display image
            img = mpimg.imread(entry.image_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{Path(entry.image_path).name}")
            axes[i].axis('off')
            
            # Print details
            print(f"\n{i+1}. {Path(entry.image_path).name}")
            print(f"   Description: {entry.scene_description[:100]}...")
            
        except Exception as e:
            print(f"Error loading image {entry.image_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{Path(entry.image_path).name}", 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{Path(entry.image_path).name}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

class DatabaseCreator:
    """Creates and populates image analysis database."""
    
    def __init__(self, db_path: str = "image_analysis.db"):
        """Initialize the database creator."""
        self.database = ImageAnalysisDatabase(db_path)
        self.api = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.api = AsyncSceneDescriptionAPI()
        await self.api.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.api:
            await self.api.__aexit__(exc_type, exc_val, exc_tb)
    
    def _find_image_files(self, directory_path: str) -> List[str]:
        """Find all image files in directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        image_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    async def process_single_image(self, image_path: str, sample_id: int = None) -> Optional[DatabaseEntry]:
        """Process a single image and store in database."""
        # Check if already processed
        existing_entry = self.database.get_entry_by_path(image_path)
        if existing_entry:
            print(f"✓ Skipping {Path(image_path).name} (already processed)")
            return existing_entry
        
        try:
            print(f"Processing {Path(image_path).name}...")
            start_time = time.time()
            
            # Generate scene description
            result = await self.api.generate_scene_description_async(image_path, sample_id)
            
            processing_time = time.time() - start_time
            
            # Calculate image hash and size
            image_hash = self.database._calculate_image_hash(image_path)
            image_size = self.database._get_image_size(image_path)
            
            # Create database entry
            db_entry = DatabaseEntry(
                image_hash=image_hash,
                image_path=image_path,
                scene_description=result.description,
                objects_list=result.description,  # Same as description for now
                processing_time=processing_time,
                sample_id=sample_id,
                timestamp=datetime.now().isoformat(),
                image_size=image_size
            )
            
            # Insert into database
            if self.database.insert_entry(db_entry):
                print(f"✓ Stored {Path(image_path).name} in database")
                return db_entry
            else:
                print(f"✗ Failed to store {Path(image_path).name}")
                return None
                
        except Exception as e:
            print(f"✗ Error processing {Path(image_path).name}: {e}")
            return None
    
    async def process_directory(self, directory_path: str, max_concurrent: int = 10) -> List[DatabaseEntry]:
        """Process all images in a directory and store in database."""
        image_files = self._find_image_files(directory_path)
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_files)} images in {directory_path}")
        print(f"Processing with max {max_concurrent} concurrent requests...")
        
        # Process images with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_path: str, index: int):
            async with semaphore:
                return await self.process_single_image(image_path, sample_id=index)
        
        tasks = [process_with_semaphore(path, i) for i, path in enumerate(image_files)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        return valid_results
    
    def query_database(self, keyword: str = None) -> List[DatabaseEntry]:
        """Query the database for entries."""
        if keyword:
            return self.database.search_by_description(keyword)
        else:
            return self.database.get_all_entries()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.database.get_statistics()

async def create_database_from_directory(directory_path: str, db_path: str = "image_analysis.db", max_concurrent: int = 10):
    """Create database from directory of images."""
    print(f"Creating image analysis database from: {directory_path}")
    print(f"Database file: {db_path}")
    
    async with DatabaseCreator(db_path) as creator:
        start_time = time.time()
        
        # Process all images
        results = await creator.process_directory(directory_path, max_concurrent)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Show results
        print(f"\n🎉 Database creation completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Images processed: {len(results)}")
        
        # Show statistics
        stats = creator.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        return results

def query_database(db_path: str = "image_analysis.db", keyword: str = None, visualize: bool = False):
    """Query the database from command line."""
    db = ImageAnalysisDatabase(db_path)
    
    if keyword:
        entries = db.search_by_description(keyword, limit=4)
        print(f"Found {len(entries)} entries matching '{keyword}':")
    else:
        entries = db.get_all_entries()[:4]  # Limit to first 4
        print(f"First {len(entries)} entries in database:")
    
    for entry in entries:
        print(f"\n{Path(entry.image_path).name}:")
        print(f"  Description: {entry.scene_description[:100]}...")
        print(f"  Sample ID: {entry.sample_id}")
        print(f"  Processing time: {entry.processing_time:.2f}s")
    
    if visualize and entries:
        visualize_images(entries)

def show_database_stats(db_path: str = "image_analysis.db"):
    """Show database statistics."""
    db = ImageAnalysisDatabase(db_path)
    stats = db.get_statistics()
    
    print(f"📊 Database Statistics for {db_path}:")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image analysis database from directory")
    parser.add_argument("--directory", "-d", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--db_path", type=str, default="image_analysis.db", help="Database file path (default: image_analysis.db)")
    parser.add_argument("--max_concurrent", "-m", type=int, default=10, help="Maximum concurrent requests (default: 10)")
    parser.add_argument("--query", "-q", type=str, help="Query database for keyword")
    parser.add_argument("--visualize", "-v", type=bool, default=False, help="Visualize images when querying")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()

    if args.query:
        # Query database
        query_database(args.directory + "/" + args.db_path, args.query, args.visualize)
    elif args.stats:
        # Show statistics
        show_database_stats(args.directory + "/" + args.db_path)
    else:
        # Create database
        asyncio.run(create_database_from_directory(args.directory, args.directory + "/" +args.db_path, args.max_concurrent))