#!/usr/bin/env bash
# usage: upgrade_dpi.sh <fig_dir> <dpi>
set -e

DIR=$1
DPI=$2

if [ -z "$DIR" ] || [ -z "$DPI" ]; then
    echo "Usage: $0 <fig_dir> <dpi>"
    echo "Example: $0 figures 300"
    exit 1
fi

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' not found"
    exit 1
fi

echo "🔧 Upgrading DPI to ${DPI} for all PNG files in $DIR"

# Find all PNG files recursively
find "$DIR" -name "*.png" | while read -r f; do
    echo "Processing: $f"
    
    # Create backup
    cp "$f" "${f}.backup"
    
    # Upgrade DPI using ImageMagick
    if command -v magick >/dev/null 2>&1; then
        magick "$f" -density "$DPI" -units PixelsPerInch "$f"
    elif command -v convert >/dev/null 2>&1; then
        convert "$f" -density "$DPI" -units PixelsPerInch "$f"
    else
        echo "Error: ImageMagick not found. Install with 'brew install imagemagick'"
        exit 1
    fi
    
    echo "✅ Upgraded: $f"
done

echo "🎯 DPI upgrade complete!"
echo "📊 Verifying DPI:"
find "$DIR" -name "*.png" -exec identify -format "%f: %Q dpi\n" {} \; 