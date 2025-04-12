#!/usr/bin/env bash
#
# fix_google_httpx.sh
#
# Force‑reinstall the incompatible packages to compatible versions
# Copy this file into the "scripts" subdirectory of the Python installation used for ComfyUI
# Then run this file
#

set -e  # exit on any error

echo "Reinstalling google.genai==1.5.0..."
pip install --force-reinstall -v "google.genai==1.5.0"

echo "Reinstalling httpx==0.27.2..."
pip install --force-reinstall -v "httpx==0.27.2"

echo
echo "Installed versions for verification:"
# Use pip show and awk to grab the version field
GG_VER=$(pip show google.genai | awk '/^Version:/{print $2}')
HTTPX_VER=$(pip show httpx     | awk '/^Version:/{print $2}')
echo "google.genai version: $GG_VER"
echo "httpx version:      $HTTPX_VER"

echo
read -p "Done! Press [Enter] to close this window..."
