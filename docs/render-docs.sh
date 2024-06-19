#!/bin/bash

SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd $SCRIPT_DIR

pandoc \
  -s "$SCRIPT_DIR/report.tex" \
  -o "$SCRIPT_DIR/report.html" \
  --bibliography "$SCRIPT_DIR/report.bib" \
  --csl "$SCRIPT_DIR/report.csl" \
  --citeproc \
  --toc \
  --mathjax \
  --number-sections \
  -M document-css=true \
  -M maxwidth=42em \
  --standalone \
  --css "./report.css" \
  --verbose
