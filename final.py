import re
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
import io
import json
import os
import sys

def merge_bboxes(bboxes):
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)

def bbox_intersect(b1, b2):
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2
    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

def get_tables_bboxes(doc):
    tables_per_page = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        try:
            tables = page.get_text("dict")
            # Using existing API to find rectangles that might correspond to tables
            # Alternatively, try page.find_tables() if supported in your pymupdf version
            # For no API, we'll fallback to empty for now
            bbox_list = []
            # Either implement a detection or assume empty
        except Exception:
            bbox_list = []
        tables_per_page[page_num] = bbox_list
    return tables_per_page

def bbox_in_any(bbox, bbox_list):
    for b in bbox_list:
        if bbox_intersect(bbox, b):
            return True
    return False

def merge_consecutive_blocks(blocks):
    if not blocks:
        return []
    merged = []
    current = blocks[0].copy()
    for block in blocks[1:]:
        if (block["page"] == current["page"]
            and block["font"] == current["font"]
            and block["size"] == current["size"]
            and block["flags"] == current["flags"]):
            current["text"] += " " + block["text"]
            current["bbox"] = merge_bboxes([current["bbox"], block["bbox"]])
        else:
            merged.append(current)
            current = block.copy()
    merged.append(current)
    return merged

def remove_headers_footers(blocks, margin=0.1):
    headers = set()
    footers = set()
    for b in blocks:
        y0 = b["bbox"][1]
        y1 = b["bbox"][3]
        height = b["page_height"]
        top_margin = height * margin
        bottom_margin = height * (1 - margin)
        if y1 <= top_margin:
            headers.add(id(b))
        elif y0 >= bottom_margin:
            footers.add(id(b))
    return headers, footers

def block_in_tables(page_num, bbox, tables_bbox):
    if page_num in tables_bbox:
        return bbox_in_any(bbox, tables_bbox[page_num])
    return False

def crop_and_ocr(page, bbox, zoom=2):
    rect = fitz.Rect(bbox)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img).strip()
    return text

def parse_pdf(filename):
    doc = fitz.open(filename)
    tables_bbox = get_tables_bboxes(doc)
    all_blocks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_blocks = page.get_text("dict")["blocks"]
        height = page.rect.height
        width = page.rect.width
        page_blocks = []
        for block in raw_blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                merged_spans = []
                current = None
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    font = span["font"]
                    size = span["size"]
                    flags = span["flags"]
                    bbox = span["bbox"]
                    if current is None:
                        current = {
                            "text": text,
                            "font": font,
                            "size": size,
                            "flags": flags,
                            "bboxes": [bbox]
                        }
                    else:
                        if (font == current["font"] and size == current["size"] and flags == current["flags"]):
                            current["text"] += " " + text
                            current["bboxes"].append(bbox)
                        else:
                            merged_bbox = merge_bboxes(current["bboxes"])
                            page_blocks.append({
                                "text": current["text"],
                                "font": current["font"],
                                "size": current["size"],
                                "flags": current["flags"],
                                "bbox": merged_bbox,
                                "page": page_num,
                                "page_height": height,
                                "page_width": width
                            })
                            current = {
                                "text": text,
                                "font": font,
                                "size": size,
                                "flags": flags,
                                "bboxes": [bbox]
                            }
                if current is not None:
                    merged_bbox = merge_bboxes(current["bboxes"])
                    page_blocks.append({
                        "text": current["text"],
                        "font": current["font"],
                        "size": current["size"],
                        "flags": current["flags"],
                        "bbox": merged_bbox,
                        "page": page_num,
                        "page_height": height,
                        "page_width": width
                    })
        page_blocks = merge_consecutive_blocks(sorted(page_blocks, key=lambda b: b['bbox'][1]))
        all_blocks.extend(page_blocks)
    return all_blocks, tables_bbox, doc

def extract_title(blocks, doc):
    candidates = [
        b for b in blocks if b["page"] in [0, 1]
        and len(b["text"]) > 5
        and not any(word in b["text"].lower() for word in ["page", "draft", "confidential"])
    ]
    if not candidates:
        return "", ""
    candidates.sort(key=lambda b: (-b["size"], b["page"], b["bbox"][1]))
    title_block = candidates[0]
    page = doc.load_page(title_block["page"])
    ocr_text = crop_and_ocr(page, title_block["bbox"])
    return title_block["text"], ocr_text

def ends_with_single_dot(text):
    t = text.strip()
    if t.endswith(".") and not t.endswith("..."):
        if len(t) >= 2:
            return t[-2] != "."
        else:
            return True
    return False

def heading_level_number(level_str):
    if level_str.upper().startswith("H"):
        try:
            return int(level_str[1:])
        except:
            return None
    return None

def cluster_font_sizes(candidates):
    sizes = sorted(set(b["size"] for b in candidates), reverse=True)
    return {size: f"H{i+1}" for i, size in enumerate(sizes)}

def extract_headings(blocks, doc, title_text):
    if not blocks:
        return []

    font_sizes = [b["size"] for b in blocks]
    threshold = np.mean(font_sizes) + 1.5 if font_sizes else 0

    candidates = []
    for idx, b in enumerate(blocks):
        txt = b["text"].strip()
        if idx == len(blocks) - 1:
            continue  # skip last block
        if not txt or txt == title_text:
            continue
        if len(txt) < 3 or len(txt) > 150:
            continue
        if ends_with_single_dot(txt):
            continue
        if b["size"] < threshold:
            continue
        candidates.append(b)

    if not candidates:
        return []

    size_to_level = cluster_font_sizes(candidates)

    headings = []
    for c in candidates:
        level = size_to_level.get(c["size"], "H3")
        page = doc.load_page(c["page"])
        ocr_txt = crop_and_ocr(page, c["bbox"])
        headings.append({
            "level": level,
            "text": c["text"],
            "page": c["page"],
            "ocr_text": ocr_txt
        })

    def get_block_y_pos(blk):
        # Find y position for sorting
        return blk["bbox"][1]

    headings.sort(key=lambda h: (h["page"], 
                                 get_block_y_pos(next(c for c in candidates if c["text"] == h["text"] and c["page"] == h["page"]))))
    filtered = []
    last_level = None
    for h in headings:
        lvl = heading_level_number(h["level"])
        if lvl is None:
            filtered.append(h)
            continue
        if last_level is None or lvl <= last_level + 1:
            filtered.append(h)
            last_level = lvl  # update current level
    return filtered

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    blocks, tables, doc = parse_pdf(pdf_path)

    print(f"Extracted {len(blocks)} blocks from PDF")

    title_text, title_ocr = extract_title(blocks, doc)
    print(f"Title Extracted (text): {title_text}")
    print(f"Title Extracted (OCR): {title_ocr}\n")

    header_blocks, footer_blocks = remove_headers_footers(blocks, margin=0.1)
    combined_blocks = header_blocks.union(footer_blocks)

    filtered_blocks = [
        b for b in blocks
        if id(b) not in combined_blocks and not block_in_tables(b["page"], b["bbox"], tables)
    ]

    print(f"Blocks after removing headers, footers and tables: {len(filtered_blocks)}")

    headings = extract_headings(filtered_blocks, doc, title_text)

    print(f"Extracted {len(headings)} headings:")
    for h in headings:
        print(f"{h['level']} | Page {h['page'] +1} | Text: {h['text']}")
        print(f"OCR Text:\n{h['ocr_text']}\n")

    with open("output.txt", "w", encoding='utf-8') as f:
        for b in filtered_blocks:
            f.write(f"Text: {b['text']}\n")
            f.write(f"Font: {b['font']} | Size: {b['size']} | Bold: {bool(b['flags'] & 2)} | Italic: {bool(b['flags'] & 4)}\n")
            f.write(f"Page: {b['page'] + 1}\n")
            f.write(f"BBox: {b['bbox']}\n")
            f.write("---------\n")

    output_json = {
        "title": title_ocr if title_ocr else title_text,
        "outline": [
            {
                "level": h["level"],
                "text": h["ocr_text"] if h["ocr_text"] else h["text"],
                "page": h["page"] + 1,
            }
            for h in headings
        ]
    }

    with open("output.json", "w", encoding="utf-8") as json_file:
        json.dump(output_json, json_file, ensure_ascii=False, indent=4)

    print("Saved output to output.json and output.txt")

if __name__ == "__main__":
    main()
