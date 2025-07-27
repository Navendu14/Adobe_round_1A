import fitz  # PyMuPDF
import os
from collections import defaultdict

def merge_bboxes(bboxes):
    """Return bounding box covering all input boxes."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)

def merge_consecutive_blocks(blocks):
    """
    Merge consecutive blocks on the same page if they have same font, size, flags.
    Blocks are expected to be ordered by page and vertical position.
    """
    if not blocks:
        return []

    merged = []
    current = blocks[0].copy()

    for block in blocks[1:]:
        if (block["page"] == current["page"] and
            block["font"] == current["font"] and
            block["size"] == current["size"] and
            block["flags"] == current["flags"]):
            # Merge text with space
            current["text"] += " " + block["text"]
            # Merge bbox to cover both blocks
            current["bbox"] = merge_bboxes([current["bbox"], block["bbox"]])
        else:
            merged.append(current)
            current = block.copy()
    merged.append(current)
    return merged

def parse_pdf_enhanced(pdf_filename):
    """
    Parses PDF and extracts merged text blocks with style info,
    merges consecutive spans and consecutive lines sharing font/size/style on same page.
    Stores minimal necessary features as requested.
    Also writes block details to output.txt.
    """
    if not os.path.isfile(pdf_filename):
        raise FileNotFoundError(f"File '{pdf_filename}' does not exist.")

    document = fitz.open(pdf_filename)
    all_blocks = []

    # Collect text to count repeated (header/footer) - optional for filtering; can omit if desired
    text_occurrences = defaultdict(int)

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        width, height = page.rect.width, page.rect.height
        blocks = page.get_text("dict")["blocks"]

        # Temporary collection of merged spans/lines on this page
        page_blocks = []

        for block in blocks:
            if block['type'] != 0:  # Skip non-text blocks
                continue
            for line in block["lines"]:
                # Step 1: Merge consecutive spans in line by font/size/flags
                merged_spans = []
                current_span = None

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    font = span["font"]
                    size = span["size"]
                    flags = span["flags"]
                    bbox = span["bbox"]

                    if current_span is None:
                        current_span = {
                            "text": text,
                            "font": font,
                            "size": size,
                            "flags": flags,
                            "bboxes": [bbox]
                        }
                    else:
                        if (font == current_span["font"] and size == current_span["size"] and flags == current_span["flags"]):
                            # Merge span texts and bboxes
                            current_span["text"] += " " + text
                            current_span["bboxes"].append(bbox)
                        else:
                            merged_bbox = merge_bboxes(current_span["bboxes"])
                            merged_spans.append({
                                "text": current_span["text"],
                                "font": current_span["font"],
                                "size": current_span["size"],
                                "flags": current_span["flags"],
                                "bbox": merged_bbox
                            })
                            current_span = {
                                "text": text,
                                "font": font,
                                "size": size,
                                "flags": flags,
                                "bboxes": [bbox]
                            }
                # Add last span of the line
                if current_span is not None:
                    merged_bbox = merge_bboxes(current_span["bboxes"])
                    merged_spans.append({
                        "text": current_span["text"],
                        "font": current_span["font"],
                        "size": current_span["size"],
                        "flags": current_span["flags"],
                        "bbox": merged_bbox
                    })

                # Add all merged spans from this line to page_blocks
                page_blocks.extend(merged_spans)

        # Sort page_blocks by y-coordinate (top to bottom) so merging lines is consistent
        page_blocks.sort(key=lambda b: b["bbox"][1])

        # Step 2: Merge consecutive blocks (which are consecutive lines here) with same font/size/flags/page
        # Here all blocks belong to same page: page_num
        for blk in page_blocks:
            blk["page"] = page_num
            text_occurrences[blk["text"]] += 1

        merged_page_blocks = merge_consecutive_blocks(page_blocks)

        all_blocks.extend(merged_page_blocks)

    # Output.txt writing
    with open("output.txt", "w", encoding="utf-8") as f:
        for block in all_blocks:
            is_bold = bool(block["flags"] & 2)
            is_italic = bool(block["flags"] & 4)
            f.write(
                f"Text: '{block['text']}'\n"
                f"Font: {block['font']} | Size: {block['size']} | Bold: {is_bold} | Italic: {is_italic}\n"
                f"Coords: {block['bbox']}\n"
                f"Page: {block['page'] + 1}\n"
                "-----------------------\n"
            )

    return all_blocks

def extract_title(blocks):
    """
    Extract the likely title from blocks.
    Conservative approach: pick block with largest font on page 0 or 1.
    """
    candidates = [b for b in blocks if b["page"] in [0, 1] and len(b["text"]) > 5]

    if not candidates:
        return ""

    candidates.sort(key=lambda b: (-b["size"], b["page"], b["bbox"][1]))

    return candidates[0]["text"].strip()

if __name__ == "__main__":
    pdf_file="file05.pdf"

    try:
        blocks = parse_pdf_enhanced(pdf_file)
        print(f"Total merged text blocks: {len(blocks)}")
        for i, b in enumerate(blocks[:10]):
            print(f"Block {i+1}: Text: '{b['text'][:60]}...', Font: {b['font']}, Size: {b['size']}, Page: {b['page'] + 1}")
        title = extract_title(blocks)
        print("\nDetected Title:", title if title else "[No Title Found]")

        print("\nAll merged blocks saved to 'output.txt'.")

    except FileNotFoundError as e:
        print(str(e))
