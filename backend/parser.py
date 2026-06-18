import fitz  # PyMuPDF
import pdfplumber
import os
from PIL import Image
import pytesseract
# Tell pytesseract where tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── MAIN FUNCTION ───────────────────────────────────────────
# This function takes a PDF file path
# and returns a list of chunks (text, tables, image text)
# each chunk has: text, page number, type

def parse_pdf(file_path: str) -> list:
    chunks = []

    print(f"📄 Starting to parse: {file_path}")

    # ── STEP 1: Extract normal text ──────────────────────────
    chunks += extract_text(file_path)

    # ── STEP 2: Extract tables ───────────────────────────────
    chunks += extract_tables(file_path)

    # ── STEP 3: Extract text from images ─────────────────────
    chunks += extract_image_text(file_path)

    print(f"✅ Total chunks extracted: {len(chunks)}")
    return chunks


# ─── FUNCTION 1: Extract normal text ─────────────────────────
def extract_text(file_path: str) -> list:
    chunks = []
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean the text
            text = text.strip()
            
            # Only save if there is actual text
            if text and len(text) > 20:
                chunks.append({
                    "text": text,
                    "page": page_num + 1,  # page numbers start from 1
                    "type": "text"
                })
                print(f"  ✅ Text extracted from page {page_num + 1}")
        
        doc.close()
        
    except Exception as e:
        print(f"  ❌ Error extracting text: {e}")
    
    return chunks


# ─── FUNCTION 2: Extract tables ──────────────────────────────
def extract_tables(file_path: str) -> list:
    chunks = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                for table_index, table in enumerate(tables):
                    # Convert table to plain text sentences
                    table_text = convert_table_to_text(table)
                    
                    if table_text and len(table_text) > 10:
                        chunks.append({
                            "text": table_text,
                            "page": page_num + 1,
                            "type": "table"
                        })
                        print(f"  ✅ Table {table_index + 1} extracted from page {page_num + 1}")
    
    except Exception as e:
        print(f"  ❌ Error extracting tables: {e}")
    
    return chunks


# ─── FUNCTION 3: Convert table rows to plain text ────────────
def convert_table_to_text(table: list) -> str:
    if not table:
        return ""
    
    text_parts = []
    
    # First row is usually the header
    headers = table[0] if table[0] else []
    headers = [str(h).strip() if h else "" for h in headers]
    
    # Go through each row after header
    for row in table[1:]:
        if not row:
            continue
        
        row_parts = []
        for i, cell in enumerate(row):
            if cell and str(cell).strip():
                # If we have a header for this column
                if i < len(headers) and headers[i]:
                    row_parts.append(f"{headers[i]}: {str(cell).strip()}")
                else:
                    row_parts.append(str(cell).strip())
        
        if row_parts:
            text_parts.append(", ".join(row_parts))
    
    return ". ".join(text_parts)


# ─── FUNCTION 4: Extract text from images using OCR ──────────
import tempfile
def extract_image_text(file_path: str) -> list:
    chunks = []
    
    try:
        import shutil
        import tempfile
        from io import BytesIO
        
        # Copy PDF to temp location first to avoid Windows lock
        temp_pdf = os.path.join(tempfile.gettempdir(), "temp_doc.pdf")
        shutil.copy2(file_path, temp_pdf)
        
        doc = fitz.open(temp_pdf)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Read directly from memory - no temp file
                    img_obj = Image.open(BytesIO(image_bytes))
                    
                    # Convert to RGB to avoid mode errors
                    img_obj = img_obj.convert("RGB")
                    
                    ocr_text = pytesseract.image_to_string(img_obj)
                    ocr_text = ocr_text.strip()
                    
                    if ocr_text and len(ocr_text) > 20:
                        chunks.append({
                            "text": ocr_text,
                            "page": page_num + 1,
                            "type": "image_text"
                        })
                        print(f"  ✅ Text found in image on page {page_num + 1}")
                    else:
                        print(f"  ℹ️ No text found in image on page {page_num + 1}")
                    
                except Exception as e:
                    import traceback
                print(f"  ⚠️ Could not process image on page {page_num + 1}: {e}")
                traceback.print_exc()
        
        doc.close()
        
        # Clean up temp pdf
        os.remove(temp_pdf)
        
    except Exception as e:
        print(f"  ❌ Error extracting image text: {e}")
    
    return chunks
# ─── TEST: Run this file directly to test ────────────────────
if __name__ == "__main__":
    # Change this to any PDF you have for testing
    test_file = "uploads/test.pdf"
    
    if os.path.exists(test_file):
        result = parse_pdf(test_file)
        print("\n📦 CHUNKS EXTRACTED:")
        for i, chunk in enumerate(result):
            print(f"\nChunk {i+1}:")
            print(f"  Page: {chunk['page']}")
            print(f"  Type: {chunk['type']}")
            print(f"  Text: {chunk['text'][:100]}...")
    else:
        print("❌ No test PDF found. Upload a PDF to the uploads folder first.")