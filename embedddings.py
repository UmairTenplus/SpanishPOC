import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from docx import Document
import os
import re
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_pdf(file_path, extract_images_with_ocr=True):
    text = ""
    doc = fitz.open(file_path)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text += page.get_text()

        if extract_images_with_ocr:
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))

                # Run OCR on the image
                ocr_text = pytesseract.image_to_string(img_pil, lang="spa")
 
                text += f"\n[{page_index+1} image {img_index+1}]\n{ocr_text}\n"

    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    buffer_line = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue  # skip blank lines

        # Skip junky OCR lines with mostly symbols or single letters
        if len(line) <= 2 and not re.match(r'[A-Za-z0-9]', line):
            continue

        # If line ends in a lowercase letter or number, and next line continues, join
        if buffer_line:
            combined = buffer_line + ' ' + line
            if len(combined) < 200:
                buffer_line = combined
                continue
            else:
                cleaned_lines.append(buffer_line)
                buffer_line = line
        else:
            buffer_line = line

    if buffer_line:
        cleaned_lines.append(buffer_line)

    return "\n".join(cleaned_lines)


def recursive_chunk(text, chunk_size=2000, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# === Example Usage ===
# === Main Processing of All Files ===
dataset_path = "dataset"
output_dir = "chunked_output"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.pdf', '.docx')):
        file_path = os.path.join(dataset_path, filename)
        print(f"\nProcessing file: {filename}")
        try:
            raw_text = extract_text(file_path)
            cleaned_text = clean_text(raw_text)
            chunks = recursive_chunk(cleaned_text)

            print(f" -> Extracted {len(chunks)} chunks.")

            # Save chunks for inspection (optional)
            for i, chunk in enumerate(chunks):
                chunk_filename = os.path.join(output_dir, f"{filename}_chunk_{i+1}.txt")
                with open(chunk_filename, "w", encoding="utf-8") as f:
                    f.write(chunk)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nAll files processed successfully.")
