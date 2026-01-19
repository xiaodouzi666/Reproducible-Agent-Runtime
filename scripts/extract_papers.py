import os
import subprocess
import re

PAPERS_DIR = "Papers"
CORPUS_DIR = "demo_data/corpus"

def extract_text_from_pdf(pdf_path):
    """Rxruns pdftotext to extract text from a PDF file."""
    try:
        # -layout maintains physical layout, which might help with columns
        result = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def clean_text(text):
    """Basic cleanup of text."""
    # Remove excessive newlines while preserving paragraph breaks (double newlines)
    # This is tricky with -layout. Let's try to normalize.
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

def find_section(text, section_names):
    """Finds a section by fuzzy matching headers."""
    text_lower = text.lower()
    start_idx = -1
    found_section_name = ""
    
    for name in section_names:
        # Simple search for now. Improve if needed (e.g., regex for standalone headers)
        idx = text_lower.find(name.lower())
        if idx != -1:
            if start_idx == -1 or idx < start_idx:
                start_idx = idx
                found_section_name = name
    
    if start_idx == -1:
        return None, None

    # Find the next likely section header to determine end
    # Common next sections after Abstract: Introduction
    # Common next sections after Methods: Results, Discussion, Conclusion, References
    potential_next_sections = ["introduction", "result", "discussion", "conclusion", "reference", "acknowledgment"]
    
    end_idx = len(text)
    
    # Look for the earliest occurrence of a next section AFTER the start index
    search_start_pos = start_idx + len(found_section_name)
    
    # Heuristic: limit search for end to some reasonable length if we want specific sections? 
    # But Methods can be long.
    
    for next_name in potential_next_sections:
        idx = text_lower.find(next_name, search_start_pos)
        if idx != -1:
             # Ensure it's not immediately adjacent (false positive)
             if idx < end_idx:
                 end_idx = idx
                 
    return found_section_name, text[start_idx:end_idx]

def chunk_text(text, prefix):
    """Chunks text into paragraphs and assigns IDs."""
    # Split by double newlines or other paragraph markers
    # normalizing spaces
    paras = re.split(r'\n\s*\n', text)
    
    chunks = []
    pid = 1
    for p in paras:
        p = p.strip()
        # Filter out short garbage lines (page numbers, headers etc)
        if len(p) < 30: 
            continue
            
        chunks.append(f"[{prefix}_{pid:02d}] {p}")
        pid += 1
    return chunks

def process_file(filename):
    if not filename.endswith(".pdf"):
        return
    
    print(f"Processing {filename}...")
    pdf_path = os.path.join(PAPERS_DIR, filename)
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return

    # Simply extracting everything for now to see structure or using the logic
    # Better logic:
    # 1. Whole text clean up
    # 2. Extract Abstract
    # 3. Extract Methods
    
    # Naive text cleaning for regex matching
    # pdftotext -layout preserves layout, but regex works better on flow text.
    # We might lose structure. Let's try simple string matching first.
    
    abstract_variants = ["Abstract", "ABSTRACT"]
    methods_variants = ["Methodology", "Methods", "Experimental", "Computational Details", "EXPERIMENTAL SECTION"]
    
    _, abstract_text = find_section(text, abstract_variants)
    _, methods_text = find_section(text, methods_variants)
    
    output_lines = []
    
    file_id = filename.split('.')[0].replace('-', '_') # Simple valid ID identifier
    
    if abstract_text:
        # clean content
        content = clean_text(abstract_text)
        # remove the title itself if included
        for v in abstract_variants:
            if content.lower().startswith(v.lower()):
                content = content[len(v):].strip()
        
        chunks = chunk_text(content, f"{file_id}_abs")
        output_lines.append(f"## Abstract\n")
        output_lines.extend(chunks)
        output_lines.append("\n")

    if methods_text:
        content = clean_text(methods_text)
        for v in methods_variants:
             if content.lower().startswith(v.lower()):
                content = content[len(v):].strip()
        
        chunks = chunk_text(content, f"{file_id}_meth")
        output_lines.append(f"## Methods\n")
        output_lines.extend(chunks)
        output_lines.append("\n")
        
    if not output_lines:
        print(f"Warning: No sections found for {filename}")
        # Fallback: dump first 2000 chars as 'Abstract' candidates? No, user wants specific sections.
        # Let's verify manual check data later.
    
    output_file = os.path.join(CORPUS_DIR, filename.replace(".pdf", ".md"))
    with open(output_file, 'w') as f:
        f.write("\n\n".join(output_lines))

def main():
    if not os.path.exists(PAPERS_DIR):
        print(f"Directory {PAPERS_DIR} not found.")
        return

    for f in os.listdir(PAPERS_DIR):
        process_file(f)

if __name__ == "__main__":
    main()
