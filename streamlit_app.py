import streamlit as st
import pandas as pd
import io
import tempfile
import os
from typing import List, Dict, Optional, Tuple
import re

# Import PDF processing libraries
try:
    import pymupdf as fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

def check_dependencies():
    """Check which PDF processing libraries are available"""
    available = []
    missing = []
    
    if PYMUPDF_AVAILABLE:
        available.append("PyMuPDF")
    else:
        missing.append("pymupdf")
    
    if PDFMINER_AVAILABLE:
        available.append("PDFMiner")
    else:
        missing.append("pdfminer.six")
    
    if TABULA_AVAILABLE:
        available.append("Tabula")
    else:
        missing.append("tabula-py")
    
    if CAMELOT_AVAILABLE:
        available.append("Camelot")
    else:
        missing.append("camelot-py[cv]")
    
    return available, missing

def extract_with_pymupdf(pdf_path: str) -> Tuple[List[pd.DataFrame], Dict]:
    """Extract tables using PyMuPDF"""
    if not PYMUPDF_AVAILABLE:
        return [], {"error": "PyMuPDF not available"}
    
    try:
        doc = fitz.open(pdf_path)
        tables = []
        metadata = {
            "total_pages": len(doc),
            "method": "PyMuPDF",
            "tables_found": 0
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try to find tables
            tabs = page.find_tables()
            
            for tab in tabs:
                # Extract table data
                table_data = tab.extract()
                if table_data:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    df.name = f"Page_{page_num + 1}_Table_{len(tables) + 1}"
                    tables.append(df)
        
        metadata["tables_found"] = len(tables)
        doc.close()
        return tables, metadata
        
    except Exception as e:
        return [], {"error": f"PyMuPDF error: {str(e)}"}

def extract_with_tabula(pdf_path: str, pages: str = "all") -> Tuple[List[pd.DataFrame], Dict]:
    """Extract tables using Tabula"""
    if not TABULA_AVAILABLE:
        return [], {"error": "Tabula not available"}
    
    try:
        # Try different extraction methods
        methods = [
            {"method": "lattice", "multiple_tables": True},
            {"method": "stream", "multiple_tables": True},
            {"method": "lattice", "multiple_tables": False},
            {"method": "stream", "multiple_tables": False}
        ]
        
        best_result = []
        best_metadata = {"tables_found": 0}
        
        for method_config in methods:
            try:
                if method_config["multiple_tables"]:
                    dfs = tabula.read_pdf(
                        pdf_path,
                        pages=pages,
                        multiple_tables=True,
                        lattice=(method_config["method"] == "lattice"),
                        stream=(method_config["method"] == "stream")
                    )
                else:
                    df = tabula.read_pdf(
                        pdf_path,
                        pages=pages,
                        lattice=(method_config["method"] == "lattice"),
                        stream=(method_config["method"] == "stream")
                    )
                    dfs = [df] if not df.empty else []
                
                if len(dfs) > len(best_result):
                    best_result = dfs
                    best_metadata = {
                        "method": f"Tabula ({method_config['method']})",
                        "tables_found": len(dfs),
                        "multiple_tables": method_config["multiple_tables"]
                    }
                    
            except Exception as e:
                continue
        
        # Add names to DataFrames
        for i, df in enumerate(best_result):
            df.name = f"Table_{i + 1}"
        
        return best_result, best_metadata
        
    except Exception as e:
        return [], {"error": f"Tabula error: {str(e)}"}

def extract_with_camelot(pdf_path: str, pages: str = "all") -> Tuple[List[pd.DataFrame], Dict]:
    """Extract tables using Camelot"""
    if not CAMELOT_AVAILABLE:
        return [], {"error": "Camelot not available"}
    
    try:
        # Try both lattice and stream methods
        methods = ["lattice", "stream"]
        best_result = []
        best_metadata = {"tables_found": 0}
        
        for method in methods:
            try:
                tables = camelot.read_pdf(pdf_path, pages=pages, flavor=method)
                
                if len(tables) > len(best_result):
                    best_result = [table.df for table in tables]
                    best_metadata = {
                        "method": f"Camelot ({method})",
                        "tables_found": len(tables),
                        "accuracy": [table.accuracy for table in tables] if tables else []
                    }
                    
            except Exception as e:
                continue
        
        # Add names to DataFrames
        for i, df in enumerate(best_result):
            df.name = f"Table_{i + 1}"
        
        return best_result, best_metadata
        
    except Exception as e:
        return [], {"error": f"Camelot error: {str(e)}"}

def extract_text_with_pdfminer(pdf_path: str) -> Tuple[str, Dict]:
    """Extract raw text using PDFMiner"""
    if not PDFMINER_AVAILABLE:
        return "", {"error": "PDFMiner not available"}
    
    try:
        text = extract_text(pdf_path, laparams=LAParams())
        metadata = {
            "method": "PDFMiner",
            "text_length": len(text),
            "lines": len(text.split('\n'))
        }
        return text, metadata
        
    except Exception as e:
        return "", {"error": f"PDFMiner error: {str(e)}"}

def analyze_pdf_structure(text: str) -> Dict:
    """Analyze PDF structure to identify headers and patterns"""
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Try to identify repeating headers
    line_frequencies = {}
    for line in non_empty_lines:
        if len(line) > 20:  # Only consider substantial lines
            line_frequencies[line] = line_frequencies.get(line, 0) + 1
    
    # Find potential headers (lines that repeat)
    potential_headers = {line: count for line, count in line_frequencies.items() 
                        if count > 1 and len(line) > 30}
    
    # Analyze structure
    analysis = {
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "potential_headers": potential_headers,
        "has_repeating_headers": len(potential_headers) > 0,
        "avg_line_length": sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
    }
    
    return analysis

def clean_dataframe(df: pd.DataFrame, remove_empty_rows: bool = True, 
                   remove_empty_cols: bool = True) -> pd.DataFrame:
    """Clean extracted DataFrame"""
    df_clean = df.copy()
    
    if remove_empty_rows:
        df_clean = df_clean.dropna(how='all')
    
    if remove_empty_cols:
        df_clean = df_clean.dropna(axis=1, how='all')
    
    # Try to identify and remove header repetitions
    if len(df_clean) > 1:
        # Check if first row repeats in the data
        first_row = df_clean.iloc[0].astype(str)
        duplicate_rows = []
        
        for i in range(1, len(df_clean)):
            if df_clean.iloc[i].astype(str).equals(first_row):
                duplicate_rows.append(i)
        
        if duplicate_rows:
            df_clean = df_clean.drop(duplicate_rows)
    
    return df_clean

def main():
    st.set_page_config(
        page_title="PDF Table Extractor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 PDF Table Extractor")
    st.markdown("Extract tabular data from PDFs using multiple extraction methods")
    
    # Check dependencies
    available_libs, missing_libs = check_dependencies()
    
    if missing_libs:
        st.warning(f"Missing libraries: {', '.join(missing_libs)}")
        st.markdown("Install them using:")
        st.code(f"pip install {' '.join(missing_libs)}")
    
    st.success(f"Available libraries: {', '.join(available_libs)}")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Sidebar for options
            st.sidebar.header("Extraction Options")
            
            # Method selection
            available_methods = []
            if PYMUPDF_AVAILABLE:
                available_methods.append("PyMuPDF")
            if TABULA_AVAILABLE:
                available_methods.append("Tabula")
            if CAMELOT_AVAILABLE:
                available_methods.append("Camelot")
            if PDFMINER_AVAILABLE:
                available_methods.append("PDFMiner (Text)")
            
            if not available_methods:
                st.error("No PDF processing libraries available!")
                return
            
            selected_method = st.sidebar.selectbox("Select extraction method", available_methods)
            
            # Page selection
            pages_option = st.sidebar.selectbox(
                "Pages to extract",
                ["all", "first", "last", "custom"]
            )
            
            pages_param = "all"
            if pages_option == "first":
                pages_param = "1"
            elif pages_option == "last":
                pages_param = "-1"
            elif pages_option == "custom":
                custom_pages = st.sidebar.text_input("Enter pages (e.g., '1,3,5-7')", "1")
                pages_param = custom_pages
            
            # Cleaning options
            st.sidebar.subheader("Data Cleaning")
            remove_empty_rows = st.sidebar.checkbox("Remove empty rows", True)
            remove_empty_cols = st.sidebar.checkbox("Remove empty columns", True)
            
            # Extract button
            if st.sidebar.button("Extract Tables", type="primary"):
                with st.spinner("Extracting tables..."):
                    
                    # Extract based on selected method
                    if selected_method == "PyMuPDF":
                        tables, metadata = extract_with_pymupdf(tmp_path)
                    elif selected_method == "Tabula":
                        tables, metadata = extract_with_tabula(tmp_path, pages_param)
                    elif selected_method == "Camelot":
                        tables, metadata = extract_with_camelot(tmp_path, pages_param)
                    elif selected_method == "PDFMiner (Text)":
                        text, metadata = extract_text_with_pdfminer(tmp_path)
                        tables = []
                    
                    # Display results
                    if "error" in metadata:
                        st.error(metadata["error"])
                    else:
                        col1, col2 = st.columns([2, 1])
                        
                        with col2:
                            st.subheader("Extraction Metadata")
                            st.json(metadata)
                        
                        with col1:
                            if selected_method == "PDFMiner (Text)":
                                st.subheader("Extracted Text")
                                st.text_area("Raw Text", text, height=400)
                                
                                # Analyze structure
                                st.subheader("PDF Structure Analysis")
                                analysis = analyze_pdf_structure(text)
                                
                                st.write(f"**Total lines:** {analysis['total_lines']}")
                                st.write(f"**Non-empty lines:** {analysis['non_empty_lines']}")
                                st.write(f"**Has repeating headers:** {analysis['has_repeating_headers']}")
                                st.write(f"**Average line length:** {analysis['avg_line_length']:.1f}")
                                
                                if analysis['potential_headers']:
                                    st.write("**Potential repeating headers:**")
                                    for header, count in analysis['potential_headers'].items():
                                        st.write(f"- Appears {count} times: {header[:100]}...")
                                
                            else:
                                st.subheader("Extracted Tables")
                                
                                if not tables:
                                    st.warning("No tables found. Try a different extraction method.")
                                else:
                                    # Display each table
                                    for i, df in enumerate(tables):
                                        st.write(f"**Table {i+1}** ({df.shape[0]} rows × {df.shape[1]} columns)")
                                        
                                        # Clean the dataframe
                                        df_clean = clean_dataframe(df, remove_empty_rows, remove_empty_cols)
                                        
                                        # Show original vs cleaned
                                        tab1, tab2 = st.tabs(["Cleaned Data", "Original Data"])
                                        
                                        with tab1:
                                            st.dataframe(df_clean, use_container_width=True)
                                            
                                            # Download button
                                            csv = df_clean.to_csv(index=False)
                                            st.download_button(
                                                label=f"Download Table {i+1} as CSV",
                                                data=csv,
                                                file_name=f"table_{i+1}.csv",
                                                mime="text/csv"
                                            )
                                        
                                        with tab2:
                                            st.dataframe(df, use_container_width=True)
                                        
                                        st.divider()
            
            # PDF Structure Analysis (always available)
            if st.sidebar.button("Analyze PDF Structure"):
                with st.spinner("Analyzing PDF structure..."):
                    if PDFMINER_AVAILABLE:
                        text, text_metadata = extract_text_with_pdfminer(tmp_path)
                        
                        if "error" not in text_metadata:
                            st.subheader("PDF Structure Analysis")
                            analysis = analyze_pdf_structure(text)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Total Lines", analysis['total_lines'])
                                st.metric("Non-empty Lines", analysis['non_empty_lines'])
                                st.metric("Average Line Length", f"{analysis['avg_line_length']:.1f}")
                            
                            with col2:
                                st.metric("Has Repeating Headers", "Yes" if analysis['has_repeating_headers'] else "No")
                                st.metric("Potential Headers Found", len(analysis['potential_headers']))
                            
                            if analysis['potential_headers']:
                                st.subheader("Potential Repeating Headers")
                                for header, count in analysis['potential_headers'].items():
                                    with st.expander(f"Appears {count} times"):
                                        st.text(header)
                            
                            # Show sample text
                            st.subheader("Sample Text (First 1000 characters)")
                            st.text_area("", text[:1000], height=200)
                        else:
                            st.error("Could not analyze PDF structure")
                    else:
                        st.error("PDFMiner not available for structure analysis")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()