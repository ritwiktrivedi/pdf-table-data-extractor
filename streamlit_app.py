import streamlit as st
import pandas as pd
import io
import tempfile
import os
from typing import List, Dict, Optional, Tuple
import re
import numpy as np

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

    return available, missing


def is_likely_header_row(row_data: List[str], min_non_empty_ratio: float = 0.6) -> bool:
    """
    Determine if a row is likely a header row based on content analysis
    """
    if not row_data:
        return False

    # Convert to strings and clean
    clean_row = [str(cell).strip() for cell in row_data]

    # Calculate ratio of non-empty cells
    non_empty_cells = sum(
        1 for cell in clean_row if cell and cell.lower() not in ['', 'nan', 'none'])
    non_empty_ratio = non_empty_cells / \
        len(clean_row) if len(clean_row) > 0 else 0

    # If too many empty cells, likely not a header
    if non_empty_ratio < min_non_empty_ratio:
        return False

    # Check for header-like characteristics
    header_indicators = 0

    for cell in clean_row:
        if not cell or cell.lower() in ['', 'nan', 'none']:
            continue

        # Check for common header patterns
        if any(keyword in cell.lower() for keyword in ['name', 'date', 'amount', 'total', 'id', 'number', 'type', 'status']):
            header_indicators += 1

        # Check if mostly alphabetic (header-like)
        if re.match(r'^[a-zA-Z\s]+$', cell):
            header_indicators += 1

        # Check for camelCase or snake_case (common in headers)
        if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*[a-zA-Z0-9]$', cell):
            header_indicators += 1

    # If we have some header indicators and good fill ratio, likely a header
    return header_indicators >= min(2, len([c for c in clean_row if c]))


def detect_real_header(table_data: List[List[str]], max_header_row: int = 3,
                       header_mode: str = "auto") -> Tuple[int, List[str]]:
    """
    Detect the actual header row in table data
    Args:
        table_data: List of rows (each row is a list of cells)
        max_header_row: Maximum row index to check for headers
        header_mode: "auto", "first_only", or "every_page"
    Returns: (header_row_index, header_row_data)
    """
    if not table_data:
        return -1, []

    if header_mode == "first_only":
        # Only use first row as header
        return 0, table_data[0]

    best_header_idx = 0
    best_score = -1

    # Check first few rows for the best header candidate
    for i in range(min(max_header_row, len(table_data))):
        row = table_data[i]

        # Skip completely empty rows
        if not any(str(cell).strip() for cell in row):
            continue

        # Calculate header score
        score = 0

        # Check non-empty ratio
        non_empty_cells = sum(1 for cell in row if str(cell).strip() and str(
            cell).strip().lower() not in ['nan', 'none'])
        non_empty_ratio = non_empty_cells / len(row) if len(row) > 0 else 0

        if non_empty_ratio > 0.5:
            score += 2

        # Check for header-like content
        if is_likely_header_row(row):
            score += 3

        # Prefer earlier rows (typical header position)
        score += (max_header_row - i) * 0.5

        if score > best_score:
            best_score = score
            best_header_idx = i

    return best_header_idx, table_data[best_header_idx] if best_header_idx < len(table_data) else []


def merge_split_rows(df: pd.DataFrame, id_column_index: int = 0, threshold_empty_cols: int = 2) -> pd.DataFrame:
    """
    Merge rows that are likely split across multiple lines due to formatting issues.
    Args:
        df: DataFrame to clean.
        id_column_index: Index of column which likely starts a new row (e.g., an ID or serial number).
        threshold_empty_cols: If a row has this many or more empty columns, it may be a continuation.
    Returns:
        Cleaned DataFrame with split rows merged.
    """
    merged_rows = []
    current_row = None

    for _, row in df.iterrows():
        row_values = row.tolist()
        non_empty_count = sum(1 for val in row_values if str(val).strip())

        if current_row is None:
            current_row = row_values
            continue

        # Heuristic: If the "ID" column is not empty and there are enough filled values, start a new row
        is_new_row = str(row_values[id_column_index]).strip(
        ) != "" and non_empty_count > len(row_values) - threshold_empty_cols

        if is_new_row:
            merged_rows.append(current_row)
            current_row = row_values
        else:
            # Append current row's values to previous row
            for i in range(len(row_values)):
                if str(row_values[i]).strip():
                    if not str(current_row[i]).strip():
                        current_row[i] = row_values[i]
                    else:
                        current_row[i] = str(
                            current_row[i]) + " " + str(row_values[i])

    if current_row:
        merged_rows.append(current_row)

    return pd.DataFrame(merged_rows, columns=df.columns)


def extract_with_pymupdf(pdf_path: str, header_mode: str = "auto") -> Tuple[List[pd.DataFrame], Dict]:
    """Extract tables using PyMuPDF with improved header detection"""
    if not PYMUPDF_AVAILABLE:
        return [], {"error": "PyMuPDF not available"}

    try:
        doc = fitz.open(pdf_path)
        tables = []
        metadata = {
            "total_pages": len(doc),
            "method": "PyMuPDF",
            "tables_found": 0,
            "extraction_details": []
        }

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Try to find tables
            tabs = page.find_tables()

            for tab_idx, tab in enumerate(tabs):
                # Extract table data
                table_data = tab.extract()
                if not table_data:
                    continue

                # Detect real header row
                header_mode_param = "auto" if header_mode == "Auto-detect" else \
                    "first_only" if header_mode == "Only at the beginning" else "every_page"
                header_idx, header_row = detect_real_header(
                    table_data, header_mode=header_mode_param)

                extraction_detail = {
                    "page": page_num + 1,
                    "table": tab_idx + 1,
                    "original_rows": len(table_data),
                    "detected_header_row": header_idx,
                    # First 5 columns for preview
                    "header_content": header_row[:5] if header_row else []
                }

                if header_idx >= 0 and header_idx < len(table_data):
                    # Use detected header
                    header = table_data[header_idx]
                    data_rows = table_data[header_idx + 1:]

                    # Clean header names
                    clean_header = []
                    for i, col in enumerate(header):
                        col_name = str(col).strip()
                        if not col_name or col_name.lower() in ['', 'nan', 'none']:
                            col_name = f"Column_{i+1}"
                        clean_header.append(col_name)

                    # Create DataFrame
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=clean_header)
                        df = merge_split_rows(df)
                    else:
                        df = pd.DataFrame(columns=clean_header)
                else:
                    # Fallback: use first row as header
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])

                df.name = f"Page_{page_num + 1}_Table_{tab_idx + 1}"
                tables.append(df)
                metadata["extraction_details"].append(extraction_detail)

        metadata["tables_found"] = len(tables)
        doc.close()
        return tables, metadata

    except Exception as e:
        return [], {"error": f"PyMuPDF error: {str(e)}"}


def extract_with_tabula(pdf_path: str, pages: str = "all", header_mode: str = "auto") -> Tuple[List[pd.DataFrame], Dict]:
    """Extract tables using Tabula with improved header detection"""
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
        best_metadata = {"tables_found": 0, "extraction_details": []}

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

                # Process each DataFrame to fix headers
                processed_dfs = []
                extraction_details = []

                for i, df in enumerate(dfs):
                    if df.empty:
                        continue

                    # Convert DataFrame to list format for header detection
                    table_data = [df.columns.tolist()] + df.values.tolist()

                    # Detect real header
                    header_mode_param = "auto" if header_mode == "Auto-detect" else \
                        "first_only" if header_mode == "Only at the beginning" else "every_page"
                    header_idx, header_row = detect_real_header(
                        table_data, header_mode=header_mode_param)

                    detail = {
                        "table": i + 1,
                        "original_rows": len(df),
                        "detected_header_row": header_idx,
                        "header_content": header_row[:5] if header_row else []
                    }

                    if header_idx > 0:  # Header is not the first row
                        # Reconstruct DataFrame with correct header
                        new_header = table_data[header_idx]
                        new_data = table_data[header_idx + 1:]

                        # Clean header names
                        clean_header = []
                        for j, col in enumerate(new_header):
                            col_name = str(col).strip()
                            if not col_name or col_name.lower() in ['', 'nan', 'none']:
                                col_name = f"Column_{j+1}"
                            clean_header.append(col_name)

                        if new_data:
                            new_df = pd.DataFrame(
                                new_data, columns=clean_header)
                            new_df = merge_split_rows(new_df)
                        else:
                            new_df = pd.DataFrame(columns=clean_header)

                        new_df.name = f"Table_{i + 1}"
                        processed_dfs.append(new_df)
                    else:
                        # Use original DataFrame
                        df.name = f"Table_{i + 1}"
                        processed_dfs.append(df)

                    extraction_details.append(detail)

                if len(processed_dfs) > len(best_result):
                    best_result = processed_dfs
                    best_metadata = {
                        "method": f"Tabula ({method_config['method']})",
                        "tables_found": len(processed_dfs),
                        "multiple_tables": method_config["multiple_tables"],
                        "extraction_details": extraction_details
                    }

            except Exception as e:
                continue

        return best_result, best_metadata

    except Exception as e:
        return [], {"error": f"Tabula error: {str(e)}"}


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


def clean_dataframe(df: pd.DataFrame, remove_empty_rows: bool = True,
                    remove_empty_cols: bool = True) -> pd.DataFrame:
    """Clean extracted DataFrame"""
    df_clean = df.copy()

    if remove_empty_rows:
        # Remove rows that are completely empty or only contain NaN/None/empty strings
        df_clean = df_clean.dropna(how='all')
        mask = df_clean.astype(str).apply(
            lambda x: x.str.strip().str.lower().isin(['', 'nan', 'none'])).all(axis=1)
        df_clean = df_clean[~mask]

    if remove_empty_cols:
        # Remove columns that are completely empty
        df_clean = df_clean.dropna(axis=1, how='all')
        for col in df_clean.columns:
            if df_clean[col].astype(str).str.strip().str.lower().isin(['', 'nan', 'none']).all():
                df_clean = df_clean.drop(columns=[col])

    return df_clean


def main():
    st.set_page_config(
        page_title="PDF Table Extractor",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 PDF Table Extractor")
    st.markdown(
        "Extract tabular data from PDFs using PyMuPDF and Tabula with improved header detection")

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
            if PDFMINER_AVAILABLE:
                available_methods.append("PDFMiner (Text)")

            if not available_methods:
                st.error("No PDF processing libraries available!")
                return

            selected_method = st.sidebar.selectbox(
                "Select extraction method", available_methods)

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
                custom_pages = st.sidebar.text_input(
                    "Enter pages (e.g., '1,3,5-7')", "1")
                pages_param = custom_pages

            # Header detection options
            st.sidebar.subheader("Header Detection")
            header_frequency = st.sidebar.radio(
                "How often do headers appear?",
                ["Only at the beginning", "On every page", "Auto-detect"],
                index=2,
                help="Choose how to handle headers across multiple pages"
            )

            # Cleaning options
            st.sidebar.subheader("Data Cleaning")
            remove_empty_rows = st.sidebar.checkbox("Remove empty rows", True)
            remove_empty_cols = st.sidebar.checkbox(
                "Remove empty columns", True)

            # Extract button
            if st.sidebar.button("Extract Tables", type="primary"):
                with st.spinner("Extracting tables..."):

                    # Extract based on selected method
                    if selected_method == "PyMuPDF":
                        tables, metadata = extract_with_pymupdf(
                            tmp_path, header_frequency)
                    elif selected_method == "Tabula":
                        tables, metadata = extract_with_tabula(
                            tmp_path, pages_param, header_frequency)
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
                            else:
                                st.subheader("Extracted Tables")

                                if not tables:
                                    st.warning(
                                        "No tables found. Try a different extraction method.")
                                else:
                                    # Display each table
                                    for i, df in enumerate(tables):
                                        st.write(
                                            f"**Table {i+1}** ({df.shape[0]} rows × {df.shape[1]} columns)")

                                        # Show extraction details if available
                                        if "extraction_details" in metadata and i < len(metadata["extraction_details"]):
                                            details = metadata["extraction_details"][i]
                                            st.caption(
                                                f"Header detected at row {details['detected_header_row']}")

                                        # Clean the dataframe
                                        df_clean = clean_dataframe(
                                            df, remove_empty_rows, remove_empty_cols)

                                        # Show original vs cleaned
                                        tab1, tab2 = st.tabs(
                                            ["Cleaned Data", "Original Data"])

                                        with tab1:
                                            st.dataframe(
                                                df_clean, use_container_width=True)

                                            # Download button
                                            csv = df_clean.to_csv(index=False)
                                            st.download_button(
                                                label=f"Download Table {i+1} as CSV",
                                                data=csv,
                                                file_name=f"table_{i+1}.csv",
                                                mime="text/csv"
                                            )

                                        with tab2:
                                            st.dataframe(
                                                df, use_container_width=True)

                                        st.divider()

            # PDF Structure Analysis (always available)
            if st.sidebar.button("Analyze PDF Structure"):
                with st.spinner("Analyzing PDF structure..."):
                    if PDFMINER_AVAILABLE:
                        text, text_metadata = extract_text_with_pdfminer(
                            tmp_path)

                        if "error" not in text_metadata:
                            st.subheader("PDF Structure Analysis")

                            # Show sample text
                            st.subheader("Sample Text (First 1000 characters)")
                            st.text_area("", text[:1000], height=200)
                        else:
                            st.error("Could not analyze PDF structure")
                    else:
                        st.error(
                            "PDFMiner not available for structure analysis")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
