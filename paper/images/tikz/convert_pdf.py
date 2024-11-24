from pathlib import Path
import fitz     # from PyMuPDF package


def convert_pdf(filename: Path):
    pdf = fitz.open(filename)
    page = pdf.load_page(0)
    pixmap = page.get_pixmap(dpi=300)
    output_filename = filename.stem + ".png"
    pixmap.save(filename.parent.parent / output_filename)
    print(f"Saved image: {output_filename}")


def cleanup(directory: Path):
    file_types_to_remove = [
        "aux",
        "fdb_latexmk",
        "fls",
        "log",
        "gz",
    ]
    for file_type in file_types_to_remove:
        for file in directory.glob(f"*.{file_type}"):
            file.unlink()


if __name__ == "__main__":
    directory = Path("images/tikz")
    for file in directory.glob("*.pdf"):
        convert_pdf(file)
    cleanup(directory)