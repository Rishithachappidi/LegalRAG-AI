import fitz
import os


class PDFExtractor:
    """
    Extract text from uploaded PDF files.
    """

    def __init__(self, output_folder="extracted_text"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract(self, uploaded_file):
        """
        Extract text from one uploaded PDF.

        Parameters:
            uploaded_file : Streamlit UploadedFile

        Returns:
            dict containing extraction information.
        """

        pdf = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        extracted_text = ""

        for page in pdf:
            extracted_text += page.get_text()

        filename = uploaded_file.name.replace(".pdf", ".txt")

        save_path = os.path.join(
            self.output_folder,
            filename
        )

        with open(
            save_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(extracted_text)

        return {
            "document_name": uploaded_file.name,
            "text_file": save_path,
            "pages": len(pdf),
            "characters": len(extracted_text),
            "preview": extracted_text[:1000],
            "text": extracted_text
        }