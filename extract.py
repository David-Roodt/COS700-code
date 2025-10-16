from zipfile import ZipFile, BadZipFile

print("Starting Extract")

with ZipFile("wikiart.zip", 'r') as zip_ref:
    for file in zip_ref.infolist():
        try:
            zip_ref.extract(file, "")
        except BadZipFile:
            print(f"Skipped corrupt file: {file.filename}")
        except Exception as e:
            print(f"Error extracting {file.filename}: {e}")

print("Finished!")
