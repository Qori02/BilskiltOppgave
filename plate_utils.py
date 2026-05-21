def clean_plate_text(text):
    return (
        text.replace(" ", "")
            .replace("-", "")
            .replace(".", "")
            .replace("_", "")
            .replace(":", "")
            .replace("[", "")
            .replace("]", "")
            .replace("|", "")
            .upper()
    )