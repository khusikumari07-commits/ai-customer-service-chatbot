from multilingual_ai import multilingual_response


# --------------------------------------------------
# Hinglish / mixed-language test
# --------------------------------------------------

mixed_message = (
    "Mujhe artificial intelligence ke baare mein "
    "simple language mein samjha do aur ek real-life "
    "example bhi do."
)

mixed_response = multilingual_response(
    mixed_message
)

print("\n--- Hinglish Test ---")
print("User:", mixed_message)
print("Assistant:", mixed_response)