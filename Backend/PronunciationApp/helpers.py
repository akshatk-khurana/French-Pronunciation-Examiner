harmful_characters = ["<", ">", "?", "!"]

def sanitise(s):
    s = s.strip()
    for char in harmful_characters:
        s = s.replace(char, "")
    
    return s