import random

def load_in_files(source=None):
    """Load in a file of text"""
    with open(source) as f:
        lines = f.read().splitlines()
    return lines

if __name__ == "__main__":
    adjectives = load_in_files('./name-generator/adjectives.txt')
    nouns = load_in_files('./name-generator/nouns.txt')
    random_name = str( random.choice(adjectives) ) + str( random.choice(nouns).title() )
    print(random_name)
