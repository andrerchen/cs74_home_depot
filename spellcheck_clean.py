
filename = "spellcheck_raw.txt"

with open(filename) as f:
	lines = f.readlines()

lines = [line.replace(',', '').replace('\'', '') for line in lines]

spellcheck = {}

for line in lines:
	tokens = line.split(':')
	spellcheck[tokens[0].strip()] = tokens[1].strip()


