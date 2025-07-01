# 1  Tag unknown tokens

python header_curator.py corpus/**/*.xml

# 2  Clean the whole corpus into ./clean/

python apply_header_map.py corpus/**/*.xml
