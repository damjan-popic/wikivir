# Clean the categories

python header_curator.py corpus/**/*.xml

## Clean the whole corpus into ./clean/

python apply_header_map.py corpus/**/*.xml

# activate the same venv you used for Classla etc.:
pip install bertopic[visualization] sentence-transformers

# one doc per line, UTF-8 (tokenised/cleansed)
python run_bertopic.py docs.txt --output slovene_topics --workers 8

# if you reconnect later:
python run_bertopic.py docs.txt --output slovene_topics --resume

