from gitingest import ingest

summary, tree, content = ingest("https://github.com/cyclotruc/gitingest")
print(summary)