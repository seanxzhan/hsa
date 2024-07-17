```
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
```

flexicubes doesn't work with pyrender, make sure the file where you run flexicubes doesn't import other files containing pyrener

preprocess_3 to get entire graph
write further merge info json
preprocess_17 save_data=False to get further merged graph
peek to get parts of interest
preprocess_17 to get actual data