docker build -t alessiogastaldo/jupyter-notebook-deeplearning .

docker run -d -p 8888:8888 -v /Users/alessiogastaldo/Documents/Personal/personal_repo/datascience-learning/examples/deep_learning_with_python:/home/jovyan/work alessiogastaldo/jupyter-notebook-deeplearning
