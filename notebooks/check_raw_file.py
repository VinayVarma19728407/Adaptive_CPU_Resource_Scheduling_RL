file_path = "C:/Users/gvkva/Desktop/My ongoing projects/Machine Learning Project/Adaptive-Multi-Core-CPU-Scheduler-RL/data/raw/batch_instance.csv"

with open(file_path, "r") as file:
    for i in range(5):
        print(file.readline())