# Game Cheater Detection System

This repository contains a Docker image for a game cheater detection system. The Docker image is saved as a .tar file named cheater-detection.tar.

## Getting Started

These instructions will guide you on how to load the Docker image from the .tar file and run the Docker container.

### Prerequisites

You need to have Docker installed on your machine. If you don't have Docker installed, please visit the official Docker website for installation instructions.

### Loading the Docker Image

The Docker image is stored in the cheater-detection.tar file. To load this image, you need to run the following command in the terminal:

```
docker load -i /path/to/cheater-detection.tar
```

Replace `/path/to/` with the actual path where the cheater-detection.tar file is located. After running this command, the Docker image will be available on your machine.

### Running the Docker Container

To run the Docker container, you need to have an Excel file named `log_20230726.xlsx` in the `D:/1.-Projects/Ran2_analysis/game-cheater-detection-jupyter/data` directory. 

Once you have the Excel file ready, you can run the Docker container using the following command:

```
docker run -p 4000:80 -v "D:/1.-Projects/Ran2_analysis/game-cheater-detection-jupyter/data":/app/data cheater-detection
```

This command will start the Docker container and map port 4000 of your machine to port 80 of the Docker container. It also mounts the data directory from your host machine to the Docker container.

### Output

After running the Docker container, a file named `cheater-list.csv` will be created in the same directory as `log_20230726.xlsx`. This file contains the list of detected cheaters.