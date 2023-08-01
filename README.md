# Game Cheater Detection System

The Game Cheater Detection System is a project that utilizes Docker to identify and list potential cheaters in games based on specific metrics. The system processes data from Excel files and generates a CSV file containing the list of detected cheaters.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

Before running the system, please ensure that you have Docker installed on your machine. If you don't have Docker installed, you can download it [here](https://www.docker.com/get-started).

### Building the Docker Image

To build the Docker image for the system, navigate to the project directory in your terminal and run the following command:

```bash
docker build -t cheater-detection .
```

### Running the System

To run the system, you need to specify the directory containing only one Excel data file (`e.g. log_20230726.xlsx`). Replace the `"D:/1.-Projects/Ran2_analysis/game-cheater-detection-jupyter/data"` path in the command below with the path to your Excel file folder. Please ensure the Excel file is named `log_20230726.xlsx` and contains a sheet named "120秒內解除轉轉樂紀錄".

Use the following command to run the system:

```bash
docker run -p 4000:80 -v "D:/1.-Projects/Ran2_analysis/game-cheater-detection-jupyter/data":/app/data cheater-detection
```

### Output

After running the Docker container, a file named `cheater-list.csv` will be created in the same directory as your excel log file. This file contains the list of detected cheaters.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* [Docker](https://www.docker.com/what-docker)
* [Python](https://www.python.org/)

Enjoy detecting cheaters in your game!

Please note: This README is just a template and might need to be adjusted to fit your specific needs, especially the Contributing, License, and Acknowledgments sections. It's important to ensure that the paths and commands are accurate for users who want to build and run your Docker image.
