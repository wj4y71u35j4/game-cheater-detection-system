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

To run the system, you need to specify the directory containing only one Excel data file (`e.g. log_20230726.xlsx`). Replace the `"path/to/your/excel/folder"` path in the command below with the path to your Excel file folder. Please ensure the Excel file contains a sheet named "120秒內解除轉轉樂紀錄".

Use the following command to run the system:

```bash
docker run -p 4000:80 -v "path/to/your/excel/folder":/app/data cheater-detection
```

### Output

After running the Docker container, a file named `cheater-list.xlsx` will be created in the same directory as your excel log file. This file contains the list of detected cheaters.

| Column Name      | Description                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------|
| UserID           | Identifies each cheater.                                                                          |
| Unique_ClientIPs | Represents the list of unique IPs that have been used in the game.                                 |
| exemption        | If `1`, the user is in the exemption list; if `0`, the user is not.                               |
| ip_max_count     | Shows the maximum times an IP appears among the cheater list.                                     |
| times            | Represents the number of times the cheater game detection was activated for each `UserID`.         |

## Acknowledgments

* [Docker](https://www.docker.com/what-docker)
* [Python](https://www.python.org/)