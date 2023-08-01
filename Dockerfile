FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY ./requirements.txt ./requirements.txt

# Install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add new user for running the application
RUN adduser --disabled-password --gecos "" user

# Change the ownership of the /app directory to the new user
RUN chown -R user:user /app

# Switch to the new user
USER user

# Copy the project files into the container.
COPY ./data ./data
COPY ./models ./models
COPY ./output ./output
COPY ./*.py ./

# Expose the port the app runs on
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]
