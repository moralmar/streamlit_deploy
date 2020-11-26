# Use an existing docker image as a base
FROM python:3.7

# streamlit runs on a default port of 8501
# So for the app to run, it is important to expose that particular port
EXPOSE 8501

# WORKDIR sets the working directory for the application.
# The rest of the commands will be executed from this path.
WORKDIR /app

# Here COPY command copies all of the files from your Docker client’s
# current directory to the working directory of the image.
COPY . .

# install some dependencies
RUN pip install -r requirements.txt

# Tell the image what to do when it starts as a container
CMD streamlit run app.py