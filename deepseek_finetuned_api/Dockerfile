# # start from python base image
# FROM python:3.10

# # change working directory
# WORKDIR /deepseek_finetuned_api

# # add requirements file to image
# COPY ./requirements.txt /deepseek_finetuned_api/requirements.txt

# # install python libraries
# RUN pip install --no-cache-dir --upgrade -r /deepseek_finetuned_api/requirements.txt

# # add python code
# COPY ./API/ /deepseek_finetuned_api/API/

# # specify default commands
# CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "80"]




# start from python base image
FROM python:3.10

# working directory
WORKDIR /deepseek_finetuned_api

# add requirements file to image
COPY ./requirements.txt /deepseek_finetuned_api/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /deepseek_finetuned_api/requirements.txt

# add python code (this includes the data folder with merged_model)
COPY ./API/ /deepseek_finetuned_api/API/

# specify default commands
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "80"]


# docker build: docker build -t name_of_image .
# create containter with the image: docker run -d --name name_of_container -p 80:80 name_of_image