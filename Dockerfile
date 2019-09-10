# import other container from the internet that contains python and anaconda standard installation
FROM continuumio/miniconda3:latest

# install all package requirements
RUN conda env update -n root --file requirements.yml

# give our container name
LABEL Name=healthscore-web-service Version=0.0.1
# expose port for accessing the container
EXPOSE 5000 

# create working directory
WORKDIR /predictive_maintenance_service

# copy all files in our local directory to container directory
# ADD source destination
ADD . /predictive_maintenance_service

# run the application
RUN chmod +x /health_score_service/start_service.sh
RUN sed -i -e 's/\r$//' /pmm_web_service/start_service.sh
ENTRYPOINT ["./start_service.sh"]