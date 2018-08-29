# Getting started
First install docker: https://www.docker.com/get-started  
Copy the config-shared.yml to config.yml with your prefered settings.  
`docker-compose up pipeline` to run a pipeline with test data.  
Change the pipeline service to your own input data. The `--profile` flag can be used to automatically add attributes to users and will match on column `Id`.
