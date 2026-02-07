FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl procps
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8080
CMD ["/entrypoint.sh"]
