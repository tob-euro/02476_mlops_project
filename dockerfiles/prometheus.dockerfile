FROM prom/prometheus:latest
COPY configs/prometheus.yaml /etc/prometheus/prometheus.yaml
CMD ["--config.file=/etc/prometheus/prometheus.yaml"]
