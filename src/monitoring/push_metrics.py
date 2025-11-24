# src/monitoring/push_metrics.py
import argparse
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def main(gateway: str, job: str, instance: str, drift: int):
    """
    Invia al Pushgateway una metrica di tipo Gauge:
    data_drift_flag = 1 (drift) / 0 (no drift)

    Sarà poi Prometheus a scrappare il Pushgateway e Grafana
    leggerà la metrica data_drift_flag.
    """
    reg = CollectorRegistry()
    g = Gauge("data_drift_flag", "1 if drift detected else 0", registry=reg)
    g.set(float(drift))
    push_to_gateway(
        gateway,
        job=job,
        grouping_key={"instance": instance},
        registry=reg,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://pushgateway:9091")
    ap.add_argument("--job", default="retrain_sentiment")
    ap.add_argument("--instance", default="airflow")
    ap.add_argument("--drift", type=int, required=True)
    args = ap.parse_args()
    main(args.gateway, args.job, args.instance, args.drift)
