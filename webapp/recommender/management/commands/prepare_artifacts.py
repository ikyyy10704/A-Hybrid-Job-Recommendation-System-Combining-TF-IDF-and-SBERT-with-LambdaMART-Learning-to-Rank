"""
Management command: membangkitkan seluruh artifact model (sekali jalan).

  python manage.py prepare_artifacts
  python manage.py prepare_artifacts --no-cache    # paksa hitung ulang SBERT

Proses: TF-IDF -> SBERT (berat) -> ekstraksi fitur -> latih LambdaMART -> evaluasi.
Hasil disimpan di folder Model/.
"""
from django.conf import settings
from django.core.management.base import BaseCommand

from recommender.ml import prepare


class Command(BaseCommand):
    help = "Membangkitkan artifact (SBERT, LambdaMART, evaluasi) ke folder Model/."

    def add_arguments(self, parser):
        parser.add_argument("--no-cache", action="store_true",
                            help="Abaikan cache sbert_norm.joblib dan hitung ulang embedding.")

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE(f"Model dir: {settings.MODEL_DIR}"))
        prepare.run(settings.MODEL_DIR, verbose=True,
                    use_sbert_cache=not options["no_cache"])
        self.stdout.write(self.style.SUCCESS("Artifact selesai dibangkitkan."))
