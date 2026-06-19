from django.contrib.auth.models import User
from django.db import models


class Profile(models.Model):
    """Profil pencari kerja milik tiap pengguna.

    Field-field di sini sama dengan masukan form rekomendasi, sehingga sekali
    disimpan dapat otomatis mengisi form (dan dipakai ulang lintas sesi).
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    full_name = models.CharField("Nama lengkap", max_length=120, blank=True)
    job_title = models.CharField("Posisi yang dicari", max_length=200, blank=True)
    skills = models.CharField("Skills / kemampuan", max_length=500, blank=True)
    experience = models.CharField("Pengalaman / deskripsi", max_length=500, blank=True)
    education = models.CharField("Pendidikan", max_length=100, blank=True)
    location = models.CharField("Lokasi", max_length=120, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profil {self.user.username}"

    @property
    def initial(self):
        base = self.full_name or self.user.username
        return (base[:1] or "?").upper()
