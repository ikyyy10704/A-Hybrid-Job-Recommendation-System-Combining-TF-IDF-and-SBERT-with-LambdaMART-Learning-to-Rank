from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import Profile

_INPUT = {"class": "input"}


class RegisterForm(UserCreationForm):
    full_name = forms.CharField(max_length=120, required=False, label="Nama lengkap",
                                widget=forms.TextInput(attrs={**_INPUT, "placeholder": "cth: Budi Santoso"}))
    email = forms.EmailField(required=False,
                             widget=forms.EmailInput(attrs={**_INPUT, "placeholder": "email@contoh.com"}))

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({**_INPUT, "placeholder": "username"})
        self.fields["password1"].widget.attrs.update({**_INPUT, "placeholder": "Kata sandi (min. 6 karakter)"})
        self.fields["password2"].widget.attrs.update({**_INPUT, "placeholder": "Ulangi kata sandi"})
        for f in self.fields.values():
            f.help_text = ""


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({**_INPUT, "placeholder": "username"})
        self.fields["password"].widget.attrs.update({**_INPUT, "placeholder": "kata sandi"})


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ["full_name", "job_title", "skills", "experience", "education", "location"]
        widgets = {
            "full_name": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: Budi Santoso"}),
            "job_title": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: Software Engineer, Data Analyst"}),
            "skills": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: Python, SQL, Machine Learning"}),
            "experience": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: 2 tahun membangun web dengan Laravel"}),
            "education": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: S1, D3, SMA"}),
            "location": forms.TextInput(attrs={**_INPUT, "placeholder": "cth: Jakarta, Bandung"}),
        }
