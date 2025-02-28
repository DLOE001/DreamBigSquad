class SMSForm(forms.Form):
    sms_text = forms.CharField(
        label="Enter SMS Message", widget=forms.Textarea(attrs={"rows": 4, "cols": 50})
    )
    image = forms.ImageField(
        label="Upload Image (Optional)",
        required=False,  # Make the image upload optional
    )
