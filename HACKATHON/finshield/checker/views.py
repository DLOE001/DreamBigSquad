from django.shortcuts import render
from .forms import SMSForm
from django.core.files.storage import FileSystemStorage  # For handling uploaded files


def is_scam(sms_text, image=None):  # Update to accept the image
    """
    Placeholder for scam detection logic.  Enhanced to (naively) consider the image.
    """
    sms_text = sms_text.lower()
    keywords = [
        "urgent",
        "payment",
        "bank",
        "account",
        "verify",
        "credit card",
        "prize",
        "won",
        "login",
        "password",
    ]

    if any(keyword in sms_text for keyword in keywords):
        return True, "Possible scam detected based on keywords."

    if image:
        # Basic image analysis (replace with actual image analysis logic)
        if image.size > 1024 * 1024 * 2:  # Check file size (2MB limit)
            return True, "Possible scam: Image is too large (over 2MB)."
        # You'd need to add actual image processing here, like checking for watermarks,
        # analyzing content using OCR, or comparing against known scam images.  This
        # placeholder just checks the size.

    return False, "No obvious scam indicators found (this doesn't guarantee safety!)."


def check_sms(request):
    if request.method == "POST":
        form = SMSForm(
            request.POST, request.FILES
        )  # Include request.FILES for image uploads
        if form.is_valid():
            sms_text = form.cleaned_data["sms_text"]
            image = form.cleaned_data["image"]

            # Save the image (optional, but good practice for demonstration)
            if image:
                fs = FileSystemStorage()  # Default storage (MEDIA_ROOT)
                filename = fs.save(image.name, image)
                uploaded_file_url = fs.url(filename)
            else:
                uploaded_file_url = None

            is_scam_result, summary = is_scam(
                sms_text, image
            )  # Pass the image to is_scam

            context = {
                "form": form,
                "sms_text": sms_text,
                "image_url": uploaded_file_url,  # Pass the image URL to the template
                "is_scam": is_scam_result,
                "summary": summary,
            }
            return render(request, "checker/result.html", context)
        else:
            context = {"form": form}
            return render(request, "checker/check_sms.html", context)
    else:
        form = SMSForm()
        context = {"form": form}
        return render(request, "checker/check_sms.html", context)
