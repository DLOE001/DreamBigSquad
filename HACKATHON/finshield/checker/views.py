from django.shortcuts import render
from .forms import SMSForm
from django.core.files.storage import FileSystemStorage
import requests  # Import the requests library
import json  # Import the json library

# Replace with your actual backend service URL
BACKEND_SERVICE_URL = (
    "http://127.0.0.1:8001/process_image/"  # use different port, or container
)


def is_scam(sms_text):
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

    return False, "No obvious scam indicators found (this doesn't guarantee safety!)."


def check_sms(request):
    if request.method == "POST":
        form = SMSForm(request.POST, request.FILES)
        if form.is_valid():
            sms_text = form.cleaned_data["sms_text"]
            image = form.cleaned_data["image"]

            if image:
                # Prepare the multipart/form-data request
                files = {"image": (image.name, image.read(), image.content_type)}
                data = {"sms_text": sms_text}  # Include SMS text in the request

                try:
                    # Send the request to the backend service
                    response = requests.post(
                        BACKEND_SERVICE_URL, files=files, data=data
                    )
                    response.raise_for_status()  # Raise an exception for bad status codes
                    result = response.json()  # Parse the JSON response

                    # Extract the verdict and summary from the response
                    is_scam_result = result.get("is_scam", False)
                    summary = result.get("summary", "No summary provided by backend.")

                except requests.exceptions.RequestException as e:
                    # Handle connection errors or bad responses
                    is_scam_result = False
                    summary = f"Error contacting backend service: {e}"

                except json.JSONDecodeError as e:
                    # Handle invalid JSON responses
                    is_scam_result = False
                    summary = f"Error decoding JSON response: {e}"

            else:
                # No Image
                is_scam_result, summary = is_scam(sms_text)

            context = {
                "form": form,
                "sms_text": sms_text,
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
