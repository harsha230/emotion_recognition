from cv_models.deepface_model import run_deepface_model


def display_cv(result: dict):
    print("\nComputer Vision Model Result:")

    print(f"Model Used        : {result.get('model_name', 'N/A')}")
    print(f"Response Time     : {result.get('response_time', 'N/A')} seconds")
    print("Estimated Cost    : $0.00 (Local Model)")

    print("\nEmotion Percentages:")
    print(result.get("explanation", "N/A"))

    print("\n")


if __name__ == "__main__":
    image_path = "images/sad.webp"

    result = run_deepface_model(image_path)
    display_cv(result)
