const form = document.getElementById("predictionForm");

const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

const predictionText = document.getElementById("predictionText");
const probabilityText = document.getElementById("probabilityText");
const riskText = document.getElementById("riskText");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    // Hide previous results/errors
    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    // Collect form data
    const formData = new FormData(form);

    // Convert FormData to JSON
    const data = {};

    formData.forEach((value, key) => {

        // Convert numeric fields into numbers
        if (
            key === "SeniorCitizen" ||
            key === "tenure" ||
            key === "MonthlyCharges" ||
            key === "TotalCharges"
        ) {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }

    });


    try {

        // Send request to Flask API
        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        const result = await response.json();


        // Handle API error
        if (!response.ok) {

            throw new Error(
                result.error || "Prediction failed."
            );

        }


        // Display prediction
        predictionText.textContent =
            result.prediction;


        // Convert probability to percentage
        const probability =
            (result.churn_probability * 100).toFixed(2);


        probabilityText.textContent =
            `${probability}%`;


        // Display risk
        riskText.textContent =
            result.risk;


        // Show result
        resultBox.classList.remove("hidden");


        // Scroll to result
        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }


    catch (error) {

        console.error(error);

        errorBox.textContent =
            error.message;

        errorBox.classList.remove("hidden");

        errorBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }

});