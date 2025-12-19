from flask import Flask, render_template, request
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
predictor = PredictionPipeline()

@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Helper function to safely parse form values
        def get_form_value(key, default=None, type_func=str):
            value = request.form.get(key, "")
            if value == "" and default is not None:
                return default
            try:
                return type_func(value) if value else default
            except (ValueError, TypeError):
                return default

        # Collect Titanic features from form
        # Required fields (from the main form)
        input_data = {
            "Pclass": get_form_value("Pclass", type_func=int),
            "Sex": get_form_value("Sex"),
            "Age": get_form_value("Age", type_func=float),
            "SibSp": get_form_value("SibSp", type_func=int),
            "Parch": get_form_value("Parch", type_func=int),
            "Fare": get_form_value("Fare", type_func=float),
            "Embarked": get_form_value("Embarked"),
        }
        
        # Optional fields (from the accordion - include only if provided)
        passenger_id = get_form_value("PassengerId", type_func=int)
        if passenger_id is not None:
            input_data["PassengerId"] = passenger_id
            
        name = get_form_value("Name")
        if name:
            input_data["Name"] = name
            
        ticket = get_form_value("Ticket")
        if ticket:
            input_data["Ticket"] = ticket
            
        cabin = get_form_value("Cabin")
        if cabin:
            input_data["Cabin"] = cabin

        # Validate required fields
        required_fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        missing_fields = [field for field in required_fields if field not in input_data or input_data[field] is None]
        
        if missing_fields:
            return f"""
            <div class="container mt-5">
                <h2>Error</h2>
                <div class="alert alert-danger">
                    <strong>Missing required fields:</strong> {', '.join(missing_fields)}<br>
                    Please fill in all required fields marked with *.
                </div>
                <a href="/" class="btn btn-secondary">Back to Form</a>
            </div>
            """

        # Make prediction
        prediction = predictor.predict(input_data)

        # Prepare result message
        result = "Survived" if prediction == 1 else "Did not survive"
        
        # Return result page with additional info
        return render_template("results.html", 
                             prediction=prediction,
                             result=result,
                             input_data=input_data)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <div class="container mt-5">
            <h2>Error</h2>
            <div class="alert alert-danger">
                <strong>Something went wrong:</strong><br>
                {str(e)}<br><br>
                <details>
                    <summary>Technical details</summary>
                    <pre>{error_details}</pre>
                </details>
            </div>
            <a href="/" class="btn btn-secondary">Back to Form</a>
        </div>
        """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)