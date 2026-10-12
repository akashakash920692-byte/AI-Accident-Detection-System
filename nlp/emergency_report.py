# ==========================================
# NLP EMERGENCY REPORT GENERATOR
# ==========================================

def generate_emergency_report(
    prediction,
    confidence,
    cars,
    motorcycles,
    buses,
    trucks,
    people,
    severity
):

    total_vehicles = cars + motorcycles + buses + trucks

    # --------------------------------------
    # Determine incident status
    # --------------------------------------

    if prediction.lower() == "accident":

        status = "ACCIDENT DETECTED"
        recommendation = (
            "Emergency response is recommended. "
            "The incident should be reviewed immediately."
        )

    else:

        status = "NO ACCIDENT DETECTED"
        recommendation = (
            "No immediate emergency response is required "
            "based on the current image analysis."
        )

    # --------------------------------------
    # Generate NLP report
    # --------------------------------------

    report = f"""
==========================================
        AI EMERGENCY INCIDENT REPORT
==========================================

INCIDENT STATUS
{status}

AI CONFIDENCE
{confidence:.2f}%

SCENE ANALYSIS
The AI system analyzed the road scene and
identified {total_vehicles} vehicle(s) and
{people} person(s).

VEHICLE DETAILS
• Cars: {cars}
• Motorcycles: {motorcycles}
• Buses: {buses}
• Trucks: {trucks}

SEVERITY ASSESSMENT
{severity}

AI ASSESSMENT
The computer vision system classified the
scene as "{prediction.upper()}".

RECOMMENDED ACTION
{recommendation}

==========================================
Generated automatically by the
AI Accident Detection System
==========================================
"""

    return report


# ------------------------------------------
# TEST THE NLP MODULE
# ------------------------------------------

if __name__ == "__main__":

    report = generate_emergency_report(
        prediction="accident",
        confidence=96.93,
        cars=3,
        motorcycles=2,
        buses=0,
        trucks=0,
        people=7,
        severity="HIGH"
    )

    print(report)