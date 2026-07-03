# GENEVO
A simulation-guided inverse-design framework that integrates mechanistic biosensor modeling, synthetic patient cohorts, surrogate modeling, and Bayesian optimization to computationally identify biosensor parameter configurations that best satisfy predefined clinical objectives.

The pipeline looks like this:

Clinical objective

        │
        ▼

Mechanistic biosensor simulator

        │
        ▼

Synthetic patient cohorts

        │
        ▼

Performance evaluation
(DR, FP, therapy, safety, etc.)

        │
        ▼

Surrogate modeling

        │
        ▼

Bayesian Optimization

        │
        ▼

Optimal biosensor parameters

        │
        ▼

Validation using the mechanistic simulator


# What problem does GENEVO solve?
Traditional biosensor design is often:

Engineer picks parameters

        │
        ▼

Tests them

        │
        ▼

Changes them manually

        │
        ▼

Repeats

GENEVO changes the workflow to:

Engineer specifies the desired clinical behavior

        │
        ▼

GENEVO automatically searches

        │
        ▼

Returns biosensor parameters predicted
to best satisfy that objective


# What are the inputs?
Examples include:
- biomarker concentration models,
- mechanistic kinetics (e.g., Langmuir binding),
- synthetic patient populations,
- clinical objectives,
- optimization constraints,
- parameter bounds.


# What are the outputs?
Things like:
- optimal dissociation constants (Kd),
- biomarker weighting,
- sensor sensitivity,
- operating thresholds,
- trade-offs between detection and safety,
- Pareto-optimal designs (for multi-objective optimization).