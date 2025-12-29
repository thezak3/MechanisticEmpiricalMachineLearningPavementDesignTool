================================================================================
TXDOT PAVEMENT FATIGUE CRACKING DESIGN TOOL
Documentation
================================================================================

Version: 1.0
Date: December 2024
Developed by: [Your Name/Organization]

================================================================================
TABLE OF CONTENTS
================================================================================

1. Overview
2. Installation and Setup
3. Tool Features
4. How to Use
5. Understanding Results
6. Technical Details
7. Limitations and Assumptions
8. Troubleshooting
9. Contact Information

================================================================================
1. OVERVIEW
================================================================================

Purpose:
The TxDOT Pavement Fatigue Cracking Design Tool is a web-based application that
predicts fatigue cracking in flexible pavements using machine learning models.
It helps TxDOT engineers make informed decisions during pavement design by
providing rapid, data-driven predictions of long-term pavement performance.

Key Capabilities:
- Predict fatigue cracking at 5, 10, 15, or 20 year design lives
- Visualize cracking progression over time
- Compare multiple design alternatives
- Perform sensitivity analysis on key parameters
- Identify out-of-range design inputs

Machine Learning Models:
The tool uses an ensemble of three machine learning models:
1. XGBoost (Extreme Gradient Boosting)
2. LightGBM (Light Gradient Boosting Machine)
3. Random Forest

Predictions are averaged across all three models for robust, reliable results.

Training Data:
- 260 unique pavement cases
- 62,400 total observations
- Represents diverse TxDOT pavement designs and conditions
- Includes various mix types, traffic levels, and structural configurations

================================================================================
2. INSTALLATION AND SETUP
================================================================================

System Requirements:
- Python 3.8 or higher
- Internet connection (for Streamlit Cloud deployment)
- Modern web browser (Chrome, Firefox, Safari, Edge)

Required Files:
1. app.py - Main application code
2. xgboost_model.pkl - Trained XGBoost model
3. lightgbm_model.pkl - Trained LightGBM model
4. random_forest_model.pkl - Trained Random Forest model
5. scaler.pkl - Feature normalization scaler
6. requirements.txt - Python dependencies

Local Installation:
1. Install Python dependencies:
   pip install streamlit pandas numpy pickle plotly scikit-learn xgboost lightgbm

2. Place all required files in the same directory

3. Run the application:
   streamlit run app.py

4. Open web browser to the displayed URL (typically http://localhost:8501)

Streamlit Cloud Deployment:
1. Create a GitHub repository
2. Upload all required files to the repository
3. Go to https://share.streamlit.io/
4. Sign in with GitHub
5. Deploy the app by selecting your repository
6. Share the generated URL with TxDOT engineers

================================================================================
3. TOOL FEATURES
================================================================================

3.1 Instructions Page
- Displayed on first use
- Explains tool purpose, features, and limitations
- Users must acknowledge understanding before proceeding

3.2 Design Input Tab
Main interface for single design analysis:
- Input pavement design parameters
- Select design life (5, 10, 15, 20 years)
- Choose mix design and traffic level
- View predicted cracking and design status
- See cracking progression over time
- Perform sensitivity analysis

3.3 Design Comparison Tool Tab
Compare three design alternatives:
- Base design (user inputs)
- Beefed-up design (AC thickness +1 inch)
- Thinned design (AC thickness -1 inch)
- Side-by-side comparison with overlaid graphs
- Performance summary table

3.4 Out-of-Range Warnings
Alerts when inputs fall outside training data ranges:
- Ensures users are aware of prediction reliability
- Orange warning boxes highlight specific parameters
- Does not prevent calculation (allows engineering judgment)

3.5 Model Confidence Indicator
Shows agreement between three machine learning models:
- High confidence: Models agree closely (spread < 5%)
- Medium confidence: Moderate agreement (spread 5-10%)
- Low confidence: Models disagree (spread > 10%)
- Displayed as hover information next to prediction

3.6 Sensitivity Analysis
Evaluates impact of key parameter changes:
- AC thickness variations
- Base thickness variations
- Mix type alternatives
- Results shown as table and text summary
- Helps identify optimization opportunities

3.7 Cracking Progression Graph
Visual representation of predicted cracking over time:
- Year-by-year predictions from 0 to design life
- Reference lines at 15% (Good/Acceptable) and 30% (Acceptable/Failure)
- Interactive Plotly graph with hover details

================================================================================
4. HOW TO USE
================================================================================

4.1 Basic Workflow

Step 1: Read Instructions
- Review the instructions page on first use
- Click "I Understand - Proceed to Tool"

Step 2: Enter Design Parameters
Design Life & Mix:
- Select design life: 5, 10, 15, or 20 years
- Choose Performance Grade (PG): 64-22, 70-22, 76-22, etc.
- Select Mix Type: Type B, Type C, Type D, Superpave, SMA
- Set RAP Content: 0-50% (typically 0%, 15%, or 30%)

Pavement Structure:
- AC Thickness: Select 4.0, 5.5, or 7.0 inches
- AC Modulus: Enter value in ksi (typical range: 583-1335)
- Base Thickness: Select 8.0, 16.0, or 24.0 inches
- Base Modulus: Enter value in psi (typical range: 36.5-250)
- Subgrade Modulus: Select 5.0, 12.5, or 20.0 psi

Traffic Loading:
- Select traffic level: Light, Medium, Heavy, or Custom
  * Light: 500,000 ESALs
  * Medium: 2,000,000 ESALs
  * Heavy: 5,000,000 ESALs
  * Custom: Enter specific ESAL value

Step 3: Calculate Prediction
- Click "Calculate Prediction" button
- Wait for results to appear (typically < 1 second)

Step 4: Review Results
- Check predicted cracking percentage
- Note design status: Good, Acceptable, or Early Failure
- Review individual model predictions
- Examine cracking progression graph
- Analyze sensitivity results

Step 5: Compare Alternatives (Optional)
- Switch to "Design Comparison Tool" tab
- Click "Compare Designs" to see alternatives
- Review overlaid progression graphs
- Compare performance metrics

4.2 Interpreting Design Status

Good (Green):
- Predicted cracking < 15%
- Design meets performance criteria
- Recommended for implementation

Acceptable (Yellow):
- Predicted cracking 15-30%
- Design is acceptable but may require monitoring
- Consider optimization if feasible

Early Failure (Red):
- Predicted cracking > 30%
- Design may experience premature failure
- Modification recommended

4.3 Using Sensitivity Analysis

Purpose:
- Identify which parameters have greatest impact on cracking
- Guide design optimization efforts
- Understand trade-offs between alternatives

Interpretation:
- Negative change = reduction in cracking (improvement)
- Positive change = increase in cracking (degradation)
- Larger magnitude = greater sensitivity

Example:
If increasing AC thickness from 5.5 to 7.0 inches reduces cracking by 8%, 
this indicates AC thickness is a critical parameter for this design.

4.4 Design Comparison Workflow

Use Case:
Quickly evaluate if increasing or decreasing AC thickness is beneficial

Steps:
1. Complete a prediction in Design Input tab
2. Navigate to Design Comparison Tool tab
3. Click "Compare Designs"
4. Review overlaid graph showing three alternatives
5. Check summary table for final cracking values
6. Read key insights for quantitative impact

Applications:
- Cost-benefit analysis (thicker vs thinner pavements)
- Risk assessment (how much worse is thinned design?)
- Optimization (is beefed-up design significantly better?)

================================================================================
5. UNDERSTANDING RESULTS
================================================================================

5.1 Prediction Accuracy

Model Performance:
- Mean Absolute Error: 2.69%
- R² Score: 0.9881
- Validation: Tested on 80 independent cases

Reliability:
Predictions are most reliable when:
- All inputs are within training data ranges
- Model confidence is High
- Design parameters are typical for TxDOT projects

Predictions may be less reliable when:
- Multiple inputs are out of range (orange warnings appear)
- Model confidence is Low
- Design is highly unusual or extreme

5.2 Model Confidence

High Confidence:
- All three models predict similar values
- Spread between models < 5%
- High reliability of prediction

Medium Confidence:
- Models show moderate agreement
- Spread between models 5-10%
- Reasonable reliability

Low Confidence:
- Models disagree significantly
- Spread between models > 10%
- Use caution; consider additional analysis
- May indicate unusual design or out-of-range inputs

5.3 Cracking Progression Patterns

Typical Pattern:
- Minimal cracking in early years (0-5 years)
- Gradual increase in middle years (5-15 years)
- Accelerated cracking in later years (15-20 years)

Concerning Patterns:
- Rapid early cracking (>5% before year 5)
- Steep acceleration (doubling year-over-year)
- Exceeding 15% before half of design life

5.4 Practical Considerations

Engineering Judgment:
- This tool aids but does not replace engineering judgment
- Consider site-specific factors not captured by models
- Account for construction quality, drainage, climate
- Validate predictions with field experience

Design Iterations:
- Try multiple design alternatives
- Use sensitivity analysis to guide improvements
- Balance performance with cost
- Consider maintenance implications

Documentation:
- Screenshot results for project files
- Note key assumptions and inputs
- Record any out-of-range warnings
- Document rationale for design decisions

================================================================================
6. TECHNICAL DETAILS
================================================================================

6.1 Input Parameters

Required Inputs:
1. Design Life: Target pavement service life (years)
2. Performance Grade (PG): Asphalt binder grade
3. Mix Type: Aggregate gradation and mix design
4. RAP Content: Percentage of recycled asphalt pavement
5. AC Thickness: Asphalt concrete layer thickness (inches)
6. Base Thickness: Base layer thickness (inches)
7. Base Modulus: Base layer stiffness (ksi)
8. Subgrade Modulus: Subgrade resilient modulus (ksi)
9. Traffic: Total equivalent single axle loads (ESALs)

Auto-Calculated Parameters:
- AC Modulus: Automatically determined from PG grade and mix type based on 
  average values from training data. This accounts for temperature and seasonal
  variations that were modeled mechanistically in the original dataset.

Derived Parameters:
The tool automatically calculates additional parameters:
- Paris Law parameters (A and n) from PG and Mix Type
- Polynomial features (Age², ESALs²)
- Interaction features (Age × ESALs, etc.)
- Paris Law damage features

6.2 Valid Input Ranges

Parameter                Min        Max        Typical
AC Thickness            4.0 in     7.0 in     5.5 in
AC Modulus              Auto-calculated from PG grade and mix type
Base Thickness          8.0 in     24.0 in    16.0 in
Base Modulus            36.5 ksi   250 ksi    37 ksi
Subgrade Modulus        5.0 ksi    20.0 ksi   10 ksi
RAP Content             0%         30%        15%
Design ESALs            100K       10M        2M

Note: AC Modulus is automatically determined from the selected Performance 
Grade and Mix Type based on average values from training data. Values outside 
these ranges trigger warnings but do not prevent calculation. Use engineering 
judgment for out-of-range predictions.

6.3 Model Architecture

XGBoost:
- Gradient boosting with decision trees
- Hyperparameters optimized via RandomizedSearchCV
- Monotone constraints enforce physical relationships
- 300 estimators, max depth 4, learning rate 0.10

LightGBM:
- Gradient boosting with leaf-wise tree growth
- Hyperparameters optimized via RandomizedSearchCV
- Monotone constraints match XGBoost
- Optimized parameters from 100-iteration search

Random Forest:
- Ensemble of decision trees
- Hyperparameters optimized via RandomizedSearchCV
- No monotone constraints (not supported)
- Optimized parameters from 100-iteration search

Ensemble Method:
- Simple average of three model predictions
- Reduces individual model bias
- Improves overall robustness
- Mean Absolute Error: 2.69%

6.4 Feature Engineering

Base Features (10):
1. Pavement_Age_Months
2. Cumulative_Monthly_ESALs
3. AC_Thickness
4. RAP_Percent
5. A (Paris Law parameter, scaled)
6. n (Paris Law parameter)
7. AC_Modulus_ksi
8. Base_Thickness
9. Base_Modulus
10. Subgrade_Modulus

Polynomial Features (5):
11. Age_Squared
12. ESALs_Squared
13. Age_x_ESALs
14. AC_Thickness_x_Modulus
15. Base_Thickness_x_Modulus

Paris Law Features (4):
16. Paris_ESALs = A × ESALs^n
17. Paris_Age = A × Age^n
18. Paris_Combined = A × (ESALs + Age)^n
19. Paris_Modulus_ESALs = (A / (AC_Modulus + 1)) × ESALs^n

Total: 19 features used by models

6.5 Target Transformation

Target Variable:
- Fatigue_Cracking_Area_Percent (0-100%)

Transformation:
1. Square root transformation applied before training
   - Reduces impact of outliers
   - Improves model performance for percentage data

2. Predictions made on sqrt-transformed scale

3. Reverse transformation applied (squaring)

4. Clipping to valid range [0, 100]

6.6 Feature Scaling

Method: StandardScaler (z-score normalization)
- Centers features to mean = 0
- Scales features to standard deviation = 1
- Fitted on training data only
- Same scaler applied to all predictions

Importance:
- Ensures features on different scales contribute equally
- Required for optimal model performance
- Critical for accurate predictions

6.7 Paris Law Parameters

Background:
Paris Law describes fatigue crack growth:
  da/dN = A × (ΔK)^n

Where:
- da/dN = crack growth rate
- A = material constant
- ΔK = stress intensity factor range
- n = exponent

Implementation:
- A and n values determined by PG grade and mix type
- Based on TxDOT pavement performance data
- A scaled by 1×10^6 for numerical stability
- Strong correlation (r = -0.908) between A and n

Feature Importance:
- Paris_Age is the most important feature
- Captures time-dependent fatigue accumulation
- Combines material properties (A, n) with time

================================================================================
7. LIMITATIONS AND ASSUMPTIONS
================================================================================

7.1 Data Limitations

Training Data Scope:
- Based on 260 TxDOT pavement cases
- May not represent all possible conditions
- Limited to flexible pavements
- Does not include rigid or composite pavements

Temporal Scope:
- Maximum observed age: 20 years (240 months)
- Predictions beyond 20 years are extrapolations
- Use caution for extended design lives

Geographic Scope:
- Based on Texas climate and conditions
- May not be applicable to other regions
- Does not account for extreme weather events

7.2 Model Limitations

Black Box Nature:
- Machine learning models are complex
- Difficult to trace specific predictions to physical mechanisms
- Rely on statistical patterns rather than mechanistic principles

Interpolation vs Extrapolation:
- Most accurate for inputs within training data ranges
- Less reliable for out-of-range inputs
- Out-of-range warnings help identify extrapolation

Model Agreement:
- Low confidence predictions warrant additional scrutiny
- Consider mechanistic analysis for critical projects
- Validate with field data when available

7.3 Assumptions

Traffic Distribution:
- ESALs accumulate linearly over time
- Monthly ESAL rate = Total ESALs / 240 months
- Does not account for seasonal variations
- Assumes consistent traffic growth

Material Properties:
- Properties remain constant over pavement life
- Does not model aging or hardening effects
- AC modulus does not account for temperature variations
- Paris Law parameters (A, n) are time-invariant

Environmental Factors:
- Climate effects embedded in training data
- Does not explicitly model temperature, moisture
- Assumes typical Texas environmental conditions
- Does not account for drainage issues

Construction Quality:
- Assumes proper construction practices
- Does not model segregation, compaction issues
- Quality control is user's responsibility

7.4 Not Included

The tool does NOT predict:
- Rutting or permanent deformation
- Thermal cracking
- Block cracking
- Reflective cracking
- Raveling or surface distress
- Roughness or ride quality

The tool does NOT account for:
- Maintenance and rehabilitation
- Overlays or treatments applied during service life
- Subgrade failures or foundation issues
- Drainage problems
- Material variability within a project

7.5 Recommended Complementary Analyses

For comprehensive pavement design, also consider:
- Mechanistic-empirical pavement design (AASHTOWare)
- Layered elastic analysis
- Finite element modeling
- Life cycle cost analysis
- Pavement management system data
- Local experience and field performance

================================================================================
8. TROUBLESHOOTING
================================================================================

8.1 Common Issues

Issue: "Error loading models"
Solution:
- Ensure all .pkl files are in the same directory as app.py
- Check file names match exactly
- Verify files are not corrupted
- Re-download model files if necessary

Issue: Predictions seem unrealistic
Solution:
- Check for out-of-range input warnings
- Verify units are correct (inches vs feet, ksi vs psi)
- Review model confidence indicator
- Compare with similar designs
- Consider running validation script

Issue: App runs slowly
Solution:
- First prediction after startup may be slow (model loading)
- Subsequent predictions should be fast (<1 second)
- Check internet connection if using Streamlit Cloud
- Clear browser cache
- Restart application

Issue: Cannot access Design Comparison Tool
Solution:
- Must run a prediction in Design Input tab first
- Click "Calculate Prediction" to enable comparison
- Refresh page and try again

Issue: Warnings about out-of-range inputs
Solution:
- Review input values against valid ranges in documentation
- Verify data entry is correct
- Consider if design is truly unusual
- Proceed with caution if inputs are intentionally out of range
- Document rationale for unusual inputs

8.2 Validation

To verify app is working correctly:

1. Run the validation script (app_validation.py)
   - Mean Absolute Error should be ~2.69%
   - R² should be ~0.9881
   - 
If validation fails, check:
   - Model files are correct versions
   - Python packages are up to date
   - No modifications to app.py prediction logic

2. Test with known case:
   - AC Thickness: 5.5 in
   - Base Thickness: 16.0 in
   - AC Modulus: 865 ksi
   - Base Modulus: 37 psi
   - Subgrade Modulus: 12.5 psi
   - PG 64-22, Type B, 15% RAP
   - Medium traffic (2M ESALs)
   - 20 year design life
   
   Expected result: ~43% cracking (typical for median case)

8.3 Getting Help

For technical support:
- Review this documentation thoroughly
- Check GitHub repository for updates
- Contact tool developer: [Your email]

For TxDOT-specific questions:
- Consult local TxDOT pavement engineer
- Reference TxDOT Pavement Design Guide
- Contact TxDOT Materials and Pavements Division

================================================================================
9. CONTACT INFORMATION
================================================================================

Tool Developer:
Name: [Your Name]
Email: [Your Email]
Phone: [Your Phone]
Organization: [Your Organization]

TxDOT Contact:
[TxDOT Project Manager Name]
[TxDOT Email]
[TxDOT Phone]
Texas Department of Transportation
Materials and Pavements Division

Version History:
Version 1.0 - December 2024
- Initial release
- Three-model ensemble (XGBoost, LightGBM, Random Forest)
- Design comparison tool
- Sensitivity analysis
- Out-of-range warnings
- Model confidence indicator

Future Enhancements Under Consideration:
- PDF report export
- Additional distress types (rutting, thermal cracking)
- Integration with TxDOT databases
- Mobile app version
- Multi-language support

================================================================================
END OF DOCUMENTATION
================================================================================

For the latest version of this tool and documentation, visit:
[GitHub Repository URL]
[Streamlit Cloud URL]

This tool is provided for engineering analysis purposes. Results should be 
validated using sound engineering judgment and supplemented with additional 
analyses as appropriate for project-specific conditions.

© 2024 [Your Organization]. All rights reserved.
