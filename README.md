# Telco Churn

---

# Project Description


Customer churn, the annual percentage at which customers stop subscribing to a service or employees leave a job, is a crucial metric for businesses to track. In this context, we will analyze data from Telco to identify key causes of churn and suggest effective strategies to mitigate it. By reducing churn rate, businesses can retain valuable customers and reduce employee turnover, ultimately leading to improved financial performance and customer satisfaction. In the contex of this project we will be looking specificly at potential causality of customer churn and ways to minimize it. 



# Project Goal
--- 
- Discover some potential drivers for churn
- Delvelope machine learning  models for churn prediction
- Reccomend some key factors to reduce future churn



# Initial Thoughts
---
My inital thoughts is that churn will be mainly decided by higher monthly payments



# Planning
---
- ### Acuire data 
- ### Prep/clean the data
    - Drop columns that make the data noisy
    - Split data into train, validate, and test
- ### Explore the data
    - Discover potentil drivers for churn
    - Create hypothesis driver correlation
    - Preform Statistical Test on drivers
- ### Create Models for churn prediction
    - Use models on train and validate data
    - Measure Models effectiveness on train and validate
    - Select best performing model for test
- ### Conclusions   
    
# Data Dictionary 

| Feature | Values | Description                                  |
|-------------|:-----------:|----------------------------------------------|
| Gender    | (M,F)       | The customer's gender.        |
| Senior Citizen |    (Yes, No)    | Boolean of senior citizenship status |
| Partner | (Yes, No)   | Customer's partnership status                |
| Dependents    | (Yes,  No)       | Status of customer's dependents        |
| Tenure |   Integer    | Number of months customer has been with Telco |
| Phone service | (Yes, No)   | Customer's Phone service status            |
| Multiple Lines    | (Yes, No, No phone service)       | Does customer have multiple phone lines.        |
| Online Security |    (Yes, No, No internet service)  | Does Customer have online security services |
| Online Backup Service | (Yes, No, No internet service |Does customer have Online backup services|
| Device Protection    | (Yes, No, No internet service) | Does the customer have a device protection service |
| Tech Support |    (Yes, No, No internet service)    | Does customer have tech support service|
| Streaming TV| (Yes, No, No internet service)   | Does the customer stream Tv|
| Streaming Movies| (Yes, No, No internet service)   | Does the customer stream movies|
| Paperless Billing| (Yes, No)       | Does the customer use paperless billing|
| Monthly Charges|    Float number   | The customers monthly bill|
| Total Charges |  Float number   | The customers total charges to date     |
| Churn   | (Yes, No)       | Churn status of customer |
| Contract Type|   (month_to_month, one_year, two_yer)   | The customer's contract type |
| Internet service type | (Fiber, DSL, None)   | What internet service type customer has |
| Paymet Type    | (Electronic check, Mailed check, Credit card (automatic), Bank transfer (automatic)   | How the customer pays their bill |


# Steps to Reproduce 
- Clone repo
- Accqire data from SQL data base (must have credentials!)
- Use env file template (instructions inside)
- Run notebook

# Takeaways and Conclusions
- 27% of the data consist of churn
- Some drivers include but not limited to are:
    - contract type
    - senior citizen status
    - monthly charges
    - dependent status
    - internet services
    - payment type
- Gender and partner status is not a churn driver


# Recommendations 
- Consider looking into contract type. At face value having month to month looks like a huge driver for churn but futher data on customer return-rate could tell a different story
- Consider looking at monthly charges and determine if charge rate is competitive with the market