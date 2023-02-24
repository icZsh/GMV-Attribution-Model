# GMV-Attribution-Model

Gross merchandises value (GMV) is the total value of merchandise sold over a given period of time through a customer-to-customer (C2C) exchange site. It is a measure of the growth of the business or use of the site to sell merchandise owned by others.

Gross merchandise value (GMV) is often used to determine the health of an e-commerce site's business because its revenue will be a function of gross merchandise sold and fees charged. It is most useful as a comparative measure over time, such as current quarter value versus previous quarter value.

## Goal

During my stay at Tencent Class, our department launched a series of special summer sales and there was great variance in Gross merchandise value (GMV) performance across different time stages. My colleagues used to analyze GMV attribution factors manually, which could be inaccurate and time-consuming. Therefore, I took the initiative and worked on an automatic GMV attribution model. The model will run when it detects significant rise or drop in GMV and analyze for factors that cause the change.

## Methodology

* Logic Trees
* Atttibution Analysis
* Jension-Shannon
* Pareto Analysis
* Historical Data Remapping



## Steps to Build the model

1. Defines the name of the SQL table and SQL query to fetch data from the table.
2. Uses Apache Spark to create a temporary view of the table and execute the SQL query.
3. Converts the result of the query to a Pandas DataFrame and assigns column names to the DataFrame.
4. Maps the names of certain columns to human-readable names using dictionaries.
5. Cleans the data by replacing null values with -1 in some columns and replacing '\N' values with -1 in other columns.
6. Adds new columns to the DataFrame, including 'user_cnt' and 'lesson_weeks', based on certain conditions.
7. Normalizes some values in the 'lesson_level' column.
8. Creates a list of unique school subjects and lesson types in the DataFrame.
9. For each school subject and lesson type combination, creates dictionaries that map a camp start day to its order and vice versa.
10. Uses the dictionaries created in step 9 to calculate various metrics, including the number of users who signed up for a camp, the number of users who transferred from one camp to another, the number of users who completed a camp, and the revenue generated from a camp.
