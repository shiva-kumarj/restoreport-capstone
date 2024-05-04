# yelp-customer-satisfaction-dashboard

This project aims to create a data analytics tool that can be used by business owners to understand their customers. Customer reviews can be used to create a better impact on businesses. This tool demonstrates the kind of analysis that can be performed on customer reviews. I am focussing on 101 businesses due to hardware limitations, but this can be very easily scaled to process much higher volume.

## Some features of the project

1. Using NLP libraries like VaderSentiment, BERT Transformers customer reviews are boiled down to important entities, and the sentiment of each entity is captured.
2. **Entities represent the specific aspects that customers discuss in their reviews**, such as food, order, service, time, staff, and ambiance.
3. These entities will be **clustered** into similar buckets. (balance will have to be maintained between too many buckets - too specific, or too few - loss of information.
4. Footfall analysis, comparing the negative effect of one entity over another on star rating, performance attribution, and many more kinds of analysis will be performed and presented in a drill-downable dashboard.
5. Report generation and broadcasting it to relevant personnel will also be possible.

## Updates

1. Added functionality to visualize average rating in various levels of analysis. Yearly, monthly, daily, hourly.


## Notes

Steps and commands to setup the environment for this project

### Create a conda environment for the project
conda create -n capstone_env python=3.12
conda activate capstone_env

### How to work with jupyter notebook inside the conda environment
conda install ipykernel
python -m ipykernel install --user --name=capstone_env
jupyter notebook

create a new jupyter notebook by selecting "capstone_env" as the kernel
