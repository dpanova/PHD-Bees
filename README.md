# PHD-Bees

## Author: Denitsa Panova

## Useful Links
- [Documentation](https://dpanova.github.io/PHD-Bees/)
- [MIRO board for BeeClassification architecture](https://miro.com/welcomeonboard/ejdQbDJWVkhMSms2dksxTnhQc2NwQzRKSjlxQ0lpVWJ0R042WWVoWlkySjg1NERhNnhSTjAyajBNNThXaVhicnwzMDc0NDU3MzUxNzQ2ODAzMzY0fDI=?share_link_id=778703627354)
- [MIRO board for BeeLitReview architecture](https://miro.com/welcomeonboard/ZUhZN1M2NmFkbGUxeGhmSVJPMTZURzhjUzlXRFpmVktXMWpEeGJBYU5nZldVd2xJOE5DcFd6dXdYSW0ybnU3a3wzMDc0NDU3MzUxNzQ2ODAzMzY0fDI=?share_link_id=366597936435)
- [MIRO board for BeeData architecture](https://miro.com/welcomeonboard/UjNiQlZGM1M2V3ZMS2pFOXVYcEtVdG5PRTY5T0hIcENaUU5odmh5OGxBUGNmQzQwWTBuVEZ5REtYbGU0V3NMV3wzMDc0NDU3MzUxNzQ2ODAzMzY0fDI=?share_link_id=301390331377)
- [Weights & Biases project](https://wandb.ai/konstantin-preslavsky-university-of-shumen/hubert-base-ls960-project?nw=nwuserdenitsapanova)

## Main Files
- [lit_report.pdf](https://github.com/dpanova/PHD-Bees/blob/main/lit_report.pdf)
- BeeLitReview.py
- BeeData.py
- BeeClassification.py
- literature_report_automation.py 
- pipeline.py
- auxiliary_functions.py

## Overview
### BeeLitReview
The Python library BeeLitReview aims to provide researchers with an optimal path for conducting comprehensive literature reviews. Its end product is an automated report that facilitates the discovery of relevant information within available scientific resources. The presented method is designed to help users extract a maximum amount of diverse information, enabling them to quickly decide where to focus their attention. The proposed framework is suitable for so-called "shallow" research and can be used in the early stages of scientific investigation. Often, research begins with an abstract search in global information sources to explore the overall subject matter before moving on to more detailed searches. This method, created through machine engineering techniques, is specifically designed for this step to obtain an optimal path through scientific articles and papers for the purpose of conducting a literature review on a particular topic.

### BeeData
The BeeData library is designed to serve a crucial role in the data processing pipeline by preparing and preprocessing input data specifically for use with the BeeClassification library. This involves various steps such as data cleaning and feature extraction, ensuring that the data is in the optimal format and quality required for effective classification. By leveraging the BeeData library, users can streamline the initial data preparation phase, thereby enhancing the accuracy and efficiency of the subsequent classification tasks performed by the BeeClassification library.

### BeeClassification
The third library developed for this dissertation is BeeClassification. This library is designed to facilitate the testing of various machine learning models in an optimized, fast, and user-friendly manner. With BeeClassification, researchers and data scientists can efficiently experiment with different algorithms and approaches to identify the most effective models for their specific needs.

BeeClassification builds upon the foundation laid by the BeeData library. After BeeData prepares and preprocesses the input data, BeeClassification takes over to apply and evaluate different classification models. This seamless integration ensures a smooth transition from data preparation to model testing, making the overall workflow more efficient.

By providing a robust framework for model testing, BeeClassification helps users save time and effort, allowing them to focus on analyzing results and refining their models. It supports a variety of machine learning techniques and offers tools for performance evaluation, making it an essential component of the data analysis and research process.

