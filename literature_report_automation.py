"""
Run the following pipeline to generate the automated report
"""

from BeeLitReview import BeeLitReview
review = BeeLitReview()
# Uncomment this section if you want to query new results
# review.researchgate_scraper()
review.scraped_data_enhancement()
review.encode_with_transformers()
review.calculate_similarity()
review.calculate_TSP()
# Manual step

review.clustering_evaluation( cosine_threshold = 0.6
                        ,year=2023
                        ,type = ['Article', 'Conference Paper','Preprint','Patent','Thesis']
                        ,read = 10
                        ,citation = 0)


review.pdf_report_generate()


