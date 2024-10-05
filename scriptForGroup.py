from importlib import reload 
import aggregators as aggr_utils
reload(aggr_utils)
import aggregators as aggr_utils

# aggregation_strategies = ["BASE"]
# recommendations_number = 10
if __name__ == "__main__":
    print("This is test.py being executed directly.")

def generate_group_recommendations_for_a_group(test_df, recommendations_number,aggregation_strategies):
    # aggregation_strategies = "BASE"
    group_ratings = test_df
    agg = aggr_utils.AggregationStrategy.getAggregator(aggregation_strategies)
    print(f"Aggregator object: {agg}")
    if agg is None:
        print("Aggregator failed to initialize.")
        return None
    group_rec = agg.generate_group_recommendations_for_group(group_ratings, recommendations_number)
    
    
    return group_rec 


