import os 
import re
import ast
import webbrowser
import pandas as pd
import streamlit as st
from typing import List, Dict


# Extract wine data
ratings_data = pd.read_csv("dataset\\XWines_Slim_150K_ratings.csv", low_memory=False)
wine_data = pd.read_csv("dataset\\XWines_Slim_1K_wines.csv")
wine_data['Grapes'] = wine_data['Grapes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
wine_data['Harmonize'] = wine_data['Harmonize'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def extract_string(sentence):
    matches = re.findall(r'\*\*(.*?)\*\*', sentence)
    return " ".join(list((matches)))

def join_list_with_and(items):
        return ', '.join(items[:-1]) + ' and ' + items[-1] if len(items) > 1 else ''.join(items)


def generate_personalized_explanation(recommended_wine: pd.Series, user_ratings: pd.DataFrame, all_wines: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a personalized explanation for a wine recommendation.

    Args:
        recommended_wine (pd.Series): The recommended wine's features.
        user_ratings (pd.DataFrame): The user's rating history.
        all_wines (pd.DataFrame): DataFrame containing all wines and their features.

    Returns:
        Dict[str, str]: A dictionary of personalized explanations for different features.
    """
    liked_wines = all_wines[all_wines['WineID'].isin(user_ratings[user_ratings['Rating'] >= 4]['WineID'])].drop_duplicates(subset=['WineID'])
    
    explanations = {}
    
    if len(liked_wines) > 2:
        # Function to check if a feature is present in more than 80% of liked wines
        def is_frequent_feature(feature_name):
            feature_counts = liked_wines[feature_name].value_counts(normalize=True)
            return feature_counts[feature_counts >= 0.8].index.tolist()
        
        # Strategy for string features: Type, Elaborate, Body, Acidity, Country
        string_features = ['Type', 'Elaborate', 'Body', 'Acidity']
        recs = []
        for feature in string_features:
            frequent_values = is_frequent_feature(feature)
            if recommended_wine[feature] in frequent_values:
                recs.append(recommended_wine[feature].lower())
        if recs:
            explanations["string_features"] = f"You've consistently shown a strong preference for {join_list_with_and(recs)} wines, and this recommendation aligns perfectly with your taste."
        
        # Strategy for location features
        location_features = ['WineryName', 'RegionName', 'Country']
        recs_loc = []
        for feature in location_features:
            frequent_values = is_frequent_feature(feature)
            if recommended_wine[feature] in frequent_values:
                recs_loc.append(recommended_wine[feature].lower())

        if recs_loc:
            if "WineryName" in recs_loc and "RegionName" in recs_loc and "Country" in recs_loc:
                explanation = (f"You've shown a strong preference for wines from {recs_loc['WineryName']} winery in the {recs_loc['RegionName']} region of {recs_loc['Country']}.")
            elif "RegionName" in recs_loc and "Country" in recs_loc:
                explanation = (f"You've shown a strong preference for wines from the {recs_loc['RegionName']} region of {recs_loc['Country']}.")
            elif "Country" in recs_loc:
                explanation = f"You've shown a strong preference for wines from {recs_loc['Country']}."

            explanations["location_features"] = explanation
        
        # Strategy for ABV
        avg_abv = liked_wines['ABV'].mean()
        if abs(recommended_wine['ABV'] - avg_abv) <= 1:
            explanations['ABV'] = f"This wine's ABV ({recommended_wine['ABV']}%) is similar to your preferred average of {avg_abv:.1f}%."
        
        # Strategy for Grapes and Harmonize
        def process_list_feature(feature_name):
            user_items = [item for items in liked_wines[feature_name] for item in items]
            recommended_items = recommended_wine[feature_name]
            common_items = set(user_items) & set(recommended_items)
            if common_items:
                if feature_name == 'Grapes':
                    return f"This wine features {join_list_with_and(list(common_items))} grapes, which are also present among your favourite wines."
                elif feature_name == 'Harmonize':
                    common_items = [item.lower() for item in common_items]
                    d = "a dish" if len(common_items) == 1 else "dishes"
                    return f"{recommended_wine['WineName']} pairs beautifully with {join_list_with_and(list(common_items))}, {d} you've enjoyed alongside previous wines, making it an excellent choice for your next meal."

            return None
        
        for feature in ['Grapes', 'Harmonize']:
            explanation = process_list_feature(feature)
            if explanation:
                explanations[feature] = explanation
    
    else:
        explanations['general'] = "We don't have enough information about your preferences yet, but we think you might enjoy this wine based on its overall characteristics. Give it a try and let us know!"
    
    return explanations


### GROUP RECOMMENDATION EXPLANATIONS ###
# Function to explain the recommended wine for a group
def personalized_grupal(rec_type, rec_subtype, group):
    if rec_type == "Majority based": 
        if rec_subtype == "Each member votes for his/her **:violet[most preferred alternative]**.":
            subexplanation = f" by identifying their {extract_string(rec_subtype)}s"
        else: 
            subexplanation = f" by storing {extract_string(rec_subtype)} while applying a threshold of above 2.5 for each rating."
    
    elif rec_type == "Consensus based":
        if rec_subtype == "**:violet[Add all]** individual ratings.":
            subexplanation = f", calculating the sum of all ratings for each wine."
        else:
            subexplanation = f", calculating the product of all ratings for each wine."
    
    else: 
        if rec_subtype == "Consider the **:violet[opinion of the most respected person]** within the group.":
            subexplanation = f" considering the perspective of the most respected member."
        else:
            subexplanation = f" ensuring than {extract_string(rec_subtype)} "
            op = "maximum" if rec_subtype == "Ensure that the **:violet[majority is satisfied]**." else "minimum"
            
            subexplanation += f" by assuming that the group rating reflects the {op} of the individual ratings."
            
    return (f"This wine has been identified as the best choice for all {len(group)} group members. "
            f"Taking into account each user's previous ratings and{subexplanation}, this wine received the highest number of positive votes among them.")


### INDIVIDUAL RECOMMENDATION EXPLANATIONS ###
# Function to explain the recommended wine for a single user
def personalized_individual(rec_type_indiv):
    # Content-based
    if rec_type_indiv == "Similar to my **:violet[past liked items]**": 
        subexplanation = f"because it shares key features with wines you've previously rated highly"
    # CF item-item
    elif rec_type_indiv == '**:violet[Similar wines]** to what I have enjoyed':
        subexplanation = f"because it is similar to other wines you have rated highly"
    # CF user-user
    else:
        subexplanation = f"because users with similar tastes to yours have rated it highly"
    
    return (f"We also recommend this wine {subexplanation}.") 


def nonpersonalized(order):
    subexplanation = "by number of ratings" if order ==  "numRatings" else ("by rating values" if order == "Rating" else "randomly")
    return (f"This wine is the top choice in our ranking, ordered **{subexplanation}**. "
            f"It could be an exciting new discovery that surprises you if you give it a try!🍷")


def get_personalized_explanation(recommended_wine, current_user):
    user_ratings = ratings_data[ratings_data['UserID'] == current_user]
    explanations = generate_personalized_explanation(recommended_wine, user_ratings, wine_data)
    pers_expl = ""
    for feature, explanation in explanations.items():
        if explanation: # print non-empty explanations
            pers_expl += explanation + " "
    return pers_expl.strip(), explanations


### EXPLANATIONS ###
# General function for the explanations of each recommendation performed
def explanation(rec, rec_type, rec_subtype, group, current_user, sorted_df, order):
    st.header("**Why this recommendation?**")
    wine_info = wine_data[wine_data['WineID'] == sorted_df['WineID'].iloc[0]].iloc[0]
    
    if rec == 'try_new':
        explanation = nonpersonalized(order)
    elif rec == 'recommend_individual':
        explanation = personalized_individual(rec_type)
    else: 
        explanation = personalized_grupal(rec_type, rec_subtype, group)

    extended_explanation, pers_explanations = get_personalized_explanation(wine_info, current_user) if rec != 'try_new' else "", ""
    
    grapes = join_list_with_and(wine_info["Grapes"])
    harmonize = [item.lower() for item in wine_info['Harmonize'] if item not in pers_explanations['Harmonize']] if extended_explanation else [item.lower() for item in wine_info['Harmonize']]
    extra_harmonize = ""
    if harmonize: 
        harmonize = join_list_with_and(harmonize)
        extra_harmonize = f'We’re confident you’ll enjoy its balance of flavors, which also complement delightfully **{harmonize}**, an ideal companion to elevate any meal experience.' 
        
    
    # Reset 'clicked' state every time the function is called
    if 'clicked' not in st.session_state or st.session_state.reset_clicked:
        st.session_state.clicked = "No"
        st.session_state.reset_clicked = False  # To prevent further resets within the same function call

    st.markdown(
        f'''
        Based on the provided features, we highly recommend **:violet[{sorted_df.iloc[0]["WineName"]}]**. This is a **{wine_info["Body"].lower()}**, rich **{wine_info["Type"].lower()} wine** crafted from a blend of **{grapes} grapes**. These distinctive varities not only add depth but also lend the wine it notably **{wine_info['Acidity'].lower()} acidity**.
        
        {extended_explanation}

        {explanation} {extra_harmonize}
        
        ***{wine_info["WineryName"]}***, located in **{wine_info["RegionName"]}**, **{wine_info["Country"]}**, produces this wine with expertise and passion. For more information about the winery and their offerings, feel free to visit their website (link below).
        ''')
    
    winery_website = st.button("Visit Winery Website")
    if winery_website:
        webbrowser.open_new_tab(wine_info['Website'])
        st.session_state.clicked = "Yes"
    
    st.markdown(
        f'''  
        We encourage you to give it a try and share your thoughts with us! Your feedback is invaluable, and we’d love to hear what you think!
        
        Please take a moment to rate your experience with **:violet[{wine_info['WineName']}]** and our recommendation service. Your feedback helps us tailor our future suggestions to suit your preferences even better.

        Thank you for choosing us, and we look forward to helping you discover your next favorite wine!
        ''')
    
    new_folder_path = os.path.join(os.getcwd(), "feedback")
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    feedback_explanation(rec, rec_type, rec_subtype, group, current_user, wine_info, st.session_state.clicked)



# Feedback CSV
FEEDBACK_COLUMNS = ['UserID', 'WineID', 'Satisfaction', 'Effectiveness', 'Fairness', 'Persuasiveness']
def save_feedback(feedback_data, csv_path):
    if not os.path.exists(csv_path):
        pd.DataFrame([feedback_data]).to_csv(csv_path, sep=';', mode='w', header=True, index=False)
    else: # Append feedback row 
        pd.DataFrame([feedback_data]).to_csv(csv_path, sep=';', mode='a', header=False, index=False)

### EXPLANATIONS FEEDBACK ###
# Function to store the user's feedback on the given recommendation explanation
def feedback_explanation(rec, rec_type, rec_subtype, group, current_user, wine_info, clicked):
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    stars_mapping = ["1", "2", "3", "4", "5"]
    people = "the group" if rec == 'recommend_group' else "your"
    
    if rec == 'recommend_group':
        group_key = "_".join(str(group))
        current_fairness_dict_key = f"fairness_dict_{group_key}"
        if current_fairness_dict_key not in st.session_state:
            st.session_state[current_fairness_dict_key] = {member: None for member in group}
        current_fairness_dict = st.session_state[current_fairness_dict_key]
    
    st.session_state['feedback_submitted'] = False
    
    st.write("")
    with st.container(border=True):
        cols = st.columns([2,2,2,3])

        with cols[0]:
            st.markdown("###### Did you find this explanation useful?")
            ex_satisfaction = st.feedback("thumbs", key="satisfaction")
            if ex_satisfaction is not None:
                st.markdown(f"You selected: {sentiment_mapping[ex_satisfaction]}. Thanks!")

        with cols[1]:
            st.markdown(f"###### Does this explanation meet {people} requirements?")
            ex_effectiveness = st.feedback("thumbs", key="effectiveness")
            if ex_effectiveness is not None:
                st.markdown(f"You selected: {sentiment_mapping[ex_effectiveness]}. Thanks!")

        with cols[2]:
            st.markdown("###### After the explanation, are you willing to give this wine a chance?")
            ex_persuasiveness = st.feedback("thumbs", key="persuasiveness")
            if ex_persuasiveness is not None:
                st.markdown(f"You selected: {sentiment_mapping[ex_persuasiveness]}. Thanks!")

        with cols[3]:
            if rec == 'recommend_group':
                st.markdown(f"###### Rate your individual satisfaction with the recommendation")
                c1, c12, c2 = st.columns([2,0.2,2.5], vertical_alignment="center")
                member = c1.selectbox("**Who are you?**", options=group)

                if current_fairness_dict[member] is not None: # If member has already voted, disable the feedback widget
                    c2.markdown(f"**{member} has already voted** and selected {current_fairness_dict[member]} star(s).")
                else: # Allow voting if the member hasn't voted yet
                    ex_fairness = c2.feedback("stars", key=f"fairness_{member}", disabled=False)  # Feedback widget is enabled
                    if ex_fairness is not None:
                        current_fairness_dict[member] = stars_mapping[ex_fairness]
                        c2.markdown(f"You selected: {stars_mapping[ex_fairness]} star(s). Thanks {member}!")
                
                st.session_state['current_fairness_dict'] = current_fairness_dict # save updated dictionary 
        
        
        expl_feed = st.button(label="Submit", key="expl")

    # Save feedback when 'submit' button has been selected
    if expl_feed: # all([ex_satisfaction is not None, ex_effectiveness is not None, ex_fairness is not None, ex_persuasiveness is not None]):
        feedback_data = {
            'UserID': "" if rec == "try_new" else (group if rec == 'recommend_group' else current_user),
            'WineID': wine_info['WineID'],
            'RecSys': (rec).split("_")[-1],
            'RecSysType': (rec_type).lower(),
            'RecSysSubtype': (extract_string(rec_subtype)).split("[")[-1].removesuffix("]"),
            'Satisfaction': (sentiment_mapping[ex_satisfaction]).split("/")[-1].removesuffix(":"),
            'Effectiveness': (sentiment_mapping[ex_effectiveness]).split("/")[-1].removesuffix(":"),
            'Fairness': current_fairness_dict if rec == 'recommend_group' else "", 
            'Persuasiveness': (sentiment_mapping[ex_persuasiveness]).split("/")[-1].removesuffix(":"),
            'Persuasiveness_link': clicked
        }
        save_feedback(feedback_data, "feedback\\explanation_feedback.csv")
        st.session_state['feedback_submitted'] = True
        if st.session_state['feedback_submitted']:
            st.write("")
            st.success("Thanks for your feedback on the explanation!")

            # Set flag to allow reset in next call
            st.session_state.reset_clicked = True
            return  # Exit the function so no further feedback options are displayed
    