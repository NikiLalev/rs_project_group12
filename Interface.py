import re
import os
import ast
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_toggle as sts

# Import Recommerder Systems
from Recommender import Recommender


# Extract wine data
ratings_data = pd.read_csv("dataset\\XWines_Slim_150K_ratings.csv", low_memory=False)
wine_data = pd.read_csv("dataset\\XWines_Slim_1K_wines.csv")
wine_data['Grapes'] = wine_data['Grapes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
wine_data['Harmonize'] = wine_data['Harmonize'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

full = pd.merge(ratings_data, wine_data, on ='WineID')

full['numRatings'] = full.groupby('WineID')['Rating'].transform('count')
wine_data = wine_data.merge(full[['WineID', 'numRatings']], on='WineID', how='left')

def extract_string(sentence):
    matches = re.findall(r'\*\*(.*?)\*\*', sentence)
    return " ".join(list((matches)))


def display_wines(sorted_df, num_recs, nonpers):
    # Check if the dataframe is empty
    if sorted_df.empty:
        st.write("No wine(s) found matching your preferences. Please try with other filters.")
        return

    # Display individual wine information
    def display_wine(col, wine_data, nonpers1, highlighted=False):
        if highlighted:
            col.write("")
            with col.container(border=True):
                st.write("**Our recommendation:**")
                st.header(f"🥇 **:violet[{wine_data['name']}]**")
                if nonpers1:
                    st.write(f'**Rating**: {wine_data["rating"]} ({wine_data["num_rating"]})')
                st.write(f'**Winery**: {wine_data["winery"]}')
                st.write(f'**Region**: {wine_data["region"]}, {wine_data["country"]}')
                if pd.notna(wine_data["web"]) and wine_data["web"]:
                        st.link_button("More info", wine_data["web"])
        else:
            col.subheader(f":violet[{wine_data['name']}]")
        
            col.write(f'**Rating**: {wine_data["rating"]} ({wine_data["num_rating"]})')
            col.write(f'**Winery**: {wine_data["winery"]}')
            col.write(f'**Region**: {wine_data["region"]}, {wine_data["country"]}')
            if pd.notna(wine_data["web"]) and wine_data["web"]:
                    col.link_button("More info", wine_data["web"])

    # Display the first recommendation prominently
    if nonpers:
        first_wine_data = {
            "name": sorted_df.iloc[0]['WineName'],
            "rating": sorted_df.iloc[0]['Rating'].round(2),
            "num_rating": sorted_df.iloc[0]['numRatings'],
            "winery": sorted_df.iloc[0]['WineryName'],
            "region": sorted_df.iloc[0]['RegionName'],
            "country": sorted_df.iloc[0]['Country'],
            "web": sorted_df.iloc[0]['Website'],
        }
        nonpers1 = True
    else:
        first_wine_data = {
            "name": sorted_df.iloc[0]['WineName'],
            "winery": sorted_df.iloc[0]['WineryName'],
            "region": sorted_df.iloc[0]['RegionName'],
            "country": sorted_df.iloc[0]['Country'],
            "web": sorted_df.iloc[0]['Website'],
        }
        nonpers1 = False

    highlight_col = st.columns(1)[0]
    display_wine(highlight_col, first_wine_data, nonpers1, highlighted=True)

    # Remaining wines
    remaining_wines = sorted_df.iloc[1:]
    st.markdown("###")

    num_cols = 5 
    for idx in range(len(remaining_wines)):
        if idx % num_cols == 0:  # New row after `num_cols`
            cols = st.columns(num_cols)

        if nonpers:
            wine_data = {
                "name": remaining_wines.iloc[idx]['WineName'],
                "rating": remaining_wines.iloc[idx]['Rating'].round(2),
                "num_rating": remaining_wines.iloc[idx]['numRatings'],
                "winery": remaining_wines.iloc[idx]['WineryName'],
                "region": remaining_wines.iloc[idx]['RegionName'],
                "country": remaining_wines.iloc[idx]['Country'],
                "web": remaining_wines.iloc[idx]['Website'],
            }
        else:
            wine_data = {
                "name": remaining_wines.iloc[idx]['WineName'],
                "winery": remaining_wines.iloc[idx]['WineryName'],
                "region": remaining_wines.iloc[idx]['RegionName'],
                "country": remaining_wines.iloc[idx]['Country'],
                "web": remaining_wines.iloc[idx]['Website'],
            }
        
        display_wine(cols[idx % num_cols], wine_data, nonpers1)
    
    if len(sorted_df) < num_recs:
        st.markdown("#####")
        st.write(f"Sorry, there are not {num_recs} wines with your preferences, just the {len(sorted_df)} provided. You can try with another combination.")
    return


### NON-PERSONALIZED RECOMMENDATION ###
# Function to recommend a wine based on user preferences
def recommend_wine_filtered(full, selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs, order='Rating'):
    # Filter
    filtered_df = full[
        (full['Type'].isin(selected_type)) &
        (full['Body'].isin(selected_body)) &
        (full['Acidity'].isin(selected_acidity)) &
        (full['Country'].isin(selected_country)) &
        (full['RegionName'].isin(selected_region)) &
        (full['Grapes'].apply(lambda x: any(grape in selected_grapes for grape in x))) &
        (full['ABV'].between(selected_ABV[0], selected_ABV[1])) &
        (full['Elaborate'].isin(selected_elaborate)) &
        (full['Harmonize'].apply(lambda x: any(harm in selected_harmonize for harm in x)))
    ]

    # Filter by minimum number of ratings
    sorted_df = filtered_df[filtered_df['numRatings'] >= min_ratings].drop_duplicates(subset=['WineID'])

    # Random ordering
    if order == 'Random':
        sorted_df = sorted_df.sample(frac=1, random_state=None)  # Shuffle rows randomly

    else:
        # Sort based on the selected order
        sorted_df = sorted_df.sort_values(by=order, ascending=False)
    
    # Return the top num_recs recommendations
    return sorted_df.head(num_recs)



### INDIVIDUAL RECOMMENDATION ###
# Function to recommend wine for a single user
def recommend_wine_for_user(individual_recommender, current_user, rec_type_indiv, rec_subtype_indiv, num_recs):

    # Collaborative Filtering
    if rec_type_indiv == "Users similar to me":
        if rec_subtype_indiv == "Based on **:violet[user interaction]**.": 
            individual_recommender.load_model("user-user")
        else: # recs_indiv == "Based on **:violet[item interaction]**.""
            individual_recommender.load_model("item-item")
        
        return individual_recommender.recommend(user_id=current_user, n=num_recs)

    else:
        st.write("NOTHING YET")
        return 

### GROUP RECOMMENDATION ###
# Function to recommend wine for a group
def recommend_wine_for_group(rec_type, rec_subtype, recs, group, threshold, impo_person, selected_type, selected_body, selected_acidity, selected_country, min_ratings, num_recs):
    filtered_df = full[
        (full['UserID'].isin(group)) & (full['Rating'] >= threshold) & (full['Type'].isin(selected_type)) & (full['Body'].isin(selected_body)) & 
        (full['Acidity'].isin(selected_acidity)) & (full['Country'].isin(selected_country))
    ]

    filtered_df['numRatings'] = filtered_df.groupby('WineID')['Rating'].transform('count')
    sorted_df = filtered_df[filtered_df['numRatings'] >= min_ratings]
   
    if rec_type == "All equal":
        if rec_subtype == recs[0]:
            copy_sorted_df = sorted_df.copy()
            eliminated_wines = []
            while not copy_sorted_df.empty and len(eliminated_wines) < num_recs:
                # For each user, find the wine with the maximum rating
                user_votes = copy_sorted_df.loc[copy_sorted_df.groupby('UserID')['Rating'].idxmax()]

                # Count the number of votes for each wine
                vote_counts = user_votes['WineID'].value_counts()

                # Identify the wine with the most votes
                winner = vote_counts.idxmax()
                eliminated_wines.append(winner)
                copy_sorted_df = copy_sorted_df[copy_sorted_df['WineID'] != winner]

            return sorted_df[sorted_df['WineID'].isin(eliminated_wines)].drop_duplicates(subset=['WineID']).set_index('WineID').loc[eliminated_wines].reset_index()

        else:
            return sorted_df[sorted_df['Rating'] >= 2.5].drop_duplicates(subset=['WineID']).sort_values(by=['numRatings'], ascending=False).head(num_recs)    
    
    elif rec_type == "Group preferences":
        aggregation = 'prod' if rec_subtype == recs[2] else 'mean'
        
        grouped_df = sorted_df.groupby('WineID').agg(Rating=('Rating', aggregation)).reset_index()
        sorted_df = sorted_df.drop(columns=['Rating']).merge(grouped_df, on='WineID')
        # Sort by average rating in decreasing order
        sorted_df = sorted_df.sort_values(by='Rating', ascending=False)
        return sorted_df.filter(items=['WineName', 'Rating', 'numRatings', 'WineryName', 'RegionName', 'Country', 'Website', 'WineID']).drop_duplicates().head(num_recs)    

    else: # rec_type == "Given criteria"
        if rec_subtype == recs[2]:
            dictator_df = sorted_df[sorted_df['UserID'] == impo_person]
            return dictator_df.sort_values(by='Rating', ascending=False).drop_duplicates().head(num_recs)
        else: 
            aggregation = 'min' if rec_subtype == recs[0] else 'max'
            grouped_df = sorted_df.groupby('WineID').agg(Rating=('Rating', aggregation)).reset_index()
            sorted_df = sorted_df.drop(columns=['Rating']).merge(grouped_df, on='WineID')
            sorted_df = sorted_df.sort_values(by='Rating', ascending=False)
            return sorted_df.filter(items=['WineName', 'Rating', 'numRatings', 'WineryName', 'RegionName', 'Country', 'Website', 'WineID']).drop_duplicates().head(num_recs)    


### GROUP RECOMMENDATION EXPLANATIONS ###
# Function to explain the recommended wine for a group
def personalized_grupal(rec_type, rec_subtype, group, threshold):
    if rec_type == "All equal": 
        if rec_subtype == "Each member vote for his/her **:violet[most preferred alternative]**.":
            subexplanation = f" founding out their {extract_string(rec_subtype)}s"
        else: 
            subexplanation = f" storing {extract_string(rec_subtype)}, but assuming a threshold of above 2.5 in each rating"
    
    elif rec_type == "Group preferences":
        if rec_subtype == "**:violet[Average all]** individual ratings.":
            subexplanation = f", for each wine, averaging all its ratings"
        elif rec_subtype == "**:violet[Average]** individual ratings **:violet[higher than a threshold]**.":
            subexplanation = f", for each wine, averaging all its ratings which were higher than {threshold}"
        else:
            subexplanation = f", for each wine, multiplying all its ratings"
    
    else: 
        if rec_subtype == "Consider the **:violet[opinion of the most respected person]** within the group.":
            subexplanation = f""
        else:
            subexplanation = f" ensuring than {extract_string(rec_subtype)} "
            op = "maximum" if rec_subtype == "Ensure that the **:violet[majority is satisfied]**." else "minimum"
            
            subexplanation += f" by assuming than the group rating is the {op} of the individual ratings"
            
    return (f"This wine has been found as the best suggestion for all the {len(group)} members of the group. "
            f"Taking into account each user's past ratings and{subexplanation}, this one appeared as the wine with the greatest amount of positive votes among them.")

### INDIVIDUAL RECOMMENDATION EXPLANATIONS ###
# Function to explain the recommended wine for a single user
def personalized_individual(rec_type_indiv, rec_subtype_indiv):
    # Content-based
    if rec_type_indiv == "Similar to my past liked items": 
        subexplanation = f"it shares key features with wines you've previously rated highly."
    
    else: 
        # CF item-item
        if rec_subtype_indiv == 'Based on **:violet[item interaction]**.':
            subexplanation = f"users who liked wines you previously enjoyed, also rated this wine highly."
        else:
        # CF user-user
            subexplanation = f"users who rated wines you rated before similarly, also rated this wine highly."

    return (f"We recommend this wine because {subexplanation}") 

def nonpersonalized():
    return (f"It could be an exciting new discovery that surprises you if you give it a shot!")


# Feedback CSV
FEEDBACK_COLUMNS = ['UserID', 'WineID', 'Satisfaction', 'Effectiveness', 'Fairness', 'Persuasiveness']
def save_feedback(feedback_data, csv_path):
    if not os.path.exists(csv_path):
        pd.DataFrame([feedback_data]).to_csv(csv_path, sep=';', mode='w', header=True, index=False)
    else: # Append feedback row 
        pd.DataFrame([feedback_data]).to_csv(csv_path, sep=';', mode='a', header=False, index=False)

### EXPLANATIONS FEEDBACK ###
# Function to store the user's feedback on the given recommendation explanation
def feedback_explanation(rec, rec_type, rec_subtype, group, wine_info):
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
        
        
        expl_feed = st.button(label="Submit")

    # Save feedback when 'submit' button has been selected
    if expl_feed: # all([ex_satisfaction is not None, ex_effectiveness is not None, ex_fairness is not None, ex_persuasiveness is not None]):
        feedback_data = {
            'UserID': "" if rec == "try_new" else (group if rec == 'recommend_group' else wine_info['UserID']),
            'WineID': wine_info['WineID'],
            'RecSys': (rec).split("_")[-1],
            'RecSysType': (rec_type).lower(),
            'RecSysSubtype': (extract_string(rec_subtype)).split("[")[-1].removesuffix("]"),
            'Satisfaction': (sentiment_mapping[ex_satisfaction]).split("/")[-1].removesuffix(":"),
            'Effectiveness': (sentiment_mapping[ex_effectiveness]).split("/")[-1].removesuffix(":"),
            'Fairness': "" if rec == "try_new" else (current_fairness_dict if rec == 'recommend_group' else sentiment_mapping[ex_fairness]), 
            'Persuasiveness': (sentiment_mapping[ex_persuasiveness]).split("/")[-1].removesuffix(":")
        }
        save_feedback(feedback_data, "feedback\\explanation_feedback.csv")
        st.session_state['feedback_submitted'] = True
        if st.session_state['feedback_submitted']:
            st.write("")
            st.markdown("##### Thanks for your feedback on the explanation!")
            return  # Exit the function so no further feedback options are displayed
    

### EXPLANATIONS ###
# General function for the explanations of each recommendation performed
def explanation(rec, rec_type, rec_subtype, group, threshold, sorted_df):
    st.header("**Why this recommendation?**")
    wine_info = wine_data[wine_data['WineID'] == sorted_df.iloc[0]['WineID']].iloc[0]
    
    def join_list_with_and(items):
        return ', '.join(items[:-1]) + ' and ' + items[-1] if len(items) > 1 else ''.join(items)

    grapes = join_list_with_and(wine_info["Grapes"])
    
    if rec == 'try_new':
        explanation = nonpersonalized()
    elif rec == 'recommend_individual':
        explanation = personalized_individual(rec_type, rec_subtype)
    else: 
        explanation = personalized_grupal(rec_type, rec_subtype, group, threshold)
    
    st.markdown(
        f'''
        The recommendation according the features provided is **:violet[{sorted_df.iloc[0]["WineName"]}]**'s wine.

        It is a {wine_info["Body"].lower()} {wine_info["Type"].lower()} wine done with {grapes} grapes which provide the {wine_info['Acidity'].lower()} acidity.
        
        {explanation}
        
        It is produced by *{wine_info["WineryName"]}* winery, located in {wine_info["RegionName"]}, {wine_info["Country"]} (access their [website]({wine_info['Website']}) for further information).
        
        And, please, do not forget to rate your experience with the service! See you soon.
        ''')
    
    new_folder_path = os.path.join(os.getcwd(), "feedback")
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    feedback_explanation(rec, rec_type, rec_subtype, group, wine_info)
    

### STORE USER FEEDBACK ###
def feedback(nonpers = False):
    st.write("")
    st.divider()
    st.write("")
    st.subheader("We'd love to hear your thoughts on our recommendations!")
    with st.container(border=True):
        cols = st.columns([3,3,3])
        sentiment_mapping = ["one", "two", "three", "four", "five"]

        with cols[0]:
            st.markdown("###### How satisfied are you with this wine recommendation?")
            selected = st.feedback("stars", key="Satisfaction1")
            if selected is not None:
                st.markdown(f"You selected {sentiment_mapping[selected]} star(s). Thank you so much for your feedback!")

        with cols[1]:
            st.markdown(f"###### Did the wine match the flavor profile you were looking for?")
            ex_enjoyment = st.feedback("thumbs", key="Enjoyment")
            if ex_enjoyment is not None:
                st.markdown(f"You selected: {sentiment_mapping[ex_enjoyment]}. Thanks!")

        with cols[2]:
            st.markdown(f"###### Would you choose this wine for your next dinner or special occasion?")
            ex_persuasiveness2 = st.feedback("thumbs", key="Persuasiveness2")
            if ex_persuasiveness2 is not None:
                st.markdown(f"You selected: {sentiment_mapping[ex_2]}. Thanks!")

        if nonpers:
            for i in [0,1,2]:
                cols[i].markdown("###")
            with cols[0]:
                st.markdown("###### How unique was this wine recommendation compared to others you’ve tried?")
                nonpers_selected = st.feedback("stars", key="nonpers_unique")
                if nonpers_selected is not None:
                    st.markdown(f"You selected {sentiment_mapping[nonpers_selected]} star(s). Thank you so much for your feedback!")
            with cols[1]:
                st.markdown(f"###### Was this a wine you wouldn’t have found on your own?")
                ex_discovery = st.feedback("thumbs", key="discovery")
                if ex_discovery is not None:
                   st.markdown(f"You selected: {sentiment_mapping[ex_discovery]}. Thanks!")
            with cols[2]:
                st.markdown(f"###### Did this wine surprise you in a good way?")
                ex_surprise = st.feedback("thumbs", key="surprise")
                if ex_surprise is not None:
                   st.markdown(f"You selected: {sentiment_mapping[ex_surprise]}. Thanks!")


    #    expl_feed1 = st.button(label="Submit")

    # Save feedback when 'submit' button has been selected
    # if expl_feed1: # all([ex_satisfaction is not None, ex_effectiveness is not None, ex_fairness is not None, ex_persuasiveness is not None]):
    #     recommendation_feedback_data = {
    #         'UserID': "" if rec == "try_new" else (group if rec == 'recommend_group' else wine_info['UserID']),
    #         'WineID': wine_info['WineID'],
    #         'RecSys': (rec).split("_")[-1],
    #         'RecSysType': (rec_type).lower(),
    #         'RecSysSubtype': (extract_string(rec_subtype)).split("[")[-1].removesuffix("]"),

    #         'Satisfaction1': (sentiment_mapping[selected]).split("/")[-1].removesuffix(":"),
    #         'Enjoyment': (sentiment_mapping[ex_enjoyment]).split("/")[-1].removesuffix(":"),
    #         'Persuasiveness2': (sentiment_mapping[ex_persuasiveness2]).split("/")[-1].removesuffix(":"),
    #         if nonpers:
    #             'nonpers_unique': (sentiment_mapping[nonpers_selected]).split("/")[-1].removesuffix(":"),
    #             'discovery': (sentiment_mapping[ex_discovery]).split("/")[-1].removesuffix(":"),
    #             'surprise': (sentiment_mapping[ex_surprise]).split("/")[-1].removesuffix(":")
    #     }
    #     save_feedback(recommendation_feedback_data, "feedback\\recommendation_feedback.csv")
    #     st.session_state['recommendation_feedback_submitted'] = True
    #     if st.session_state['recommendation_feedback_submitted']:
    #         st.write("")
    #         st.markdown("##### Thanks for your feedback on the recommendation!")
    #         return  # Exit the function so no further feedback options are displayed    
    
    


### FILTERING OPTIONS ###
# Function to allow the user to filter out some wine features
def options(num_cols, num_group, num_recom):
    cols = st.columns((num_cols))

    if num_cols != 2:
        selected_type = cols[0].multiselect('Wine type(s)', sorted(list(wine_data['Type'].unique())), default=None)
        selected_body = cols[1].multiselect('Wine body(s)', sorted(list(wine_data['Body'].unique())), default=None)
        selected_acidity = cols[2].multiselect('Acidity level(s)', sorted(list(wine_data['Acidity'].unique())), default=None)
        selected_country = cols[3].multiselect('Country(ies) of production', sorted(list(wine_data['Country'].unique())), default=None)
        if selected_country:
            selected_region = cols[4].multiselect('Region(s) of production', sorted(list(wine_data[wine_data['Country'].isin(selected_country)]['RegionName'].unique())), default=None)
        
        selected_grapes = cols[0].multiselect('Grape(s) type', sorted(set([item for sublist in wine_data['Grapes'].dropna() for item in sublist])), default=None)
        selected_ABV = cols[1].slider('Wine ABV', min(wine_data['ABV'].unique()), max(wine_data['ABV'].unique()), (min(wine_data['ABV'].unique()), max(wine_data['ABV'].unique())))
        selected_elaborate = cols[2].multiselect('Type(s) of elaboration', sorted(list(wine_data['Elaborate'].unique())), default=None)
        selected_harmonize = cols[3].multiselect('Harmonize(s)', sorted(set([item for sublist in wine_data['Harmonize'].dropna() for item in sublist])), default=None)
        min_ratings = cols[4].slider(f"Minimun number of ratings", 1, num_group, value = 1, step = 1)

    else: # cols == 0
        selected_type = cols[0].multiselect('Select wine type(s)', sorted(list(wine_data['Type'].unique())), default=None)
        selected_body = cols[1].multiselect('Select wine body(s)', sorted(list(wine_data['Body'].unique())), default=None)
        cols[0].write("")
        cols[1].write("")

        selected_acidity = cols[0].multiselect('Select acidity level(s)', sorted(list(wine_data['Acidity'].unique())), default=None)
        selected_elaborate = cols[1].multiselect('Select type(s) of elaboration', sorted(list(wine_data['Elaborate'].unique())), default=None)
        cols[0].write("")
        cols[1].markdown("####")

        cols[0].markdown("###### Alcohol")
        selected_grapes = cols[0].multiselect('Select grape(s) type', sorted(set([item for sublist in wine_data['Grapes'].dropna() for item in sublist])), default=None)
        selected_ABV = cols[1].slider('Select wine ABV', min(wine_data['ABV'].unique()), max(wine_data['ABV'].unique()), (min(wine_data['ABV'].unique()), max(wine_data['ABV'].unique())))
        cols[0].write("")
        cols[1].markdown("######")

        cols[0].markdown("###### Location")
        selected_country = cols[0].multiselect('Select country(ies) of production', sorted(list(wine_data['Country'].unique())), default=None)
        if selected_country:
            selected_region = cols[1].multiselect('Select region(s) of production', sorted(list(wine_data[wine_data['Country'].isin(selected_country)]['RegionName'].unique())), default=None)
        cols[0].write("")
        cols[1].markdown("######")

        cols[0].markdown("###### In combination with...")
        selected_harmonize = cols[0].multiselect('Select harmonize(s)', sorted(set([item for sublist in wine_data['Harmonize'].dropna() for item in sublist])), default=None)

        st.markdown("######")    
        num_recom = st.slider(f"Number of recommendations", 1, 20, value = 1, step = 1)
        min_ratings = st.slider(f"Minimun number of ratings", 1, 500, value = 1, step = 1)

    # Default values if nothing is selected
    selected_type = selected_type or wine_data['Type'].unique()
    selected_body = selected_body or wine_data['Body'].unique()
    selected_acidity = selected_acidity or wine_data['Acidity'].unique()
    selected_region = (selected_region or wine_data[wine_data['Country'].isin(selected_country)]['RegionName'].unique()) if selected_country else wine_data['RegionName'].unique()
    selected_country = selected_country or wine_data['Country'].unique()
    selected_elaborate = selected_elaborate or wine_data['Elaborate'].unique()
    selected_grapes = selected_grapes or set([item for sublist in wine_data['Grapes'] for item in sublist])
    selected_ABV = selected_ABV or wine_data['ABV'].unique()
    selected_harmonize = selected_harmonize or set([item for sublist in wine_data['Harmonize'] for item in sublist])

    return selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recom


### INDIVIDUAL CF OPTIONS ###
# Function to allow the user to specify some options regarding the CF recommendations
def generic_options_indiv(): 
    recs_indiv = ['Based on **:violet[item interaction]**.', 'Based on **:violet[user interaction]**.']
    st.write("")
    cols = st.columns((3))
    rec_type_indiv = cols[0].radio("How do you wanna deal with the recommendations?", recs_indiv)

    return rec_type_indiv


### GROUP OPTIONS ###
# Function to allow the user to specify some options regarding the group recommendations
def generic_options(recs):
    cols = st.columns((3))
    rec_type = cols[0].radio("How do you wanna deal with the recommendations?", recs)

    st.write("")
    num_people = cols[1].slider(f"Number of people within the group", 2, 10, value = 2, step = 1)  
    
    st.write("")      
    group = cols[2].multiselect("Select the members of the group", ratings_data['UserID'].unique(), max_selections=num_people)
    
    impo_person = cols[2].selectbox('Select the most important person in the group:', group) if rec_type =='Consider the **:violet[opinion of the most respected person]** within the group.' else None
    threshold = cols[2].slider(f"Select the minimum rating to be taken into account", 1.0, 5.0, value = 1.0, step = 0.5) if rec_type =='**:violet[Average]** individual ratings **:violet[higher than a threshold]**.' else 1.0
    
    return rec_type, num_people, group, impo_person, threshold


# Main website
def main():
    st.set_page_config(page_title="Wine Recommender", page_icon=":wine_glass:", layout='wide', initial_sidebar_state="expanded")

    with st.sidebar:
        st.title("WINE RECOMMENDATION SYSTEM")
        st.write("Please select your preferences to get a personalized wine recommendation.")
        st.image("https://raw.githubusercontent.com/rogerioxavier/X-Wines/refs/heads/main/x-wine-logo-300-color.png")
        
        selected = option_menu('WineRecSys', ["Introduction", 'Recommend','About'], 
                                icons=['play-btn','search','info-circle'],menu_icon='book', default_index=0)
    
    # Introudction page of the website 
    if selected=="Introduction":
        st.title("WELCOME TO OUR WINE RECOMMENDATION SYSTEM!")
        st.image("https://images.squarespace-cdn.com/content/v1/63a07d348623ab5b47d4008d/e8783bc3-0e9d-4913-a619-f02cd27f4570/barrels.jpg", use_column_width="auto")
        st.subheader("This is a simple web application that recommends a wine based on user preferences.")

    # Main page of the website, where all the recommendation systems are located 
    if selected=="Recommend":

        if 'page' not in st.session_state:
            st.session_state.page = 'home'
        
        # Page navigation logic
        if st.session_state.page == 'home':
            st.header("What would you like to do?")

            cols = st.columns(3, gap="small", vertical_alignment="top")
            with cols[0]:
                st.header(":violet[RANDOM]")
                st.image("https://cdn.icon-icons.com/icons2/2066/PNG/512/search_icon_125165.png", width=100)
                if st.button("Let's try something new"):
                    st.session_state.page = 'try_new'
            with cols[1]:
                st.header(":violet[MY RECOMMENDATIONS]")
                st.image("https://cdn.iconscout.com/icon/free/png-256/free-person-icon-download-in-svg-png-gif-file-formats--user-profile-account-avatar-interface-pack-icons-1502146.png", width=100)
                if st.button("Recommend me"):
                    st.session_state.page = 'recommend_individual'
            with cols[2]:
                st.header(":violet[GROUP RECOMMENDATIONS]")
                st.image("https://cdn.icon-icons.com/icons2/1744/PNG/512/3643747-friend-group-people-peoples-team_113434.png", width=100)
                if st.button("Recommend to all"):
                    st.session_state.page = 'recommend_group'

        ####### NON-PERSONALIZED PAGE #######
        elif st.session_state.page == 'try_new':
            st.title("LET'S TRY SOMETHING NEW")

            c0, c1, c2, c3 = st.columns([0.5, 1.5, 5, 0.5])
            c1.image("https://cdn.icon-icons.com/icons2/2066/PNG/512/search_icon_125165.png", width=200)

            c2.write("Select your preferences to get a new recommendation.")
            with c2.expander("ℹ️ General instructions", expanded=False):
                st.write("Here you will have some information about how to use the following features")
                
                st.markdown(
                    """
                    ### Non-personalized recommender
                    Recommends items based on the filtering and order you choose.
                    """
                )

            st.divider()
            
            col1, col2, col3 = st.columns([2.1,0.1,4.1])

            with col1:
                with st.container(border=True):
                    st.markdown("#### Choose your preferences")
                    selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs = options(num_cols=2, num_group=1, num_recom="")
                st.markdown('####')

                # Store the flag in session state that the button was pressed
                if st.button("Recommend Wine(s)"):
                    st.session_state.recommend_wine_clicked = True

                # Button to go back to home page
                if st.button("Back Home"):
                    st.session_state.page = 'home'
                    st.session_state.recommend_wine_clicked = False
                
            with col3: 
                subcols = st.columns([6,2])
                subcols[0].title("List of Recommended Wines")
                order = subcols[1].selectbox("Order by", ["Rating's value", 'Number of ratings', 'Random']) 
                if order == 'Number of ratings':
                    order = 'numRatings' 
                elif order == "Rating's value":
                    order = 'Rating' 
                else:
                    order = 'Random'

                # Only display the wines after "Recommend Wine" is pressed
                if st.session_state.get('recommend_wine_clicked', False):
                    sorted_df = recommend_wine_filtered(full, selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs, order)
                    display_wines(sorted_df, num_recs, nonpers = True)
            
            if st.session_state.get('recommend_wine_clicked', False):
                st.markdown("###")
                explanation(st.session_state.page, "Non-personalized", "", "", "", sorted_df)

            feedback(nonpers=True)  


        ####### INDIVIDUAL RECOMMENDER PAGE #######
        elif st.session_state.page == 'recommend_individual':
            st.title("MY RECOMMENDATIONS")
            c0, c1, c12, c2, c3 = st.columns([0.5, 1.5, 0.2, 5, 0.5])
            c1.image("https://cdn.iconscout.com/icon/free/png-256/free-person-icon-download-in-svg-png-gif-file-formats--user-profile-account-avatar-interface-pack-icons-1502146.png", width=200)
            c2.write("Select your preferences to get a personalized recommendation.")
            with c2.expander("ℹ️ General instructions", expanded=False):
                st.write("Here you will have some information about how to use the following features")
                
                st.markdown(
                    """
                    ### Individual recommendations
                    Personalized suggestions tailored to your preferences based on past interactions and similar users.

                    ##### Similar to my past liked items
                    Recommendation of items that are similar to those you’ve previously liked, using item features and interactions.

                    ##### Users similar to me
                    Recommendation of items based on the preferences and interactions of users with similar tastes to yours.

                    ##### Based on item interaction
                    Recommendation of items that are similar to those you’ve engaged with based on how other users interacted with them.

                    ##### Based on user interaction
                    Recommendation of items that users with similar preferences to you have liked or interacted with.
                    """
                )

            
            current_user = c1.selectbox("Before starting, please select your **:violet[username]**:", options=sorted(list(ratings_data['UserID'].unique())), index=None)

            if current_user:
                st.divider()
                st.session_state.recommend_wine_clicked = False

                individual_recommender = Recommender()

                col1, col2, col3 = st.columns([2,0.1,6])

                with col1:
                    with st.container(border=True):
                        st.markdown("#### Which type of recommendations would you prefer?")
                        rec_type_indiv = st.radio("Select:", options=["Similar to my past liked items", "Users similar to me"])
                        st.write("")
                        num_recs = st.slider(f"Number of recommendations", 1, 20, value = 1, step = 1)        
                        st.write("")
                    
                    if st.button("Recommend Wine(s)"):
                        st.session_state.recommend_wine_clicked = True
                    if st.button("Back Home"):
                        st.session_state.page = 'home'
                        st.session_state.recommend_wine_clicked = False
                
                with col3: 
                    st.title(rec_type_indiv.upper())
                    recs_indiv = []
                    rec_subtype_indiv = generic_options_indiv() if rec_type_indiv == "Users similar to me" else ""
                    
                    with st.expander("**Add additional filters**", expanded=False):
                        selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs = options(num_cols=5, num_group=500, num_recom=num_recs)
                
                st.divider()
                if st.session_state.get('recommend_wine_clicked', False):
                    st.title("List of Recommended Wines")

                    sorted_df = recommend_wine_for_user(individual_recommender, current_user, rec_type_indiv, rec_subtype_indiv, num_recs)
                    sorted_df.rename(columns={'item': 'WineID', 'score': 'Rating'}, inplace=True)
                    sorted_df = pd.merge(sorted_df, wine_data, on='WineID', how='left')
                    sorted_df = recommend_wine_filtered(sorted_df, selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs, 'Rating')

                    display_wines(sorted_df, num_recs, nonpers = False)
                    st.markdown("#")
                    explanation(st.session_state.page, rec_type_indiv, rec_subtype_indiv, "", "", sorted_df)

                feedback() 


        ####### GROUP RECOMMENDER PAGE #######
        elif st.session_state.page == 'recommend_group':
            st.title("GROUP WINE RECOMMENDATIONS")

            c0, c1, c2, c3 = st.columns([0.5, 1.5, 5, 0.5])
            c1.image("https://cdn.icon-icons.com/icons2/1744/PNG/512/3643747-friend-group-people-peoples-team_113434.png", width=200)

            c2.write("Select your preferences to get a personalized recommendation.")
            with c2.expander("ℹ️ General instructions", expanded=False):
                st.write("Here you will have some information about how to use the following features")
                
                st.markdown(
                    """
                    ### Group recommendations
                    Recommends items to a group of users based on different strategies.

                    ##### All equal
                    Recommends the most popular items among your group members. 

                    ##### Group preferences
                    Considers the preferences of all group members.

                    ##### Given criteria
                    Considers only a subset of items, based on user roles or any other relevant criteria.
                    """
                )

            st.divider()
            # st.session_state.recommend_wine_clicked = False
            col1, col2, col3 = st.columns([2,0.1,6])

            with col1:
                with st.container(border=True):
                    st.markdown("#### Which type of recommendations would you prefer?")
                    rec_type = st.radio("Select:", options=["All equal", "Group preferences", "Given criteria"])
                    st.write("")
                    num_recs = st.slider(f"Number of recommendations", 1, 20, value = 1, step = 1)        
                    st.write("")
                
                if st.button("Recommend Wine(s)"):
                    st.session_state.recommend_wine_clicked = True
                if st.button("Back Home"):
                    st.session_state.page = 'home'
                    st.session_state.recommend_wine_clicked = False
            
            with col3: 
                if rec_type == "All equal":
                    recs = ['Each member vote for his/her **:violet[most preferred alternative]**.', 'Each member vote **:violet[as many alternatives as they wish]**.']
                elif rec_type == "Group preferences": 
                    recs = ['**:violet[Average all]** individual ratings.', '**:violet[Average]** individual ratings **:violet[higher than a threshold]**.',
                            'More importance to **:violet[higher ratings]**.']
                else: # rec_type == "Given criteria"
                    recs = [' Ensure that **:violet[no one is dissatisfied]**.', 'Ensure that the **:violet[majority is satisfied]**.', 'Consider the **:violet[opinion of the most respected person]** within the group.']
                
                st.title(rec_type.upper())
                rec_subtype, num_people, group, impo_person, threshold = generic_options(recs)
                
                with st.expander("**Add additional filters**", expanded=False):
                    selected_type, selected_body, selected_acidity, selected_country, selected_region, selected_ABV, selected_grapes, selected_elaborate, selected_harmonize, min_ratings, num_recs = options(num_cols=5, num_group=num_people, num_recom=num_recs)

            st.divider()
            if st.session_state.get('recommend_wine_clicked', False):
                st.title("List of Recommended Wines")

                sorted_df = recommend_wine_for_group(rec_type, rec_subtype, recs, group, threshold, impo_person, selected_type, selected_body, selected_acidity, selected_country, min_ratings, num_recs)
                display_wines(sorted_df, num_recs, nonpers = False)
                st.markdown("#")
                explanation(st.session_state.page, rec_type, rec_subtype, group, threshold, sorted_df)

            feedback()


# Run the app
if __name__ == '__main__':
    main()