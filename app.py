from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

app = Flask(__name__)



@app.route('/')
def home():
    data = pd.read_csv('data.csv', header=None, delimiter=';')
    data = data.rename(columns={0: 'Transactions'})
    sample_data = data.head(10)  # Get the first 5 rows as sample data
    return render_template('index.html', sample_data=sample_data)

@app.route('/updated_results')
def updated_results():
    # Implement the code to display updated results here
    return "Updated Results Here"

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    # Get parameter values from the form
    min_support = float(request.form['min_support'])
    min_confidence = float(request.form['min_confidence'])
    min_lift = float(request.form['min_lift'])
    min_length = float(request.form['min_length'])

    data = pd.read_csv('data.csv', header=None, delimiter=';')
    data = data.rename(columns={0: 'Transactions'})
    oht = data['Transactions'].str.get_dummies(',')
    oht = oht.astype(bool)
    # Convert frozensets to regular sets and format for display
    def format_items(itemset):
        return ', '.join(itemset)
    
    # Use Apriori to find frequent item sets
    frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
    if not frequent_itemsets.empty:
        association_rules_df = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
        filtered_rules = association_rules_df[
            (association_rules_df['confidence'] >= min_confidence) &
            (association_rules_df['lift'] >= min_lift) &
            (association_rules_df['antecedents'].apply(lambda x: len(x)) >= int(min_length))
        ]
        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(format_items)
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(format_items)

        filtered_rules = filtered_rules.sort_values(by=['confidence'], ascending=False)
        numberOfAssociations = filtered_rules.shape[0]
    else:
        filtered_rules = []
        numberOfAssociations = 0

    # Redirect to the results page
    return render_template('result.html', n=numberOfAssociations, mins=min_support, minc=min_confidence,
                           minl=min_lift, minlen=min_length, rules=filtered_rules)

if __name__ == '__main__':
    app.run(debug=True)





    