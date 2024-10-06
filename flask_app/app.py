from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_dance.contrib.google import make_google_blueprint, google
from flask_cors import CORS
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os
# from sklearn.neighbors import NearestNeighbors
import torch
# from transformers import BertModel, BertTokenizer

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

google_bp = make_google_blueprint(client_id="187280521460-bv2i6m1q33scecqgq3usbd0cf8uhrv8s.apps.googleusercontent.com", client_secret="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", redirect_to="google_authorized")
app.register_blueprint(google_bp, url_prefix="/login")

items = []

 
# Load BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_md")
# Load the dataset
df = pd.read_csv('/Users/tejalnair/Desktop/BRH/impact-zero/NetZero - Sheet4.csv')
df2 = pd.read_csv('/Users/tejalnair/Desktop/BRH/impact-zero/NetZero - Sheet5.csv')
df3 = pd.read_csv('/Users/tejalnair/Desktop/BRH/impact-zero/NetZero - Sheet8.csv')
# df.columns = ['Action', 'Impact_Score']

# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(1)
#     return embeddings
    
# df['embeddings'] = df['Action'].apply(lambda x: get_embedding(x).numpy()[0])
# embedding_matrix = np.vstack(df['embeddings'])

# # Initialize NearestNeighbors model
# nn = NearestNeighbors(n_neighbors=1, metric='cosine')
# nn.fit(embedding_matrix)

categories = {
    "Transportation": "items related to the movement of people or goods, including vehicles, fuels, and public transport",
    "Disposables": "items made of synthetic polymers, such as plastic bottles, bags, paper, and containers that have single-use or throw away",
    "Energy Consumption": "items or activities related to the usage of energy, such as gas, electricity, or fossil fuels",
    "Food production": "items related to growing, harvesting, or producing food, including agriculture, farming, and food processing",
    "Housing and Construction": "items related to buildings, homes, or construction materials, including apartments, cement, and insulation",
    "Personal care and Hygiene": "items used for personal grooming or hygiene, such as soap, shampoo, and cosmetics",
    "Technology": "items related to technology, such as tv, and phone, or technology services, such as chatgpt"
}

category_descriptions = list(categories.values())
category_names = list(categories.keys())
 

def categorize_item(item, categories):
    category_descriptions = list(categories.values())
    category_names = list(categories.keys())
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Combine the item and the category descriptions for vectorization
    all_texts = [item] + category_descriptions

    # Vectorize the item and category descriptions
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute the cosine similarity between the item vector and category vectors
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Find the category with the highest similarity score
    highest_similarity_index = cosine_similarities.argmax()
    best_category = category_names[highest_similarity_index]

    return best_category



@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/new-page', methods=['GET', 'POST'])
def new_page():

    global items
    print(f"Request method: {request.method}")

    if request.method == 'POST':
        # user_input = request.form['user_input']

        selected_image = request.form.get('selected_image')
        print(f"Selected image received: {selected_image}")
        session['selected_image'] = selected_image 

        
        # print(session['items'])
        return redirect(url_for('new_page'))
    items = []
    selected_image = session.get('selected_image')
    # session['selected_image'] = selected_image  # Save selected image in session
    print(f"Selected image in session: {selected_image}")
    return render_template('new_page.html', items=items, selected_image = selected_image)

@app.route('/categorize', methods=['POST'])
def categorize():
    user_input = request.form['user_input']
    
    # Categorize the input
    category = categorize_item(user_input, categories)
    
    # Predict the impact score (without closest action)
    impact_score = predict_score(user_input)
    
    # Return the results as JSON (only category and impact score)
    return jsonify({
        "category": category,
        "impact_score": impact_score
    })
def get_carbon_footprint(action):
    result = np.array(df[df['Action'].str.lower() == action.lower()]['Units (g/xxx)'])
    if result.size > 0:
        return result[0]  # Return the first matched unit
    else:
        # If no exact match, add the item with a default carbon footprint of 5
        new_row = {'Action': action, 'Units (g/xxx)': 'unit', 'Carbon per unit of use': 5}
        # df = df.append(new_row, ignore_index=True)
        return new_row['Units (g/xxx)']
    # return None  # Return None if no match is found
@app.route('/search_item', methods=['POST'])
def search_item():
    search_term = request.form.get('search_term').lower()
    
    # Log search term
    print(f"Search term received: {search_term}")

    # Filter the dataframe for matching items
    filtered_df = df2[df2['Action'].str.lower().str.contains(search_term, na=False)]
    
    # Log the filtered items
    print(f"Filtered items: {filtered_df[['Action', 'Carbon per unit of use', 'Why this has carbon footprint']]}")

    # Prepare the result with item name, carbon per unit, and explanation
    items = filtered_df[['Action', 'Carbon per unit of use', 'Why this has carbon footprint']].to_dict(orient='records')


    # Log the items being returned
    print(f"Items being returned: {items}")

    return jsonify(items)
# Route to handle item information request
@app.route('/get-item-info', methods=['POST'])
def get_item_info():
    global items
    user_input = request.form['user_input'].strip()
    print("User Input:", user_input)
    items.append(user_input)
    print("added to list", items)
    
    unit = get_carbon_footprint(user_input)
    
    if unit:
        print("Unit Found:", unit)
        return jsonify({'unit': unit})  # Only return the unit in the response
    else:
        print("No Unit Found")
        return jsonify({'error': 'No matching item found'})
def get_carbon_using(action):
    result = np.array(df[df['Action'].str.lower() == action.lower()]['Carbon per unit of use'])
    if result.size > 0:
        return float(result[0])  # Ensure it's a float for multiplication
    else:
       # If no exact match, add the item with a default carbon footprint of 5
        new_row = {'Action': action, 'Units (g/xxx)': 'unit', 'Carbon per unit of use': 5}
        # df = df.append(new_row, ignore_index=True)
        return new_row['Carbon per unit of use']
    # return None  
@app.route('/get-carbon-usage', methods=['POST'])
def get_carbon_usage():
    user_input = request.form['user_input'].strip()
    print("User Input:", user_input)
    
    value = get_carbon_using(user_input)
    
    if value:
        print("Unit Found:", value)
        return jsonify({'carbon_usage': value})  # Only return the unit in the response
    else:
        print("No Unit Found")
        return jsonify({'error': 'No matching item found'})

@app.route('/result')
def result():
    global items
    # items = session.get('items', [])  # Get the list of items from session
    print(items)
    # categorized_items = [(item, categorize_item(item, categories)) for item in items]  # Categorize each item
    category_counts = {category: 0 for category in categories}
    
    for item in items:
        category = categorize_item(item, categories)
        if category:
            category_counts[category] += 1
    selected_image = session.get('selected_image')
    return render_template('result.html', category_counts=category_counts, selected_image = selected_image)

@app.route('/about')
def about():
    if request.method == 'POST':
        # user_input = request.form['user_input']

        selected_image = request.form.get('selected_image')
        print(f"Selected image received: {selected_image}")
        session['selected_image'] = selected_image 

        
        # print(session['items'])
        return redirect(url_for('new_page'))
    items = []
    selected_image = session.get('selected_image')
    # session['selected_image'] = selected_image  # Save selected image in session
    print(f"Selected image in session: {selected_image}")
    return render_template('about.html', items=items, selected_image = selected_image)

@app.route('/info')
def info():
    if request.method == 'POST':
        # user_input = request.form['user_input']

        selected_image = request.form.get('selected_image')
        print(f"Selected image received: {selected_image}")
        session['selected_image'] = selected_image 

        
        # print(session['items'])
        return redirect(url_for('new_page'))
    items = []
    selected_image = session.get('selected_image')
    # session['selected_image'] = selected_image  # Save selected image in session
    print(f"Selected image in session: {selected_image}")
    return render_template('info.html', items=items, selected_image = selected_image)

@app.route('/rec')
def rec():
    global items
    if request.method == 'POST':
        # user_input = request.form['user_input']

        selected_image = request.form.get('selected_image')
        print(f"Selected image received: {selected_image}")
        session['selected_image'] = selected_image 

        
        # print(session['items'])
        return redirect(url_for('new_page'))
    
    selected_image = session.get('selected_image')

    # session['selected_image'] = selected_image  # Save selected image in session
    print(f"Selected image in session: {selected_image}")
    alternatives = []
    amounts = []
    total_amount = 0

    for item in items:
        print(item)
        item = item.lower()
        if item in df3['Action'].str.lower().values:
        # Find the alternative corresponding to the given action name
            print('im here')
            alternative = df3.loc[df3['Action'].str.lower() == item, 'Alternative'].values[0]
            amount = df3.loc[df3['Action'].str.lower() == item, 'Amount'].values[0]
            total_amount += amount
            # alternatives.append(alternative)
            alternatives.append({
                    'action': item, 
                    'alternative': alternative, 
                    'amount': float(amount)  # Convert np.float64 to a regular float
                })
    max_brightness_reduction = 0.2  # Minimum brightness level (e.g., 20%)
    brightness_level = max(1 - total_amount / 1000, max_brightness_reduction)  # Scale brightness
    
    print("the items", items)
    print("the alternatives",alternatives)
    session['alternatives'] = alternatives
    return render_template('rec.html', items=items, selected_image = selected_image, alternatives = alternatives, brightness_level=brightness_level)
@app.route('/login')
def login():
    print("Redirecting to:", url_for('google.login', _external=True))
    if not google.authorized:
        return redirect(url_for('google.login'))
    # resp = google.get("/plus/v1/people/me")
    # assert resp.ok, resp.text
    # user_info = resp.json()
    # session['user_email'] = user_info['emails'][0]['value']
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('index'))

@app.route('/google_login')
def google_login():
    if 'user_email' in session:
        return f"Hello, {session['user_email']}! You are logged in with Google."
    return redirect(url_for('login'))
@app.route('/login/google/authorized')
def google_authorized():
    try:
        resp = google.authorized_response()
        if resp is None:
            return 'Access denied: reason={0} error={1}'.format(
                request.args['error_reason'],
                request.args['error_description']
            )
        session['google_token'] = (resp['access_token'], '')
        return redirect(url_for('index'))
    except Exception as e:
        return str(e)

@app.route('/show_routes')
def show_routes():
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        options = {}
        for arg in rule.arguments:
            options[arg] = "[{0}]".format(arg)

        methods = ','.join(rule.methods)
        url = url_for(rule.endpoint, **options)
        line = urllib.parse.unquote("{:50s} {:20s} {}".format(rule.endpoint, methods, url))
        output.append(line)

    for line in sorted(output):
        print(line)

    return "Routes printed to console"

if __name__ == '__main__':
    app.run(debug=True)
