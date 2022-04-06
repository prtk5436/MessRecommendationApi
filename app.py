from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "ENTER MESS NAME AT THE END OF URL......for example: 'https://plp-recommendation.herokuapp.com/mess?title=amul'  here, we have type 'amul' mess name in this format: '/mess?title=amul'"


@app.route('/mess', methods=['GET'])
def recommend_movies():
    res = recommendation.results(request.args.get('title'))
    return jsonify({'recommended': res})


if __name__ == '__main__':
    app.run(port=5000, debug=True)