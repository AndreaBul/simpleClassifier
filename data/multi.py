from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Multi(Resource):
    def get(self, num):
        return {'result':num*10}

api.add_resource(Multi, '/multi/<int:num>')
if __name__ == '__main__':
    app.run(debug=True)