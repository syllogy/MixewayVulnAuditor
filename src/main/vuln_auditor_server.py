#!/usr/bin/python3
import json
from collections import namedtuple
from flask import Flask, request
from flask_restful import Api
import os

from model.audit_model import predict, get_trained_model

cert = (os.environ['CERTIFICATE'])
key = (os.environ['PRIVATEKEY'])
app = Flask(__name__)
api = Api(app)
model, tokenizer = get_trained_model()


class AuditResult(object):
    id = 0
    audit = 0

    def __init__(self, id, audit):
        self.id = id
        self.audit = audit

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def encode_b(obj):
    if isinstance(obj, AuditResult):
        return obj.__dict__
    return obj


@app.route('/vuln/perdict', methods=['POST'])
def audit_vuln():
    response = []
    vulns_to_audit = json.loads(request.data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    for vuln in vulns_to_audit:
        response.append(AuditResult(vuln.id, predict(model, tokenizer, vuln)))
    return json.dumps(response, default=encode_b)


@app.route('/vuln/train', methods=['POST'])
def train_model():
    response = []
    vulns_to_audit = json.loads(request.data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return json.dumps(response, default=encode_b)


@app.route('/init', methods=['GET'])
def init():
    response = []
    return json.dumps(response, default=encode_b)


if __name__ == '__main__':
    context = (cert, key)
    app.run(host="localhost", port=8445, debug=True, ssl_context=context)
