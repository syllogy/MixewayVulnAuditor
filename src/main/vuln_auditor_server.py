#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import json
from collections import namedtuple

from model.audit_model import get_trained_model

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


@app.route('/users', methods=['POST'])
def audit_vuln():
    response = []
    vulns_to_audit = json.loads(request.data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    for vuln in vulns_to_audit:
        response.append(AuditResult(vuln.id, 0))
    return json.dumps(response, default=encode_b)


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
