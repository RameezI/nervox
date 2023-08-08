import json
import unittest
from nervox.utils import serialize_to_json


class JsonSerializationTestCase(unittest.TestCase):     

    def test_default_encoder(self):
        obj = {'key': 'value'}
        result = serialize_to_json(obj)
        self.assertDictEqual(json.loads(result), {'key': 'value'})


if __name__ == '__main__':
    unittest.main()