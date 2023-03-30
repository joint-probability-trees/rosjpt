import unittest
import numpy as np
import pandas as pd
import jpt
import rospy
import probabilistic_reasoning_msgs.srv
import std_srvs.srv
import json


class TestROSJPT(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(69)
        symbolic = np.random.randint(0, 4, 1000).reshape(-1, 1).astype(object)
        symbolic[symbolic == 0] = "A"
        symbolic[symbolic == 1] = "B"
        symbolic[symbolic == 2] = "C"
        symbolic[symbolic == 3] = "D"

        integer = np.random.randint(0, 10, 1000).reshape(-1, 1)
        numeric = np.random.uniform(0, 1, 1000).reshape(-1, 1)

        df = pd.DataFrame(np.concatenate((symbolic, integer, numeric), axis=1),
                          columns=["symbolic", "integer", "numeric"])
        df["symbolic"] = df["symbolic"].astype(str)
        df["integer"] = df["integer"].astype(int)
        df["numeric"] = df["numeric"].astype(float)

        self.model = jpt.trees.JPT(jpt.infer_from_dataframe(df, precision=0.1), min_samples_leaf=0.9)

        self.model.fit(df)
        self.model.save("minimal_test.jpt")

        # update service with new jpt
        reset = rospy.ServiceProxy("/unittest/jpt/reset", std_srvs.srv.Empty)
        reset()

    def test_sample_mpe(self):
        sample_mpe = rospy.ServiceProxy("/unittest/jpt/sample_mpe", probabilistic_reasoning_msgs.srv.sample_mpe)
        samples = sample_mpe(200)
        samples = json.loads(samples.samples)
        self.assertEqual(200, len(samples))
        self.assertTrue(all(self.model.likelihood(samples) > 0))

    def test_mpe_satisfiable(self):
        mpe = rospy.ServiceProxy("/unittest/jpt/mpe", probabilistic_reasoning_msgs.srv.mpe)
        evidence = {"symbolic": "A", "integer": [1, 2, 3], "numeric": [0., 0.5]}
        evidence_dump = json.dumps(evidence)
        result = mpe(evidence_dump)
        self.assertTrue(result.likelihood > 0)
        self.assertTrue(result.satisfiable)
        explanation = json.loads(result.mpe)[0]

        self.assertEqual(explanation["symbolic"], ['A'])
        self.assertEqual(explanation["integer"], [1])
        self.assertAlmostEqual(explanation["numeric"][0], 0, delta=0.001)
        self.assertAlmostEqual(explanation["numeric"][1], 0.5)

    def test_mpe_unsatisfiable(self):
        mpe = rospy.ServiceProxy("/unittest/jpt/mpe", probabilistic_reasoning_msgs.srv.mpe)
        evidence = {"numeric": -1}
        evidence_dump = json.dumps(evidence)
        result = mpe(evidence_dump)
        self.assertEqual(result.mpe, "")
        self.assertEqual(result.likelihood, 0)
        self.assertFalse(result.satisfiable)

    def test_infer_satisfiable(self):
        infer = rospy.ServiceProxy("/unittest/jpt/infer", probabilistic_reasoning_msgs.srv.infer)
        evidence = {"symbolic": "A", "integer": [1, 2, 3], "numeric": [0., 0.5]}
        evidence_ = {"symbolic": "A", "integer": {1, 2, 3}, "numeric": [0., 0.5]}
        query = {"integer": 2}

        result = infer(json.dumps(query), json.dumps(evidence))
        self.assertTrue(result.satisfiable)
        self.assertAlmostEqual(result.probability, self.model.infer(self.model.bind(query), self.model.bind(evidence_)))

    def test_infer_unsatisfiable(self):
        infer = rospy.ServiceProxy("/unittest/jpt/infer", probabilistic_reasoning_msgs.srv.infer)
        evidence = {"symbolic": "A", "integer": [1, 2, 3], "numeric": [-10., -9]}
        query = {"integer": 4}

        result = infer(json.dumps(query), json.dumps(evidence))
        self.assertFalse(result.satisfiable)
        self.assertEqual(result.probability, 0)

    def test_apply_evidence_satisfiable(self):
        apply_evidence = rospy.ServiceProxy("/unittest/jpt/apply_evidence",
                                            probabilistic_reasoning_msgs.srv.apply_evidence)
        reset = rospy.ServiceProxy("/unittest/jpt/reset", std_srvs.srv.Empty)
        evidence = {"symbolic": "A", "integer": [1, 2, 3], "numeric": [0., 0.5]}
        evidence_ = {"symbolic": "A", "integer": {1, 2, 3}, "numeric": [0., 0.5]}

        response = apply_evidence(json.dumps(evidence))
        self.assertTrue(response.satisfiable)

        infer = rospy.ServiceProxy("/unittest/jpt/infer", probabilistic_reasoning_msgs.srv.infer)
        query = {"integer": 3}

        prob_server = (infer(json.dumps(query), json.dumps({}))).probability

        conditional_model = self.model.conditional_jpt(self.model.bind(evidence_))
        self.assertAlmostEqual(conditional_model.infer(conditional_model.bind(query)), prob_server)
        reset()

    def test_apply_evidence_unsatisfiable(self):
        apply_evidence = rospy.ServiceProxy("/unittest/jpt/apply_evidence",
                                            probabilistic_reasoning_msgs.srv.apply_evidence)
        evidence = {"numeric": [-1., -0.5]}
        response = apply_evidence(json.dumps(evidence))
        self.assertFalse(response.satisfiable)


if __name__ == '__main__':
    unittest.main()
