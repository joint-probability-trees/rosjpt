import json
from typing import Dict

import rospy
import probabilistic_reasoning_msgs.srv
import std_srvs.srv
import jpt
import jpt.base.intervals
from collections.abc import Iterable


class JPTReasoner:
    """Probabilistic reasoning using joint probability trees as ros service."""

    def __init__(self):
        """
        Create the reasoner and all its service interfaces.
        """

        rospy.init_node("jpt")
        self.tree = jpt.trees.JPT.load(rospy.get_param("path"))
        self.mpe_service = rospy.Service("mpe", probabilistic_reasoning_msgs.srv.mpe, self.handle_mpe)
        self.infer_service = rospy.Service("infer", probabilistic_reasoning_msgs.srv.infer, self.handle_infer)
        self.sample_mpe_service = rospy.Service("sample_mpe", probabilistic_reasoning_msgs.srv.sample_mpe,
                                                self.handle_sample_mpe)
        self.reset = rospy.Service("reset", std_srvs.srv.Empty, self.handle_reset)
        self.apply = rospy.Service("apply_evidence", probabilistic_reasoning_msgs.srv.apply_evidence,
                                   self.handle_apply_evidence)

    def assignment_from_json_dict(self, assignment: Dict) -> jpt.variables.LabelAssignment:
        """
        Create a usable assignment for the model from (ambiguous) json dictionary

        :param assignment: An assignment received from a service request
        :return: jpt.variables.LabelAssignment
        """
        result = dict()
        for variable_name, value in assignment.items():
            variable = self.tree.varnames[variable_name]

            if variable.integer or variable.symbolic:
                if not isinstance(value, Iterable):
                    parsed_value = value
                else:
                    parsed_value = set(value)

            elif variable.numeric:
                if not isinstance(value, Iterable):
                    parsed_value = value
                else:
                    parsed_value = list(value)
            else:
                raise ValueError("Variable of type %s unknown." % type(variable))

            result[variable_name] = parsed_value

        return self.tree.bind(result)

    def assignment_to_json_dict(self, assignment: jpt.variables.LabelAssignment) -> Dict:
        """
        Parse an answer from the model to a json serializable format.
        :param assignment: The assignment to convert
        :return: json serializable dictionary
        """

        # initialize result
        result = dict()

        # for every variable and its value
        for variable, value in assignment.items():

            # easily handle discrete structures
            if variable.integer:
                parsed_value = list(value)
            elif variable.symbolic:
                parsed_value = list(value)

            # if numeric
            elif variable.numeric:

                # simplify the result
                value = value.simplify()

                # convert RealSet to list of lists
                if isinstance(value, jpt.base.intervals.RealSet):
                    parsed_value = [[interval.lower, interval.upper] for interval in value.intervals]

                # convert ContinuousSet to list
                elif isinstance(value, jpt.base.intervals.ContinuousSet):
                    parsed_value = [value.lower, value.upper]
                else:
                    raise ValueError("Assignment of type %s is unknown." % type(value))
            else:
                raise ValueError("Variable of type %s unknown." % type(variable))

            result[variable.name] = parsed_value

        return result

    def handle_mpe(self, request: probabilistic_reasoning_msgs.srv.mpeRequest) -> \
            probabilistic_reasoning_msgs.srv.mpeResponse:
        """
        Perform an MPE inference on this reasoner.
        :param request: An mpeRequest with the evidence
        :return: An mpeResponse with the MPE state, likelihood and rather if it's possible or not.
        """
        evidence = self.assignment_from_json_dict(json.loads(request.evidence))
        response = probabilistic_reasoning_msgs.srv.mpeResponse()

        result = self.tree.mpe(evidence, fail_on_unsatisfiability=False )

        if result is None:
            response.satisfiable = False
            return response

        mpes, likelihood = result

        response.likelihood = likelihood
        response_mpes = []

        for mpe in mpes:
            response_mpes.append(self.assignment_to_json_dict(mpe))

        response.mpe = json.dumps(response_mpes)
        response.satisfiable = True
        return response

    def handle_infer(self, request: probabilistic_reasoning_msgs.srv.inferRequest) -> \
            probabilistic_reasoning_msgs.srv.inferResponse:
        """
        Perform a conditional query in the reasoner.
        :param request: An infer request with query and evidence
        :return: An infer response with the probability and rather if its possible or not.
        """
        response = probabilistic_reasoning_msgs.srv.inferResponse()
        query = self.assignment_from_json_dict(json.loads(request.query))
        evidence = self.assignment_from_json_dict(json.loads(request.evidence))
        probability = self.tree.infer(query, evidence, fail_on_unsatisfiability=False)

        if probability is None:
            response.satisfiable = False
            return response

        response.probability = probability
        response.satisfiable = True

        return response

    def handle_sample_mpe(self, request: probabilistic_reasoning_msgs.srv.sample_mpeRequest) -> \
            probabilistic_reasoning_msgs.srv.sample_mpeResponse:
        """
        Sample from the MPE state of this reasoner.
        :param request: A sample_mpe request that contains the number of samples that are required
        :return: sample_response with the json serialized list of samples
        """
        mpe, likelihood = self.tree.mpe({})

        mpe_tree = self.tree.conditional_jpt(mpe[0])

        samples = mpe_tree.sample(request.amount)

        response = probabilistic_reasoning_msgs.srv.sample_mpeResponse()
        response.samples = json.dumps(samples.tolist())
        return response

    def __del__(self):
        """
        Remove ros parameters of this class on destruction of the server.
        """
        rospy.delete_param("path")

    def handle_reset(self, request: std_srvs.srv.EmptyRequest) -> std_srvs.srv.EmptyResponse:
        """
        Reset the model of this reasoner and undo all alternations.
        :param request: An empty request
        :return: An empty response
        """
        self.tree = jpt.trees.JPT.load(rospy.get_param("path"))
        return std_srvs.srv.EmptyResponse()

    def handle_apply_evidence(self, request: probabilistic_reasoning_msgs.srv.apply_evidenceRequest) -> \
            probabilistic_reasoning_msgs.srv.apply_evidenceResponse:
        """
        Applies evidence to the model via jpt.trees.JPT.conditional_jpt. This will not do anything if the evidence
        is unsatisfiable.
        :param request: A request with evidence that will be applied.
        :return: Response with rather the evidence is satisfiable or not.
        """

        evidence = self.assignment_from_json_dict(json.loads(request.evidence))

        conditional_jpt = self.tree.conditional_jpt(evidence, fail_on_unsatisfiability=False)

        result = probabilistic_reasoning_msgs.srv.apply_evidenceResponse()

        if conditional_jpt is not None:
            self.tree = conditional_jpt
            result.satisfiable = True
        else:
            result.satisfiable = False

        return result
