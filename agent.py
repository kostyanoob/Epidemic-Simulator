from MyDate import MyDate  # class
import pandas as pd
import numpy as np
from typing import Tuple, List, Union


class ReturnFromIsolation:
    """
    A class that will be in charge of choosing among the isolated people, the ones that should be re-tested.
    """
    def __init__(self):
        pass

    def get_re_test_list(self, personid_isolationdays_tuple_list):
        """
        :param personid_isolationdays_tuple_list - an iterable containing 2-tuples, with the person_id being the first
        coordinate and the number of isolation days being the second coordinate.

        Returns a list of person_ids that should be retested
        """
        pass


class ReturnFromIsolationDelayInterval(ReturnFromIsolation):
    """
    A class that will be in charge of choosing among the isolated people, the ones that should be re-tested.
    """
    def __init__(self, raw_name):
        super().__init__()
        parsed_name = raw_name.split("-")
        bad_name = False
        if len(parsed_name) != 2:
            bad_name = True
        else:
            parsed_name_0 = parsed_name[0].split("_")
            parsed_name_1 = parsed_name[1].split("_")
            if parsed_name_0[0] != "Delay" or parsed_name_1[0] != "Interval":
                bad_name = True
            else:
                try:
                    self.delay, self.interval = int(parsed_name_0[1]), int(parsed_name_1[1])
                except ValueError:
                    bad_name = True

        if bad_name:
            raise TypeError("return from isolation policy raw name must be in a form of Delay_<int>-Interval_<int>, "
                            "given: {}".format(raw_name))

    def get_re_test_list(self, personid_isolationdays_tuple_list):
        """
        :param personid_isolationdays_tuple_list - an iterable containing 2-tuples, with the person_id being the first
        coordinate and the number of isolation days being the second coordinate.

        Returns a list of person_ids that at the current have spent exactly "delay + k * interval" days
         (for k=0,1,2,...), and hence should be retested.
        """
        ret_list = []
        for person_id, isolation_days in personid_isolationdays_tuple_list:
            if isolation_days >= self.delay and (isolation_days - self.delay) % self.interval == 0:
                ret_list.append(person_id)
        return ret_list


class Agent:
    def __init__(self, name=None, rfi=None):

        if not isinstance(name, str):
            raise TypeError("The name of the agent must be a string")
        self.name = name

        if not isinstance(rfi, ReturnFromIsolation):
            raise TypeError("The return from isolation must be a proper class inherited from ReturnFromIsolation class")
        self.rfi = rfi

    def __str__(self):
        if self.name == None:
            return "{} : instance of Agent".format(self.name)
        return "Anonymous Agent"

    def request(self):
        raise NotImplementedError

    def decision(self):
        raise NotImplementedError

    def get_re_test_list(self, personid_isolationdays_tuple_list):
        """
        :param personid_isolationdays_tuple_list - an iterable containing 2-tuples, with the person_id being the first
        coordinate and the number of isolation days being the second coordinate.

        Returns a list of person_ids that should be retested
        """
        return self.rfi.get_re_test_list(personid_isolationdays_tuple_list)


class No_policy_agent(Agent):
    def request(self):
        return 0  # "no request"

    def decision(self):
        return 100  # "no isolation"

    def __str__(self):
        if self.name is not None:
            result = super().__str__() + "\nType: No policy agent"
            return result
        return "No policy agent: Anonymous "


class Symptom_based_agent(Agent):
    def request(self):
        return 0  # "no request"

    def decision(self):
        return 101  # isolate

    def __str__(self):
        if self.name is not None:
            result = super().__str__() + "\nType: SB, Symptom based agent"
            return result
        return " SB: Symptom based agent: **Anonymous** "


class Risk_factor_greedy_agent(Agent):
    """ This agent is denoted RFG
    """

    def request(self):
        return 1  # "apply_test"

    def apply_test(self, bdget):
        return ["RFG", bdget]

    def decision(self):
        """Send isolate flag to isolate tested negative people (on the list it resquested )
        """
        return 101  # isolate

    def __str__(self):
        if self.name is None:
            result = super().__str__() + "\nType:RFG, Risk factor greedy agent"
            return result
        return "RFG: Risk factor greedy agent: **Anonymous** "


class Random_agent(Agent):
    """ This agent is denoted Rand
    """

    def request(self):
        return 1  # "apply_test"

    def apply_test(self, bdget):
        return ["Rand", bdget]

    def decision(self):
        """Send isolate flag to isolate tested negative people (on the list it resquested )
        """
        return 101  # isolate

    def __str__(self):
        if self.name is None:
            result = super().__str__() + "\nType:Rand, Random agent"
            return result
        return "Rand: Random agent: **Anonymous** "


class Optimization_agent(Agent):
    """ This agent is denoted Optimization
    """

    def request(self):
        return 1  # "apply_test"

    def apply_test(self, bdget):
        return ["Optimization", bdget]

    def decision(self):
        """Send isolate flag to isolate tested negative people (on the list it resquested )
        """
        return 101  # isolate

    def __str__(self):
        if self.name is None:
            result = super().__str__() + "\nType:Optimization, Optimization-based agent"
            return result
        return "Optimization: Optimization agent: **Anonymous** "


def agent_factory(agent_type: str, return_from_isolation_raw_name: str, agent_name="Isolation_Strategy_Agent"):
    if "Delay" in return_from_isolation_raw_name and "Interval" in return_from_isolation_raw_name:
        rfi = ReturnFromIsolationDelayInterval(return_from_isolation_raw_name)
    else:
        raise NotImplementedError(
            "Return form isolation policy cannot be configured for a desired raw-name "
            "{} is not implemented".format(return_from_isolation_raw_name))

    if agent_type == "Optimization":
        return Optimization_agent(agent_name, rfi)
    elif agent_type == "Rand":
        return Random_agent(agent_name, rfi)
    elif agent_type == "RFG":
        return Risk_factor_greedy_agent(agent_name, rfi)
    elif agent_type == "Symp":
        return Symptom_based_agent(agent_name, rfi)
    elif agent_type == "nopolicy":
        return No_policy_agent(agent_name, rfi)
    else:
        raise NotImplementedError(
            "Agent type {} is not implemented".format(agent_type))
# similarly to isolate, apply_score depends on the class of agent, return the B highly score people (negative)
# automatic reverse isolation (depend of the agent)
