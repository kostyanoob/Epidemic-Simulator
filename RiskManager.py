from typing import Tuple, List, Union
import numpy as np
import yaml

from Util.numeric import floatify_string, modified_sigmoid


class RiskManager:
    def __init__(self, load_path: str = ""):
        """

        :param load_path: <str> if specified, then a yaml file will be used
        to load the configurations of the risk coefficients and discounts.
        """

        self.coeff_ext_risk_kind = None
        self.coeff_ext_risk_vector = None
        self.coeff_int_risk_kind = None
        self.coeff_int_risk_vector = None
        self.coeff_grp_risk_kind = None
        self.coeff_grp_risk_vector = None
        self.discount_f_neg_kind = None
        self.discount_f_neg_vector = None
        self.discount_f_pos_kind = None
        self.discount_f_pos_vector = None
        self.t1 = None
        self.s1 = None
        self.t2 = None
        self.s2 = None
        self.t3 = None
        self.s3 = None
        self.t4 = None
        self.s4 = None
        self.t5 = None
        self.s5 = None
        self.p_test_error_idle = None
        self.p_test_error_infected = None
        self.p_test_error_detectable = None
        self.p_test_error_contagious = None
        self.p_test_error_symptomatic = None
        self.p_test_error_non_contagious = None

        if load_path != "":
            self.load(load_path)

    def set_f_neg_discount(self, risk_discount_kind: str, arg: Union[np.ndarray, List, str, Tuple]):
        """
        sets a discount factor vector (for custom kind) or a function (for linear and uniform kinds)
        :param risk_discount_kind: "sigmoid" or "custom" string
        :param arg: in case of a sigmoid - a 2-tuple, 2-entry list or an ndarray of length 2 (1D)
                    in case of a custom - a string, a list or a tuple...
        :return:
        """
        allowed_discount_kinds = ["custom", "sigmoid"]
        if risk_discount_kind not in allowed_discount_kinds:
            raise DiscountTypeException(
                risk_discount_kind, allowed_discount_kinds)

        if type(arg) is np.ndarray:
            if len(arg.shape) != 1:
                raise DiscountArgNdArrayShapeException(arg)
            else:
                discount_vector = arg
        elif type(arg) in [list, tuple]:
            discount_vector = np.array(arg)
        elif type(arg) is str:
            filtered_lst = filter(lambda s: s != '', arg.split(","))
            discount_vector = np.array(
                list(map(floatify_string, filtered_lst)))
        else:
            # type(arg) not in [str, list, tuple, np.ndarray]:
            raise DiscountArgTypeException(arg)

        self.discount_f_neg_kind = risk_discount_kind
        if risk_discount_kind == "custom":
            self.discount_f_neg_vector = discount_vector
        elif risk_discount_kind == "sigmoid":
            if len(discount_vector) != 2:
                raise DiscountArgSigmoidException(discount_vector)
            else:
                self.discount_f_neg_vector = discount_vector

    def get_discount_f_neg(self, time_elapsed: int) -> float:
        """
        Return a discount factor term "f_neg" for a given "time elapsed" value
        :param time_elapsed:
        :return:
        """
        if self.discount_f_neg_kind is None:
            raise ConfigurationKindException("f_neg_Discount")
        elif self.discount_f_neg_kind in ["custom", "sigmoid"] and self.discount_f_neg_vector is None:
            raise ConfigurationArrayMissingException(
                "Discount f neg kind", self.discount_f_neg_kind)

        if self.discount_f_neg_kind == "sigmoid":
            # a,b
            coefficient, shift = self.discount_f_neg_vector[0], self.discount_f_neg_vector[1]
            return modified_sigmoid(x=time_elapsed, coefficient=coefficient, shift=shift)
        elif self.discount_f_neg_kind == "custom":
            if time_elapsed >= len(self.discount_f_neg_vector):
                return 1.0
            elif time_elapsed < 0:
                return 0.0
            else:
                return self.discount_f_neg_vector[time_elapsed]

    def get_discount_f_pos(self, time_elapsed: int) -> float:
        """
        Return a discount factor for a given "time elapsed" value
        :param time_elapsed:
        :return:  
        """
        if self.discount_f_pos_kind is None:
            raise ConfigurationKindException("f_pos_Discount")
        elif self.discount_f_pos_kind in ["custom_threshold", "sigmoid"] and self.discount_f_pos_vector is None:
            raise ConfigurationArrayMissingException(
                "Discount f pos kind", self.discount_f_pos_kind)

        if self.discount_f_pos_kind == "sigmoid":
            # a,b
            coefficient, shift = self.discount_f_pos_vector[0], self.discount_f_pos_vector[1]
            return modified_sigmoid(x=time_elapsed, coefficient=coefficient, shift=shift)
        elif self.discount_f_pos_kind == "custom_threshold":
            if time_elapsed > self.discount_f_pos_vector[0]:
                return 1.0
            else:
                return 0.0

    def set_coefficients(self, risk_type, risk_factor_coeff_kind: str, arg: Union[np.ndarray, List, str, Tuple, None] = None):
        """
        Sets the risk factor coefficients.
        :param risk_factor_coeff_kind: "uniform", "linear", "custom"
        :param arg: None for "linear" and for "uniform" kinds, an iterable for "custom".
                    The iterable should represent the first [0,1,2,...] values of the
                    coefficient vectors, or alternatively a string of comma separated
                    float values can be passed.
        :return:
        """
        allowed_risk_type = ["ext_risk", "int_risk", "group_risk"]
        allowed_coeff_kinds = ["linear", "uniform", "custom"]
        if risk_factor_coeff_kind not in allowed_coeff_kinds:
            raise RiskFactorCoefficientTypeException(
                risk_factor_coeff_kind, allowed_coeff_kinds)

        if risk_type not in allowed_risk_type:
            raise RiskTypeException(risk_type, allowed_risk_type)
        if risk_type == 'ext_risk':
            self.coeff_ext_risk_kind = risk_factor_coeff_kind
            self.coeff_ext_risk_vector = None

            if risk_factor_coeff_kind == "custom":
                if type(arg) is np.ndarray:
                    if len(arg.shape) != 1:
                        raise RiskFactorCoefficientArgNdArrayShapeException(
                            arg)
                    else:
                        self.coeff_ext_risk_vector = arg
                elif type(arg) in [list, tuple]:
                    self.coeff_ext_risk_vector = np.array(arg)
                elif type(arg) is str:
                    filtered_lst = filter(lambda s: s != '', arg.split(","))
                    self.coeff_ext_risk_vector = np.array(
                        list(map(floatify_string, filtered_lst)))
                else:
                    # type(arg) not in [list, tuple, np.ndarray]:
                    raise RiskFactorCoefficientArgTypeException(arg)
        if risk_type == 'int_risk':
            self.coeff_int_risk_kind = risk_factor_coeff_kind
            self.coeff_int_risk_vector = None

            if risk_factor_coeff_kind == "custom":
                if type(arg) is np.ndarray:
                    if len(arg.shape) != 1:
                        raise RiskFactorCoefficientArgNdArrayShapeException(
                            arg)
                    else:
                        self.coeff_int_risk_vector = arg
                elif type(arg) in [list, tuple]:
                    self.coeff_int_risk_vector = np.array(arg)
                elif type(arg) is str:
                    filtered_lst = filter(lambda s: s != '', arg.split(","))
                    self.coeff_int_risk_vector = np.array(
                        list(map(floatify_string, filtered_lst)))
                else:
                    # type(arg) not in [list, tuple, np.ndarray]:
                    raise RiskFactorCoefficientArgTypeException(arg)
        if risk_type == 'group_risk':
            self.coeff_grp_risk_kind = risk_factor_coeff_kind
            self.coeff_grp_risk_vector = None

            if risk_factor_coeff_kind == "custom":
                if type(arg) is np.ndarray:
                    if len(arg.shape) != 1:
                        raise RiskFactorCoefficientArgNdArrayShapeException(
                            arg)
                    else:
                        self.coeff_grp_risk_vector = arg
                elif type(arg) in [list, tuple]:
                    self.coeff_grp_risk_vector = np.array(arg)
                elif type(arg) is str:
                    filtered_lst = filter(lambda s: s != '', arg.split(","))
                    self.coeff_grp_risk_vector = np.array(
                        list(map(floatify_string, filtered_lst)))
                else:
                    # type(arg) not in [list, tuple, np.ndarray]:
                    raise RiskFactorCoefficientArgTypeException(arg)

    def get_coefficients(self, risk_type, num_coefficients: int) -> np.ndarray:
        """
        :param risk_type: one of the following allowed risk type:["ext_risk", "int_risk", "group_risk"]
        :param num_coefficients: int, specifying how many
        :return: numpy ndarray of coefficients, of a length specified by num_coefficients
                 if num_coefficients is larger than the number of "custom" exiting coefficients,
                 the missing most significant coefficients will be the replica of the
                 most significant existing coefficient.
        """
        allowed_risk_type = ["ext_risk", "int_risk", "group_risk"]
        if risk_type not in allowed_risk_type:
            raise RiskTypeException(risk_type, allowed_risk_type)

        if risk_type == 'ext_risk':
            if self.coeff_ext_risk_kind is None:
                raise ConfigurationKindException("Risk coefficient")
            elif self.coeff_ext_risk_kind == "custom" and self.coeff_ext_risk_vector is None:
                raise ConfigurationArrayMissingException(
                    "Risk coefficient", self.coeff_ext_risk_kind)

            if self.coeff_ext_risk_kind == "linear":
                risk_factor_coefficients = np.arange(
                    start=1, stop=num_coefficients + 1, step=1, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_ext_risk_kind == "uniform":
                risk_factor_coefficients = np.ones(
                    num_coefficients, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_ext_risk_kind == "custom":
                if num_coefficients < len(self.coeff_ext_risk_vector):
                    risk_factor_coefficients = self.coeff_ext_risk_vector[:num_coefficients]
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                elif num_coefficients > len(self.coeff_ext_risk_vector):
                    risk_factor_coefficients = self.coeff_ext_risk_vector[-1] * np.ones(
                        num_coefficients)
                    risk_factor_coefficients[:len(
                        self.coeff_ext_risk_vector)] = self.coeff_ext_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                else:
                    risk_factor_coefficients = self.coeff_ext_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()  # not necessary!
            else:
                raise RiskFactorCoefficientTypeException(
                    self.coeff_ext_risk_kind, ["linear", "uniform", "custom"])
            return risk_factor_coefficients

        if risk_type == 'int_risk':
            if self.coeff_int_risk_kind is None:
                raise ConfigurationKindException("Risk coefficient")
            elif self.coeff_int_risk_kind == "custom" and self.coeff_int_risk_vector is None:
                raise ConfigurationArrayMissingException(
                    "Risk coefficient", self.coeff_int_risk_kind)

            if self.coeff_int_risk_kind == "linear":
                risk_factor_coefficients = np.arange(
                    start=1, stop=num_coefficients + 1, step=1, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_int_risk_kind == "uniform":
                risk_factor_coefficients = np.ones(
                    num_coefficients, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_int_risk_kind == "custom":
                if num_coefficients < len(self.coeff_int_risk_vector):
                    risk_factor_coefficients = self.coeff_int_risk_vector[:num_coefficients]
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                elif num_coefficients > len(self.coeff_int_risk_vector):
                    risk_factor_coefficients = self.coeff_int_risk_vector[-1] * np.ones(
                        num_coefficients)
                    risk_factor_coefficients[:len(
                        self.coeff_int_risk_vector)] = self.coeff_int_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                else:
                    risk_factor_coefficients = self.coeff_int_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()  # not necessary!
            else:
                raise RiskFactorCoefficientTypeException(
                    self.coeff_int_risk_kind, ["linear", "uniform", "custom"])
            return risk_factor_coefficients

        if risk_type == 'group_risk':
            if self.coeff_grp_risk_kind is None:
                raise ConfigurationKindException("Risk coefficient")
            elif self.coeff_grp_risk_kind == "custom" and self.coeff_grp_risk_vector is None:
                raise ConfigurationArrayMissingException(
                    "Risk coefficient", self.coeff_grp_risk_kind)

            if self.coeff_grp_risk_kind == "linear":
                risk_factor_coefficients = np.arange(
                    start=1, stop=num_coefficients + 1, step=1, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_grp_risk_kind == "uniform":
                risk_factor_coefficients = np.ones(
                    num_coefficients, dtype=np.float32)
                risk_factor_coefficients /= risk_factor_coefficients.sum()
            elif self.coeff_grp_risk_kind == "custom":
                if num_coefficients < len(self.coeff_grp_risk_vector):
                    risk_factor_coefficients = self.coeff_grp_risk_vector[:num_coefficients]
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                elif num_coefficients > len(self.coeff_grp_risk_vector):
                    risk_factor_coefficients = self.coeff_grp_risk_vector[-1] * np.ones(
                        num_coefficients)
                    risk_factor_coefficients[:len(
                        self.coeff_grp_risk_vector)] = self.coeff_grp_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()
                else:
                    risk_factor_coefficients = self.coeff_grp_risk_vector
                    risk_factor_coefficients /= risk_factor_coefficients.sum()  # not necessary!
            else:
                raise RiskFactorCoefficientTypeException(
                    self.coeff_grp_risk_kind, ["linear", "uniform", "custom"])
            return risk_factor_coefficients

    def load(self, path: str):
        """
        load the configuration from yaml file, and set them to the current object.
        :param path: str, a path to the yaml file, containing the configurations
        """
        try:
            with open(path) as file:
                documents = yaml.full_load(file)
                self.coeff_ext_risk_kind = documents.get(
                    'coeff_ext_risk_kind', None)
                ext_coeff_list = documents.get('coeff_ext_risk_vector', None)
                self.coeff_ext_risk_vector = np.array(
                    ext_coeff_list) if ext_coeff_list is not None else None

                self.coeff_int_risk_kind = documents.get(
                    'coeff_int_risk_kind', None)
                int_coeff_list = documents.get('coeff_int_risk_vector', None)
                self.coeff_int_risk_vector = np.array(
                    int_coeff_list) if int_coeff_list is not None else None

                self.coeff_grp_risk_kind = documents.get(
                    'coeff_grp_risk_kind', None)
                grp_coeff_list = documents.get('coeff_grp_risk_vector', None)
                self.coeff_grp_risk_vector = np.array(
                    grp_coeff_list) if grp_coeff_list is not None else None

                self.discount_f_neg_kind = documents.get(
                    'discount_f_neg_kind', None)
                neg_discount_list = documents.get(
                    'discount_f_neg_vector', None)
                self.discount_f_neg_vector = np.array(
                    neg_discount_list) if neg_discount_list is not None else None

                self.discount_f_pos_kind = documents.get(
                    'discount_f_pos_kind', None)
                pos_discount_list = documents.get(
                    'discount_f_pos_vector', None)
                self.discount_f_pos_vector = np.array(
                    pos_discount_list) if pos_discount_list is not None else None

                t1_value = documents.get('t1', None)
                self.t1 = t1_value[0] if t1_value is not None else None
                s1_value = documents.get('s1', None)
                self.s1 = s1_value[0] if s1_value is not None else None
                t2_value = documents.get('t2', None)
                self.t2 = t2_value[0] if t2_value is not None else None
                s2_value = documents.get('s2', None)
                self.s2 = s2_value[0] if s2_value is not None else None
                t3_value = documents.get('t3', None)
                self.t3 = t3_value[0]if t3_value is not None else None
                s3_value = documents.get('s3', None)
                self.s3 = s3_value[0] if s3_value is not None else None
                t4_value = documents.get('t4', None)
                self.t4 = t4_value[0] if t4_value is not None else None
                s4_value = documents.get('s4', None)
                self.s4 = s4_value[0] if s4_value is not None else None
                t5_value = documents.get('t5', None)
                self.t5 = t5_value[0] if t5_value is not None else None
                s5_value = documents.get('s5', None)
                self.s5 = s5_value[0] if s5_value is not None else None
                p_test_error_idle_value = documents.get(
                    'p_test_error_idle', None)
                self.p_test_error_idle = p_test_error_idle_value[
                    0] if p_test_error_idle_value is not None else None
                p_test_error_infected_value = documents.get(
                    'p_test_error_infected', None)
                self.p_test_error_infected = p_test_error_infected_value[
                    0] if p_test_error_infected_value is not None else None
                p_test_error_detectable_value = documents.get(
                    'p_test_error_detectable', None)
                self.p_test_error_detectable = p_test_error_detectable_value[
                    0] if p_test_error_detectable_value is not None else None
                p_test_error_contagious_value = documents.get(
                    'p_test_error_contagious', None)
                self.p_test_error_contagious = p_test_error_contagious_value[
                    0] if p_test_error_contagious_value is not None else None
                p_test_error_symptomatic_value = documents.get(
                    'p_test_error_symptomatic', None)
                self.p_test_error_symptomatic = p_test_error_symptomatic_value[
                    0] if p_test_error_symptomatic_value is not None else None
                p_test_error_non_contagious_value = documents.get(
                    'p_test_error_non_contagious', None)
                self.p_test_error_non_contagious = p_test_error_non_contagious_value[
                    0] if p_test_error_non_contagious_value is not None else None
        except:
            return

    def save(self, path: str):
        """
        dump the configurations (risk factors, discounts) to a yaml file
        :param path: str, a path to the yaml file, to dump the current object to
        :return:
        """
        dict_file = {}
        if self.coeff_ext_risk_kind is not None:
            dict_file["coeff_ext_risk_kind"] = self.coeff_ext_risk_kind
        if self.coeff_ext_risk_vector is not None:
            dict_file["coeff_ext_risk_vector"] = self.coeff_ext_risk_vector.tolist()

        if self.coeff_int_risk_kind is not None:
            dict_file["coeff_int_risk_kind"] = self.coeff_int_risk_kind
        if self.coeff_int_risk_vector is not None:
            dict_file["coeff_int_risk_vector"] = self.coeff_int_risk_vector.tolist()

        if self.coeff_grp_risk_kind is not None:
            dict_file["coeff_grp_risk_kind"] = self.coeff_grp_risk_kind
        if self.coeff_grp_risk_vector is not None:
            dict_file["coeff_grp_risk_vector"] = self.coeff_grp_risk_vector.tolist()

        if self.discount_f_neg_kind is not None:
            dict_file['discount_f_neg_kind'] = self.discount_f_neg_kind
        if self.discount_f_neg_vector is not None:
            dict_file["discount_f_neg_vector"] = self.discount_f_neg_vector.tolist()
        if self.t1 is not None:
            dict_file['s1'] = self.t1
        if self.t2 is not None:
            dict_file['s2'] = self.t2
        if self.t3 is not None:
            dict_file['s3'] = self.t3
        if self.t4 is not None:
            dict_file['s4'] = self.t4
        if self.t5 is not None:
            dict_file['t5'] = self.t5
        if self.s1 is not None:
            dict_file['s1'] = self.s1
        if self.s2 is not None:
            dict_file['s2'] = self.s2
        if self.s3 is not None:
            dict_file['s3'] = self.s3
        if self.s4 is not None:
            dict_file['s4'] = self.s4
        if self.s5 is not None:
            dict_file['s5'] = self.s5
        if self.p_test_error_idle is not None:
            dict_file['p_test_error_idle'] = self.p_test_error_idle
        if self.p_test_error_infected is not None:
            dict_file['p_test_error_infected'] = self.p_test_error_infected
        if self.p_test_error_detectable is not None:
            dict_file['p_test_error_detectable'] = self.p_test_error_detectable
        if self.p_test_error_contagious is not None:
            dict_file['p_test_error_contagious'] = self.p_test_error_contagious
        if self.p_test_error_symptomatic is not None:
            dict_file['p_test_error_symptomatic'] = self.p_test_error_symptomatic
        if self.p_test_error_non_contagious is not None:
            dict_file['p_test_error_non_contagious'] = self.p_test_error_non_contagious
        with open(path, 'w') as file:
            documents = yaml.dump(dict_file, file)

#################################### Exceptions ####################################


class RiskTypeException(Exception):
    def __init__(self, current, allowed):
        self.message = "Risk type must be one of the following strings: {}. Given: \"{}\"".format(
            str(allowed), str(current))
        super().__init__(self.message)


class RiskFactorCoefficientTypeException(Exception):
    def __init__(self, current, allowed):
        self.message = "Risk factor coefficient must be one of the following strings: {}. Given: \"{}\"".format(
            str(allowed), str(current))
        super().__init__(self.message)


class RiskFactorCoefficientArgTypeException(Exception):
    def __init__(self, current):
        self.message = "Risk factor coefficient argument for \"custom\" type " \
                       "must be of a type (string, tuple, list, or numpy ndarray). Given \"{}\"".format(
                           str(current))
        super().__init__(self.message)


class RiskFactorCoefficientArgNdArrayShapeException(Exception):
    def __init__(self, arg):
        self.message = "While defining risk factor coefficients for \"custom\" type " \
                       "you chose to use numpy ndarray. The array must be single dimensional, " \
                       "whereas you gave {}-dimensional array".format(
                           len(arg.shape))
        super().__init__(self.message)


class DiscountTypeException(Exception):
    def __init__(self, current, allowed):
        self.message = "Risk discount factor coefficient must be " \
                       "one of the following strings: {}. Given: {}".format(
                           str(allowed), str(current))
        super().__init__(self.message)


class DiscountArgTypeException(Exception):
    def __init__(self, current):
        self.message = "Discount argument for  " \
                       "must be of a type (string, tuple, list, or numpy ndarray). Given \"{}\"".format(
                           str(current))
        super().__init__(self.message)


class DiscountArgNdArrayShapeException(Exception):
    def __init__(self, arg):
        self.message = "While defining discount " \
                       "you chose to use numpy ndarray. The array must be single dimensional, " \
                       "whereas you gave {}-dimensional array".format(
                           len(arg.shape))
        super().__init__(self.message)


class DiscountArgSigmoidException(Exception):
    def __init__(self, arg):
        self.message = "While defining discount of a kind \"sigmoid\"" \
                       "you chose to use numpy ndarray for the argument. The array must be single dimensional, " \
                       "whereas you gave {}-dimensional array".format(
                           len(arg.shape))
        super().__init__(self.message)


class ConfigurationKindException(Exception):
    def __init__(self, message):
        self.message = message + " kind was not configured."
        super().__init__(self.message)


class ConfigurationArrayMissingException(Exception):
    def __init__(self, attr: str, kind: str):
        self.message = "{} of a kind \"{}\" kind was " \
                       "not configured with a proper array.".format(attr, kind)
        super().__init__(self.message)
