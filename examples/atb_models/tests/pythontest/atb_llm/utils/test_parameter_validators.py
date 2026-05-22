# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from atb_llm.utils.parameter_validators import (
    StringParameterValidator, IntParameterValidator, FloatParameterValidator, ListParameterValidator,
    DictionaryParameterValidator, CompositeParameterValidator, ParameterValidator, ValidationPipeline,
    ValidationStrategies, ArgumentParser, Field
)

# Common variables
KEY1 = 'key1'
KEY2 = 'key2'
KEY3 = 'key3'
KEY4 = 'key4'
STR_123 = '123'
STR_123456 = '123456'
TEST_ARG = 'test_arg'
NEW_TEST_ARG = 'new_test_arg'
TEST = 'test'


class FieldTest(unittest.TestCase):
    def test_valid_field_initialization(self):
        field_for_test = Field(ge=1, le=10)
        self.assertEqual(field_for_test.ge, 1)
        self.assertEqual(field_for_test.le, 10)
        self.assertIsNone(field_for_test.gt)
        self.assertIsNone(field_for_test.lt)
        self.assertIsNone(field_for_test.min_length)
        self.assertIsNone(field_for_test.max_length)

    def test_invalid_field_initialization_ge_gt(self):
        with self.assertRaises(ValueError):
            Field(ge=1, gt=2)

    def test_invalid_field_initialization_le_lt(self):
        with self.assertRaises(ValueError):
            Field(le=10, lt=9)

    def test_invalid_field_initialization_range_length(self):
        with self.assertRaises(ValueError):
            Field(ge=1, min_length=2)

    def test_invalid_field_initialization_ge_le(self):
        with self.assertRaises(ValueError):
            Field(ge=10, le=1)

    def test_invalid_field_initialization_gt_lt(self):
        with self.assertRaises(ValueError):
            Field(gt=10, lt=9)

    def test_invalid_field_initialization_min_max_length(self):
        with self.assertRaises(ValueError):
            Field(min_length=10, max_length=5)

    def test_field_with_length_constraints(self):
        field_for_test = Field(min_length=1, max_length=5)
        self.assertEqual(field_for_test.min_length, 1)
        self.assertEqual(field_for_test.max_length, 5)
        self.assertIsNone(field_for_test.ge)
        self.assertIsNone(field_for_test.le)
        self.assertIsNone(field_for_test.gt)
        self.assertIsNone(field_for_test.lt)


class SimpleValidatorTest(unittest.TestCase):
    str_validator = StringParameterValidator(Field(min_length=1, max_length=5), allow_none=True)
    int_validator = IntParameterValidator(Field(ge=1, le=10), Field(ge=2, lt=12), special_values=[-1], allow_none=True)
    float_validator = FloatParameterValidator(Field(ge=3.14, le=11.4514), Field(ge=2, lt=12.1), allow_none=True)

    def test_str(self):
        self.str_validator.validate(STR_123)
        with self.assertRaises(ValueError):
            self.str_validator.validate(STR_123456)

    def test_int(self):
        self.int_validator.validate(11)
        self.int_validator.validate(-1)
        self.int_validator.validate(None)
        with self.assertRaises(ValueError):
            self.int_validator.validate(14)
        with self.assertRaises(ValueError):
            self.int_validator.validate(12)

    def test_float(self):
        self.float_validator.validate(11.1)
        with self.assertRaises(ValueError):
            self.float_validator.validate(14.7)
        with self.assertRaises(ValueError):
            self.float_validator.validate(12.1)
        self.float_validator.validate(4)


class ListValidatorTest(unittest.TestCase):
    int_validator = IntParameterValidator(Field(ge=1, le=10), Field(ge=2, le=12), allow_none=True)
    list_validator = ListParameterValidator(int_validator, Field(min_length=1, max_length=4))

    def test(self):
        self.list_validator.validate([11])
        self.list_validator.validate([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            self.list_validator.validate([14])
        with self.assertRaises(ValueError):
            self.list_validator.validate([1, 2, 3, 4, 5])


class DictionaryValidatorTest(unittest.TestCase):
    int_validator = IntParameterValidator(Field(ge=1, le=10), Field(ge=2, le=12), allow_none=True)
    str_validator = StringParameterValidator(Field(min_length=1, max_length=5), allow_none=True)
    list_validator = ListParameterValidator(int_validator, Field(min_length=1, max_length=4))
    dict_validator = DictionaryParameterValidator({
        KEY1: int_validator,
        KEY2: str_validator,
        KEY3: list_validator
    }, allow_addition_key=False)

    def test_key(self):
        self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123})

    def test_value(self):
        self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 99, KEY2: STR_123, KEY3: [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123456, KEY3: [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3, 4, 5, 6]})


class CompositeValidatorTest(unittest.TestCase):
    int_validator = IntParameterValidator(Field(ge=1, le=10), Field(ge=2, le=12), allow_none=True)
    str_validator = StringParameterValidator(Field(min_length=1, max_length=5), allow_none=True)
    list_validator = ListParameterValidator(int_validator, Field(min_length=1, max_length=4))
    composite_validator = CompositeParameterValidator({
        int: int_validator,
        str: str_validator,
        list: list_validator
    })
    dict_validator = DictionaryParameterValidator({
        KEY1: int_validator,
        KEY2: str_validator,
        KEY3: list_validator,
        KEY4: composite_validator
    })

    def test(self):
        self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: 1})
        self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: STR_123})
        self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 99, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: 1})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123456, KEY3: [1, 2, 3], KEY4: 1})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3, 4, 5, 6], KEY4: 1})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: 99})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: STR_123456})
        with self.assertRaises(ValueError):
            self.dict_validator.validate({KEY1: 1, KEY2: STR_123, KEY3: [1, 2, 3], KEY4: [1, 2, 3, 4, 5]})


class ValidatorsForDataClassTest:
    int_validator = IntParameterValidator(Field(ge=1, le=10), Field(ge=2, le=12), allow_none=True)
    str_validator = StringParameterValidator(Field(min_length=1, max_length=5), allow_none=True)
    list_validator = ListParameterValidator(int_validator, Field(min_length=1, max_length=4))
    composite_validator = CompositeParameterValidator({
        int: int_validator,
        str: str_validator
    })
    dict_validator = DictionaryParameterValidator({
        KEY1: int_validator,
        KEY2: str_validator,
        KEY3: list_validator,
        KEY4: composite_validator
    })


class DataClassTest(unittest.TestCase):
    @dataclass
    class ValidatedDataClass:
        int_field: Optional[int] = field(default=None, metadata={
            'validator': ValidatorsForDataClassTest.int_validator})
        str_field: Optional[str] = field(default=None, metadata={
            'validator': ValidatorsForDataClassTest.str_validator})
        list_field: Optional[List[int]] = field(default=None, metadata={
            'validator': ValidatorsForDataClassTest.list_validator})
        dict_field: Optional[Dict[str, Any]] = field(default=None,
                                                     metadata={'validator': ValidatorsForDataClassTest.dict_validator})

        def __post_init__(self):
            for field_name, field_value in self.__dataclass_fields__.items():
                validator: ParameterValidator = field_value.metadata.get('validator')
                if validator:
                    validator.validate(getattr(self, field_name), field_name)

    def test(self):
        self.ValidatedDataClass(
            int_field=1,
            str_field=STR_123,
            list_field=[1, 2, 3],
            dict_field={KEY1: 2, KEY2: TEST, KEY3: [1, 2], KEY4: 1}
        )
        with self.assertRaises(ValueError):
            self.ValidatedDataClass(
                int_field=1,
                str_field=STR_123456,
                list_field=[1, 2, 3],
                dict_field={KEY1: 2, KEY2: TEST, KEY3: [1, 2], KEY4: STR_123456}
            )


class ValidationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = ValidationPipeline()

    def test_add(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.assertEqual(len(self.pipeline.pipeline), 1)

    def test_add_with_position(self):
        self.pipeline.add(ValidationStrategies.validate_none, True, position=0)
        self.pipeline.add(ValidationStrategies.validate_none, False, position=0)
        self.assertEqual(len(self.pipeline.pipeline), 2)
        self.assertEqual(self.pipeline.pipeline[0][1], (False,))

    def test_remove(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.pipeline.remove(ValidationStrategies.validate_none)
        self.assertEqual(len(self.pipeline.pipeline), 0)

    def test_remove_with_position(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.pipeline.add(ValidationStrategies.validate_none, False)
        self.pipeline.remove(ValidationStrategies.validate_none, position=0)
        self.assertEqual(len(self.pipeline.pipeline), 1)
        self.assertEqual(self.pipeline.pipeline[0][1], (False,))

    def test_update(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.pipeline.update(ValidationStrategies.validate_none, False)
        self.assertEqual(self.pipeline.pipeline[0][1], (False,))

    def test_update_with_position(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.pipeline.add(ValidationStrategies.validate_none, False)
        self.pipeline.update(ValidationStrategies.validate_none, False, position=0)
        self.assertEqual(self.pipeline.pipeline[0][1], (False,))
        self.assertEqual(self.pipeline.pipeline[1][1], (False,))

    def test_refresh(self):
        self.pipeline.add(ValidationStrategies.validate_none, True)
        self.pipeline.refresh()
        self.assertEqual(len(self.pipeline.pipeline), 0)

    def test_validate(self):
        self.pipeline.add(ValidationStrategies.validate_none, False)
        self.pipeline.validate(TEST)


class ArgumentParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = ArgumentParser(description="Test parser")

    def test_add_argument(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        args = self.parser.parse_args([f'--{TEST_ARG}', '5'])
        self.assertEqual(args.test_arg, 5)

    def test_update_argument(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=StringParameterValidator(Field(max_length=4)))
        self.parser.update_argument(f'--{TEST_ARG}', new_dest=NEW_TEST_ARG, new_option_strings=[f'--{NEW_TEST_ARG}'],
                                    type=str)
        args = self.parser.parse_args([f'--{NEW_TEST_ARG}', TEST])
        self.assertEqual(args.new_test_arg, TEST)

    def test_update_argument_dest(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        self.parser.update_argument(f'--{TEST_ARG}', new_dest=NEW_TEST_ARG)
        args = self.parser.parse_args([f'--{TEST_ARG}', '5'])
        self.assertEqual(args.new_test_arg, 5)

    def test_update_argument_option_strings(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        self.parser.update_argument(f'--{TEST_ARG}', new_option_strings=[f'--{NEW_TEST_ARG}'])
        args = self.parser.parse_args([f'--{NEW_TEST_ARG}', '5'])
        self.assertEqual(args.test_arg, 5)

    def test_update_argument_not_found(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        with self.assertRaises(ValueError):
            self.parser.update_argument('--non_existent_arg', new_dest=NEW_TEST_ARG)

    def test_update_argument_invalid_attribute(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        with self.assertRaises(AttributeError):
            self.parser.update_argument(f'--{TEST_ARG}', invalid_attr='value')

    def test_validate_arguments(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        self.parser.parse_args([f'--{TEST_ARG}', '5'])

    def test_validate_argument(self):
        self.parser.add_argument(f'--{TEST_ARG}', type=int, validator=IntParameterValidator(Field(ge=1, le=10)))
        self.parser.parse_args([f'--{TEST_ARG}', '5'])


if __name__ == '__main__':
    unittest.main()