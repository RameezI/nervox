# Copyright(c) 2023 Rameez Ismail - All Rights Reserved
# Author: Rameez Ismail
# Email: rameez.ismaeel@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nervox.utils import capture_params

class Transform:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__user_init__ = cls.__init__ 
        # if the user has not already decorated the __init__ method, decorate it...         
        if not hasattr(cls.__user_init__, '_wrapper_capture_params_'):
            cls.__user_init__ = capture_params(cls.__init__, **kwargs)

        def _wrapped_init(self, *args, **kwargs):
                # call the user's __init__ method
                super().__init__()
                self.__user_init__(*args, **kwargs)

        cls.__init__ = _wrapped_init
