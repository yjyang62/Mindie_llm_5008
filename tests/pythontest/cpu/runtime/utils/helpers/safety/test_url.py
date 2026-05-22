# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from mindie_llm.runtime.utils.helpers.safety.url import filter_urls_from_error


class TestUrlUtils(unittest.TestCase):

    def test_domain_replacement(self):
        """
        Verify domain URL replacement in error message.
        - Input contains domain URL with port
        - Only '://example.com:8080' is replaced, 'http' remains
        """
        error = Exception("Visit http://example.com:8080 for details")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "Visit http*** for details")

    def test_ipv4_replacement(self):
        """
        Verify IPv4 address replacement in error message.
        - Input contains IPv4 URL with port
        - Only '://192.168.1.1:80' is replaced, 'http' remains
        """
        error = Exception("Connection to http://192.168.1.1:80 failed")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "Connection to http*** failed")

    def test_ipv6_replacement(self):
        """
        Verify IPv6 address replacement in error message.
        - Input contains IPv6 URL with port
        - Only '://[::1]:8080' is replaced, 'http' remains
        """
        error = Exception("Error with http://[::1]:8080")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "Error with http***")

    def test_multiple_urls(self):
        """
        Verify multiple URL replacements in single message.
        - Input contains multiple different URL types
        - Only '://...' parts are replaced, protocol prefixes remain
        """
        error = Exception("http://example.com and http://192.168.1.1 and http://[::1]")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "http*** and http*** and http***")

    def test_no_urls(self):
        """
        Verify no replacement when no URLs present.
        - Input contains no URLs
        - Output should remain unchanged
        """
        error = Exception("No URLs here")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "No URLs here")

    def test_non_string_args(self):
        """
        Verify non-string arguments remain unchanged.
        - Input contains mixed argument types
        - Only string arguments with URLs are modified
        """
        error = Exception(123, "http://example.com", None)
        filtered = filter_urls_from_error(error)
        self.assertEqual(filtered.args[0], 123)
        self.assertEqual(filtered.args[1], "http***")
        self.assertIsNone(filtered.args[2])

    def test_mixed_case_urls(self):
        """
        Verify case-insensitive URL matching.
        - Input contains uppercase URL
        - Only '://ExAmPlE.CoM:443' is replaced, 'HTTPS' remains
        """
        error = Exception("HTTPS://ExAmPlE.CoM:443")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "HTTPS***")

    def test_url_with_query_params(self):
        """
        Verify URL with query parameters replacement.
        - Input contains URL with query string
        - Only '://example.com' is replaced, rest remains
        """
        error = Exception("http://example.com/path?query=1")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "http***/path?query=1")

    def test_url_with_fragment(self):
        """
        Verify URL with fragment identifier replacement.
        - Input contains URL with fragment
        - Only '://example.com' is replaced, fragment remains
        """
        error = Exception("http://example.com#section")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "http***#section")

    def test_non_http_urls(self):
        """
        Verify non-HTTP protocol URLs replacement.
        - Input contains FTP URL
        - Only '://example.com:21' is replaced, 'ftp' remains
        """
        error = Exception("ftp://example.com:21")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "ftp***")

    def test_invalid_ipv6_format(self):
        """
        Verify invalid IPv6 format is not replaced.
        - Input contains malformed IPv6
        - Should not be matched by regex (correct behavior)
        """
        error = Exception("http://[::1:8080]")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "http***")

    def test_url_in_middle_of_string(self):
        """
        Verify URL embedded in longer string replacement.
        - Input contains URL surrounded by text
        - Only '://192.168.1.1:8080' is replaced, rest remains
        """
        error = Exception("Error: http://192.168.1.1:8080 is unreachable")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "Error: http*** is unreachable")

    def test_url_with_unicode(self):
        """
        Verify Unicode characters in domain are not matched.
        - Input contains Unicode domain
        - Should not be matched by regex (correct behavior)
        """
        error = Exception("http://éxample.com")
        filtered = filter_urls_from_error(error)
        self.assertEqual(str(filtered), "http://éxample.com")


if __name__ == "__main__":
    unittest.main()
