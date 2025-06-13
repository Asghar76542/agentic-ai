import unittest
import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.tools.searxSearch import searxSearch

class TestAsyncLinkChecker(unittest.IsolatedAsyncioTestCase):
    async def test_link_valid_invalid_url(self):
        tool = searxSearch(base_url="http://localhost")
        result = await tool.link_valid("ftp://invalid")
        self.assertEqual(result, "Status: Invalid URL")

    @patch('aiohttp.ClientSession')
    async def test_check_all_links(self, mock_session_cls):
        mock_session = AsyncMock()
        # ClientSession() returns context manager yielding mock_session
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = AsyncMock()

        resp_ok = AsyncMock()
        resp_ok.status = 200
        resp_ok.text = AsyncMock(return_value="hello")
        cm_ok = MagicMock()
        cm_ok.__aenter__ = AsyncMock(return_value=resp_ok)
        cm_ok.__aexit__ = AsyncMock(return_value=None)
        
        resp_404 = AsyncMock()
        resp_404.status = 404
        resp_404.text = AsyncMock(return_value="not found")
        cm_404 = MagicMock()
        cm_404.__aenter__ = AsyncMock(return_value=resp_404)
        cm_404.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(side_effect=[cm_ok, cm_404])

        tool = searxSearch(base_url="http://localhost")
        results = await tool.check_all_links(["http://a", "http://b"])
        self.assertEqual(results, ["Status: OK", "Status: 404 Not Found"])

if __name__ == '__main__':
    unittest.main()
