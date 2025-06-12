import requests
from bs4 import BeautifulSoup
import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available

if __name__ == "__main__": # if running as a script for individual testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.tools.tools import Tools

class searxSearch(Tools):
    def __init__(self, base_url: str = None):
        """
        A tool for searching a SearxNG instance and extracting URLs and titles.
        """
        super().__init__()
        self.tag = "web_search"
        self.name = "searxSearch"
        self.description = "A tool for searching a SearxNG for web search"
        
        # Try multiple ways to get the base URL
        self.base_url = base_url or os.getenv("SEARXNG_BASE_URL")
        
        # If still no URL, try to load from .env file in project root
        if not self.base_url:
            self._try_load_env_file()
            self.base_url = os.getenv("SEARXNG_BASE_URL")
        
        # Default fallback
        if not self.base_url:
            self.base_url = "http://127.0.0.1:8080"
            print(f"Warning: Using default SearxNG URL: {self.base_url}")
        
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        self.paywall_keywords = [
            "Member-only", "access denied", "restricted content", "404", "this page is not working"
        ]
    
    def _try_load_env_file(self):
        """Try to load environment variables from .env file"""
        try:
            # Find project root by looking for config.ini
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):
                if os.path.exists(os.path.join(project_root, "config.ini")):
                    break
                project_root = os.path.dirname(project_root)
            
            env_file = os.path.join(project_root, ".env")
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            # Remove quotes if present
                            value = value.strip('\'"')
                            os.environ[key] = value
        except Exception:
            pass  # Silently fail if we can't load .env

    def link_valid(self, link):
        """check if a link is valid."""
        # TODO find a better way
        if not link.startswith("http"):
            return "Status: Invalid URL"
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            response = requests.get(link, headers=headers, timeout=5)
            status = response.status_code
            if status == 200:
                content = response.text.lower()
                if any(keyword in content for keyword in self.paywall_keywords):
                    return "Status: Possible Paywall"
                return "Status: OK"
            elif status == 404:
                return "Status: 404 Not Found"
            elif status == 403:
                return "Status: 403 Forbidden"
            else:
                return f"Status: {status} {response.reason}"
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def check_all_links(self, links):
        """Check all links, one by one."""
        # TODO Make it asyncromous or smth
        statuses = []
        for i, link in enumerate(links):
            status = self.link_valid(link)
            statuses.append(status)
        return statuses
    
    def execute(self, blocks: list, safety: bool = False) -> str:
        """Executes a search query against a SearxNG instance using POST and extracts URLs and titles."""
        if not blocks:
            return "Error: No search query provided."

        query = blocks[0].strip()
        if not query:
            return "Error: Empty search query provided."

        search_url = f"{self.base_url}/search"
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Pragma': 'no-cache',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': self.user_agent
        }
        data = f"q={query}&categories=general&language=auto&time_range=&safesearch=0&theme=simple".encode('utf-8')
        try:
            response = requests.post(search_url, headers=headers, data=data, verify=False)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            for article in soup.find_all('article', class_='result'):
                url_header = article.find('a', class_='url_header')
                if url_header:
                    url = url_header['href']
                    title = article.find('h3').text.strip() if article.find('h3') else "No Title"
                    description = article.find('p', class_='content').text.strip() if article.find('p', class_='content') else "No Description"
                    results.append(f"Title:{title}\nSnippet:{description}\nLink:{url}")
            if len(results) == 0:
                return "No search results, web search failed."
            return "\n\n".join(results)  # Return results as a single string, separated by newlines
        except requests.exceptions.RequestException as e:
            # More helpful error message with troubleshooting steps
            error_msg = f"""
SearxNG search failed. This usually means the SearxNG service is not running.

Error details: {str(e)}

Troubleshooting steps:
1. Check if SearxNG service is running: curl {self.base_url}
2. Start services: ./start_services.sh
3. Check Docker containers: docker ps | grep searxng
4. Check Docker logs: docker logs agenticseek-searxng-1
5. Verify environment variable: echo $SEARXNG_BASE_URL
6. Run auto-fix: ./scripts/fix_searxng.sh

If the problem persists, run: ./scripts/fix_searxng.sh
"""
            raise Exception(error_msg) from e

    def execution_failure_check(self, output: str) -> bool:
        """
        Checks if the execution failed based on the output.
        """
        return "Error" in output

    def interpreter_feedback(self, output: str) -> str:
        """
        Feedback of web search to agent.
        """
        if self.execution_failure_check(output):
            return f"Web search failed: {output}"
        return f"Web search result:\n{output}"

if __name__ == "__main__":
    search_tool = searxSearch(base_url="http://127.0.0.1:8080")
    result = search_tool.execute(["are dog better than cat?"])
    print(result)
