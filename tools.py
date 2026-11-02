import html
import os
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

# ======================= CONFIG (hardcode keys here if you add paid APIs) =======================
MODEL_NAME = "openai/gpt-oss-120b"
MODEL_PROVIDER = "groq"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools for news, weather, web search, "
    "Wikipedia, stocks, crypto, currency, dictionary, Stack Overflow, Hacker News, "
    "IP lookup, arXiv, books, air quality, recipes, translation "
    "and PyPI. Use a tool whenever it can answer the question instead of guessing. "
    "For news or headlines use get_news. "
    "Call the same tool at most twice per question. If a tool errors or returns nothing, "
    "do NOT keep retrying: tell the user what happened and answer with what you have. "
    "Keep answers short and clear."
)


# ======================= TOOLS =======================


# ======================= 1. WEATHER (Open-Meteo) =======================
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city, e.g. 'Hyderabad' or 'London'."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"name": city, "count": 1},
        )
        geo_response.raise_for_status()
        results = geo_response.json().get("results")
        place = results[0] if results else None
        if not place:
            return f"City '{city}' not found."
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "wind_speed_10m,precipitation",
            },
        )
        response.raise_for_status()
        c = response.json()["current"]
        return (
            f"Weather in {place['name']}, {place.get('country', '')}: "
            f"{c['temperature_2m']}°C (feels like {c['apparent_temperature']}°C), "
            f"humidity {c['relative_humidity_2m']}%, wind {c['wind_speed_10m']} km/h, "
            f"precipitation {c['precipitation']} mm."
        )
    except Exception as e:
        return f"Weather error: {e}"


# ======================= 2. WEB SEARCH (DuckDuckGo) =======================
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results with title, link and snippet. For news headlines use get_news instead."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return "Web search error: search package missing. Run: pip install -U ddgs"
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Web search error: {e}. Do not retry; tell the user search is unavailable."
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{i}. {r['title']}\n{r['href']}\n{r['body']}"
        for i, r in enumerate(results, 1)
    )


# ======================= NEWS (Google News RSS, no key) =======================
@tool
def get_news(topic: str = "", max_results: int = 5, country: str = "IN") -> str:
    """Get the latest news headlines. Leave topic empty for top headlines, or give a topic like 'cricket' or 'AI'. country is a 2-letter code like 'IN' or 'US'."""
    try:
        cc = country.upper()
        params = {"hl": f"en-{cc}", "gl": cc, "ceid": f"{cc}:en"}
        url = "https://news.google.com/rss"
        if topic.strip():
            url += "/search"
            params["q"] = topic
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params=params,
        )
        response.raise_for_status()
        items = ET.fromstring(response.text).findall(".//item")
        lines = []
        for i, item in enumerate(items[:max_results], 1):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            lines.append(f"{i}. {title}\n{pub}\n{link}")
        return "\n\n".join(lines) or "No news found."
    except Exception as e:
        return f"News error: {e}. Do not retry; tell the user the news service is unavailable."


# ======================= 3. WIKIPEDIA =======================
@tool
def wikipedia_summary(topic: str) -> str:
    """Get a short Wikipedia summary for a topic, person, place or concept."""
    try:
        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"action": "query", "list": "search", "srsearch": topic,
                    "srlimit": 1, "format": "json"},
        )
        search_response.raise_for_status()
        hits = search_response.json()["query"]["search"]
        if not hits:
            return f"No Wikipedia page found for '{topic}'."
        title = hits[0]["title"]
        page_response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
        )
        page_response.raise_for_status()
        page = page_response.json()
        return f"{page['title']}: {page['extract']}\nSource: {page['content_urls']['desktop']['page']}"
    except Exception as e:
        return f"Wikipedia error: {e}"


# ======================= 4. STOCK PRICE (Yahoo Finance) =======================
@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a ticker symbol, e.g. 'AAPL', 'TSLA', 'RELIANCE.NS'."""
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"interval": "1d", "range": "1d"},
        )
        response.raise_for_status()
        meta = response.json()["chart"]["result"][0]["meta"]
        price = meta["regularMarketPrice"]
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        change = f" ({(price - prev) / prev * 100:+.2f}% vs previous close)" if prev else ""
        return f"{meta['symbol']}: {price} {meta['currency']}{change}"
    except Exception as e:
        return f"Stock error for '{symbol}': {e}"


# ======================= 5. CRYPTO PRICE (CoinGecko) =======================
@tool
def get_crypto_price(coin_id: str, currency: str = "usd") -> str:
    """Get a cryptocurrency price. coin_id is the CoinGecko id, e.g. 'bitcoin', 'ethereum', 'solana'."""
    try:
        cur = currency.lower()
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"ids": coin_id.lower(), "vs_currencies": cur,
                    "include_24hr_change": "true"},
        )
        response.raise_for_status()
        data = response.json()
        if coin_id.lower() not in data:
            return f"Coin '{coin_id}' not found. Use the CoinGecko id (e.g. 'bitcoin')."
        d = data[coin_id.lower()]
        return f"{coin_id}: {d[cur]} {cur.upper()} (24h change: {d.get(cur + '_24h_change', 0):+.2f}%)"
    except Exception as e:
        return f"Crypto error: {e}"


# ======================= 6. CURRENCY CONVERTER (Frankfurter) =======================
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using ISO codes, e.g. 100 USD to INR."""
    try:
        response = requests.get(
            "https://api.frankfurter.app/latest",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
        )
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
        return f"{amount} {from_currency.upper()} = {rate} {to_currency.upper()} (as of {data['date']})"
    except Exception as e:
        return f"Currency error: {e}"


# ======================= 7. DICTIONARY (dictionaryapi.dev) =======================
@tool
def define_word(word: str) -> str:
    """Get the definition, part of speech and an example for an English word."""
    try:
        response = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(word)}",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        lines = []
        for meaning in data[0]["meanings"][:3]:
            d = meaning["definitions"][0]
            line = f"({meaning['partOfSpeech']}) {d['definition']}"
            if d.get("example"):
                line += f' — e.g. "{d["example"]}"'
            lines.append(line)
        return f"{word}:\n" + "\n".join(lines)
    except Exception:
        return f"No definition found for '{word}'."


# ======================= 8. STACK OVERFLOW (Stack Exchange) =======================
@tool
def stackoverflow_search(query: str, max_results: int = 5) -> str:
    """Search Stack Overflow questions. Useful for programming errors and how-to questions."""
    try:
        response = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"order": "desc", "sort": "relevance", "q": query,
                    "site": "stackoverflow", "pagesize": max_results},
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return "No Stack Overflow results found."
        return "\n\n".join(
            f"{i}. {html.unescape(q['title'])}\n{q['link']}\n"
            f"score: {q['score']}, answers: {q['answer_count']}, answered: {q['is_answered']}"
            for i, q in enumerate(items, 1)
        )
    except Exception as e:
        return f"Stack Overflow error: {e}"


# ======================= 9. HACKER NEWS (Algolia) =======================
@tool
def hacker_news_search(query: str, max_results: int = 5) -> str:
    """Search Hacker News stories for tech news and discussions on a topic."""
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"query": query, "tags": "story", "hitsPerPage": max_results},
        )
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", [])
        if not hits:
            return "No Hacker News stories found."
        return "\n\n".join(
            f"{i}. {h['title']}\n{h.get('url') or 'https://news.ycombinator.com/item?id=' + h['objectID']}\n"
            f"points: {h.get('points')}, comments: {h.get('num_comments')}, "
            f"discussion: https://news.ycombinator.com/item?id={h['objectID']}"
            for i, h in enumerate(hits, 1)
        )
    except Exception as e:
        return f"Hacker News error: {e}"


# ======================= 10. IP GEOLOCATION (ipwho.is) =======================
@tool
def get_ip_location(ip_address: str = "") -> str:
    """Look up the location, timezone and ISP of an IP address. Leave empty for the server's own IP."""
    try:
        response = requests.get(
            f"https://ipwho.is/{quote(ip_address)}",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
        d = response.json()
        if not d.get("success", True):
            return f"IP lookup failed: {d.get('message')}"
        return (
            f"IP {d['ip']}: {d.get('city')}, {d.get('region')}, {d.get('country')}\n"
            f"Coordinates: {d.get('latitude')}, {d.get('longitude')}\n"
            f"Timezone: {(d.get('timezone') or {}).get('id')}, "
            f"ISP: {(d.get('connection') or {}).get('isp')}"
        )
    except Exception as e:
        return f"IP lookup error: {e}"


# ======================= 11. ARXIV PAPERS (arXiv API) =======================
@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for research papers on a topic. Returns title, authors, summary and link."""
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"search_query": f"all:{query}", "max_results": max_results,
                    "sortBy": "relevance"},
        )
        response.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = ET.fromstring(response.text).findall("a:entry", ns)
        if not entries:
            return "No arXiv papers found."
        out = []
        for i, e in enumerate(entries, 1):
            title = " ".join(e.find("a:title", ns).text.split())
            summary = " ".join(e.find("a:summary", ns).text.split())[:300]
            authors = ", ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)[:3])
            link = e.find("a:id", ns).text
            out.append(f"{i}. {title}\nAuthors: {authors}\n{summary}...\n{link}")
        return "\n\n".join(out)
    except Exception as e:
        return f"arXiv error: {e}"


# ======================= 12. BOOKS (Open Library) =======================
@tool
def search_books(query: str, max_results: int = 5) -> str:
    """Search for books by title, author or topic. Returns title, author and first publish year."""
    try:
        response = requests.get(
            "https://openlibrary.org/search.json",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"q": query, "limit": max_results,
                    "fields": "title,author_name,first_publish_year,key"},
        )
        response.raise_for_status()
        data = response.json()
        docs = data.get("docs", [])
        if not docs:
            return "No books found."
        return "\n".join(
            f"{i}. {d['title']} by {', '.join(d.get('author_name', ['Unknown'])[:2])} "
            f"({d.get('first_publish_year', 'n/a')}) - https://openlibrary.org{d['key']}"
            for i, d in enumerate(docs, 1)
        )
    except Exception as e:
        return f"Books error: {e}"


# ======================= 13. AIR QUALITY (Open-Meteo) =======================
@tool
def get_air_quality(city: str) -> str:
    """Get the current air quality index (AQI) and PM2.5/PM10 levels for a city."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"name": city, "count": 1},
        )
        geo_response.raise_for_status()
        results = geo_response.json().get("results")
        place = results[0] if results else None
        if not place:
            return f"City '{city}' not found."
        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"latitude": place["latitude"], "longitude": place["longitude"],
                    "current": "us_aqi,pm10,pm2_5"},
        )
        response.raise_for_status()
        c = response.json()["current"]
        aqi = c["us_aqi"]
        level = ("Good" if aqi <= 50 else "Moderate" if aqi <= 100 else
                 "Unhealthy for sensitive groups" if aqi <= 150 else
                 "Unhealthy" if aqi <= 200 else "Very unhealthy" if aqi <= 300 else "Hazardous")
        return (f"Air quality in {place['name']}: US AQI {aqi} ({level}), "
                f"PM2.5 {c['pm2_5']} µg/m³, PM10 {c['pm10']} µg/m³")
    except Exception as e:
        return f"Air quality error: {e}"


# ======================= 14. RECIPES (TheMealDB) =======================
@tool
def search_recipes(dish: str) -> str:
    """Search for a recipe by dish name, e.g. 'biryani' or 'pasta'. Returns ingredients and instructions."""
    try:
        response = requests.get(
            "https://www.themealdb.com/api/json/v1/1/search.php",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"s": dish},
        )
        response.raise_for_status()
        data = response.json()
        meals = data.get("meals")
        if not meals:
            return f"No recipe found for '{dish}'."
        m = meals[0]
        ingredients = [
            f"{(m.get(f'strMeasure{i}') or '').strip()} {m[f'strIngredient{i}'].strip()}".strip()
            for i in range(1, 21) if (m.get(f"strIngredient{i}") or "").strip()
        ]
        return (
            f"{m['strMeal']} ({m['strCategory']}, {m['strArea']})\n"
            f"Ingredients: {', '.join(ingredients)}\n"
            f"Instructions: {m['strInstructions'][:800]}..."
        )
    except Exception as e:
        return f"Recipe error: {e}"


# ======================= 15. TRANSLATE (MyMemory) =======================
@tool
def translate_text(text: str, target_language: str, source_language: str = "en") -> str:
    """Translate text between languages using 2-letter codes, e.g. target 'hi' (Hindi), 'te' (Telugu), 'es' (Spanish)."""
    try:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
            params={"q": text, "langpair": f"{source_language}|{target_language}"},
        )
        response.raise_for_status()
        data = response.json()
        return data["responseData"]["translatedText"]
    except Exception as e:
        return f"Translation error: {e}"


# ======================= 16. PYPI PACKAGE INFO (PyPI) =======================
@tool
def pypi_package_info(package: str) -> str:
    """Get the latest version, summary and requirements for a Python package on PyPI."""
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{quote(package)}/json",
            headers={"User-Agent": "Mozilla/5.0 (agent-tools/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
        info = response.json()["info"]
        return (
            f"{info['name']} {info['version']}: {info.get('summary')}\n"
            f"Requires Python: {info.get('requires_python')}, License: {info.get('license') or 'n/a'}\n"
            f"Install: pip install {info['name']}\n"
            f"Page: {info['package_url']}"
        )
    except Exception as e:
        return f"PyPI error for '{package}': {e}"






ALL_TOOLS = [
    get_weather, web_search, get_news, wikipedia_summary, get_stock_price, get_crypto_price,
    convert_currency, define_word, stackoverflow_search, hacker_news_search,
    get_ip_location, arxiv_search, search_books, get_air_quality,
    search_recipes, translate_text, pypi_package_info,
]

# ======================= AGENT =======================
def build_agent():
    """Create the ReAct agent: LLM + tools + in-memory conversation history."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, api_key=api_key)
    return create_agent(
        model,
        ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,     # system prompt
        checkpointer=InMemorySaver(),    # remembers messages per thread_id
    )


def ask(agent, question: str, thread_id: str = "default"):
    """Send one question. Prints each tool call live. Returns (answer, tools_used)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    tools_used, answer = [], ""
    try:
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="updates",
        ):
            for update in step.values():
                if not update:
                    continue
                for m in update.get("messages", []):
                    for call in getattr(m, "tool_calls", None) or []:
                        tools_used.append(call["name"])
                        print(f"  -> {call['name']}({call['args']})", flush=True)
                    if m.type == "ai" and m.content and not getattr(m, "tool_calls", None):
                        answer = m.content
    except GraphRecursionError:
        answer = (
            "I got stuck in a tool loop and stopped. Tools tried: "
            f"{', '.join(tools_used[-6:])}. Try rephrasing, or run --test to check those tools."
        )
    return answer, tools_used


def chat():
    agent = build_agent()
    print("Agent ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        try:
            answer, tools_used = ask(agent, user_input)
        except Exception as e:
            print(f"\nError: {e}")
            continue
        print(f"\nAgent: {answer}")

if __name__ == "__main__":
    chat()