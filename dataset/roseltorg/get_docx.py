# === roseltorg_mass_docx_downloader.py ===

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from bs4 import BeautifulSoup
import time
import os
import shutil
from urllib.parse import urljoin

SEARCH_URL = "https://www.roseltorg.ru/procedures/search"
BASE_URL = "https://www.roseltorg.ru"
DOWNLOAD_DIR = os.path.abspath("downloads_all_in_one")
GECKODRIVER_PATH = "/snap/bin/geckodriver"
MAX_PROCEDURES = 50  # можно увеличить

def scroll_and_collect_procedures(driver, max_links=MAX_PROCEDURES):
    driver.get(SEARCH_URL)
    time.sleep(3)
    last_height = driver.execute_script("return document.body.scrollHeight")
    procedure_urls = set()

    while len(procedure_urls) < max_links:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.select("a.search-results__link[href^='/procedure/B']")
        for a in links:
            full_url = urljoin(BASE_URL, a["href"])
            procedure_urls.add(full_url)
            if len(procedure_urls) >= max_links:
                break
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    return list(procedure_urls)

def setup_driver():
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    profile = FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.dir", DOWNLOAD_DIR)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    profile.set_preference("pdfjs.disabled", True)

    options = Options()
    options.profile = profile
    options.headless = True

    driver = webdriver.Firefox(
        service=Service(GECKODRIVER_PATH),
        options=options
    )
    return driver

def download_docx_from_procedure(driver, procedure_url):
    print(f"🔍 Открываем процедуру: {procedure_url}")
    try:
        driver.get(procedure_url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.select(".documents__list a.documents__link")

        docx_links = [
            link for link in links
            if link.get("href") and link.get("title", "").strip().lower().endswith(".docx_files")
        ]

        if not docx_links:
            print("⚠️ Нет .docx_files документов.")
            return

        print(f"📄 Найдено .docx_files: {len(docx_links)}")
        for link in docx_links:
            href = link.get("href")
            title = link.get("title").strip().replace(" ", "_")
            print(f"⬇️ Скачиваем: {title}")
            script = f'''
                var a = document.createElement('a');
                a.href = "{href}";
                a.download = '';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            '''
            driver.execute_script(script)
            time.sleep(2)
    except Exception as e:
        print(f"❌ Ошибка при обработке {procedure_url}: {e}")

def main():
    driver = setup_driver()
    print("🔄 Ищем ссылки на процедуры...")
    procedure_urls = scroll_and_collect_procedures(driver, MAX_PROCEDURES)
    print(f"🔗 Всего процедур: {len(procedure_urls)}")

    for url in procedure_urls:
        download_docx_from_procedure(driver, url)

    print("⌛ Ожидание завершения загрузок...")
    time.sleep(10)
    driver.quit()

    files = os.listdir(DOWNLOAD_DIR)
    if files:
        print(f"✅ Скачано файлов: {len(files)}")
        for f in files:
            print(f"📁 {f}")
    else:
        print("❌ Ни одного файла не скачано.")

if __name__ == "__main__":
    main()
