import os
import re
import json
import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# ==================== 环境变量加载 ====================
def _load_dotenv():
    """从项目根目录的 .env 文件加载环境变量（不覆盖已有环境变量）。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ==================== 配置区 ====================
# 从环境变量读取 API 配置（复制 .env.example 为 .env 并填写真实值）
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 输入、输出和缓存文件路径（相对于项目根目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = str(_PROJECT_ROOT / "data" / "template_en.ts")
OUTPUT_FILE = str(_PROJECT_ROOT / "output" / "zh_CN.ts")
CACHE_FILE = str(_PROJECT_ROOT / "data" / "translation_cache.json")

# 批处理大小（每次发送给 AI 翻译的字符串数量）
BATCH_SIZE = 40
# 最大并发请求数，避免触发 API 频率限制 (Rate Limit)
CONCURRENT_REQUESTS = 50
# 单次 API 请求的最大重试次数
MAX_RETRIES = 5
# 重试时的基本等待秒数
RETRY_DELAY = 2
# ================================================


def should_translate(text: str) -> bool:
    """
    判断一个字符串是否需要翻译。
    跳过空字符串、纯数字、纯标点、URL、文件路径等，节省 Token 和费用。
    """
    if not text:
        return False
    text_stripped = text.strip()
    if not text_stripped:
        return False
    
    # 过滤 URL
    if text_stripped.startswith(("http://", "https://", "www.")):
        return False
    
    # 过滤文件路径 (包含斜杠且有后缀，或者常见的相对路径)
    if ("/" in text_stripped or "\\" in text_stripped) and re.search(r'\.[a-zA-Z0-9]+$', text_stripped):
        return False

    # 过滤纯数字、符号和标点。如果字符串中不含任何英文字母、汉字，则不需要翻译。
    # 这样可以过滤掉像 "+", "-", "12", "100/100", "::", "---" 等字符串。
    if not re.search(r'[a-zA-Z\u4e00-\u9fff]', text_stripped):
        return False
        
    return True


def clean_json_response(text: str) -> str:
    """
    清理 LLM 返回的 JSON 字符串，剥离 Markdown 代码块标记。
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def load_cache() -> dict:
    """
    加载已有的翻译缓存。
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 读取缓存文件失败 ({e})，将使用空缓存。")
    return {}


def save_cache(cache: dict):
    """
    保存翻译缓存到文件。
    """
    try:
        # 先写入临时文件，再重命名，防止写入中途崩溃损坏缓存
        temp_file = CACHE_FILE + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        os.rename(temp_file, CACHE_FILE)
    except Exception as e:
        print(f"错误: 保存缓存文件失败 ({e})")


async def translate_batch(client: AsyncOpenAI, batch: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    使用 DeepSeek API 翻译一批字符串。带有指数退避的重试机制。
    """
    system_prompt = (
        "You are a professional software localization translator for network engineering software. "
        "Translate the given English user interface strings of 'Cisco Packet Tracer' into Simplified Chinese (zh-CN).\n\n"
        "Guidelines:\n"
        "1. Translate the strings accurately and naturally into Chinese UI style.\n"
        "2. Keep technical nouns, protocol names (e.g. OSPF, BGP, VLAN, IP, TCP, UDP, Cisco, ICMP, DHCP, DNS, CLI), "
        "and brand names untranslated if they are typically used in English by IT professionals in China.\n"
        "3. Keep format specifiers (e.g. %1, %2, %n, %p) exactly as they are.\n"
        "4. Keep XML entities and HTML tags (e.g. <b>, <i>, <br>, <p>, &apos;, &quot;, &amp;, &lt;, &gt;) exactly as they are. "
        "Do not translate the tags themselves or modify the entities.\n"
        "5. Retain original punctuation, capitalization, leading/trailing whitespace, and newlines.\n"
        "6. Return ONLY the translations in the exact same JSON format as the input. Do not explain, "
        "do not add introductory or concluding text, and do not add any markdown formatting other than the JSON block itself."
    )

    user_prompt = f"Please translate the following JSON object containing string ID keys and English string values:\n{json.dumps(batch, ensure_ascii=False)}"

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=60.0
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("API returned an empty response.")
                
                cleaned_content = clean_json_response(content)
                result = json.loads(cleaned_content)
                
                # 校验返回结果中的键是否完整，并进行格式化
                formatted_result = {}
                for key in batch.keys():
                    if key in result:
                        formatted_result[key] = str(result[key])
                    elif str(key) in result:
                        formatted_result[key] = str(result[str(key)])
                    else:
                        # 缺失的键使用原词占位，后续处理
                        formatted_result[key] = batch[key]
                
                return formatted_result

            except Exception as e:
                # 打印错误并进行指数退避重试
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"\n批次翻译出错 ({e})。正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试，将在 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
        
        # 达到最大重试次数，返回空翻译（保留原文）
        print(f"\n错误: 批次翻译在 {MAX_RETRIES} 次尝试后全部失败。该批次将保留原文。")
        return {key: value for key, value in batch.items()}


async def main():
    print("=== 思科 PT 模拟器 AI 翻译汉化脚本 ===")
    
    # 检查 API Key 是否配置
    if not API_KEY:
        print("错误: 未检测到 DeepSeek API Key！")
        print("请复制项目根目录的 .env.example 为 .env，并在其中填写你的 API Key。")
        return

    # 1. 加载并解析 XML (template.ts)
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到源文件 '{INPUT_FILE}'，请确认它在当前工作目录下。")
        return
    
    print(f"正在解析源文件 '{INPUT_FILE}'...")
    try:
        tree = ET.parse(INPUT_FILE)
        root = tree.getroot()
    except Exception as e:
        print(f"错误: 解析 XML 文件失败 ({e})")
        return
    
    print("XML 文件解析成功。")

    # 2. 加载缓存
    cache = load_cache()
    print(f"加载了 {len(cache)} 条已有的翻译缓存。")

    # 3. 统计需要翻译的条目
    messages_to_process = []  # 存储需要填充的 (translation_element, source_text)
    unique_sources_to_translate = set()  # 需要调用 API 翻译的去重英文文本

    for context in root.findall('context'):
        for msg in context.findall('message'):
            source = msg.find('source')
            translation = msg.find('translation')
            
            if source is not None and translation is not None:
                source_text = source.text
                if not source_text:
                    continue
                
                # 判断翻译标签是否为 unfinished，或者是空的
                is_unfinished = translation.get('type') == 'unfinished' or not translation.text
                
                if is_unfinished:
                    messages_to_process.append((translation, source_text))
                    # 如果不需要翻译（例如 URL、数字等），在最后直接复制原文即可，不需要计入 API 翻译队列
                    if should_translate(source_text):
                        if source_text not in cache:
                            unique_sources_to_translate.add(source_text)

    total_unfinished = len(messages_to_process)
    total_need_api = len(unique_sources_to_translate)
    
    print(f"统计结果:")
    print(f" - 未完成翻译的总条目数: {total_unfinished}")
    print(f" - 过滤无须翻译的条目后，需要调用 API 翻译的去重文本数: {total_need_api}")

    # 4. 执行 API 翻译并更新缓存
    if total_need_api > 0:
        client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        # 将去重文本列表转化为字典，为了传给 LLM 的 JSON 格式
        sources_list = list(unique_sources_to_translate)
        batches = []
        for i in range(0, len(sources_list), BATCH_SIZE):
            batch_slice = sources_list[i:i + BATCH_SIZE]
            # 用字符串索引作为键，方便 LLM 对应返回
            batch_dict = {str(idx): text for idx, text in enumerate(batch_slice)}
            batches.append((batch_slice, batch_dict))
            
        print(f"共拆分为 {len(batches)} 个批次进行翻译 (每批最大 {BATCH_SIZE} 条，最大并发 {CONCURRENT_REQUESTS})...")
        
        # 创建并发任务列表
        tasks = []
        for _, batch_dict in batches:
            tasks.append(translate_batch(client, batch_dict, semaphore))
            
        # 使用 tqdm 显示异步进度条
        results = await tqdm_asyncio.gather(*tasks, desc="正在翻译进度")
        
        # 将翻译结果整合并保存至 cache
        new_translations_count = 0
        for i, batch_result in enumerate(results):
            batch_slice, _ = batches[i]
            for idx_str, translated_val in batch_result.items():
                idx = int(idx_str)
                if idx < len(batch_slice):
                    orig_val = batch_slice[idx]
                    # 确保翻译结果有效，如果返回的和原文一样且是多字节字符，或者翻译结果不为空
                    if translated_val and translated_val != orig_val:
                        cache[orig_val] = translated_val
                        new_translations_count += 1
                    elif translated_val == orig_val:
                        # 翻译相同也记录，避免重复请求
                        cache[orig_val] = translated_val
                        new_translations_count += 1
                        
        print(f"本次翻译新获取了 {new_translations_count} 条翻译结果。正在保存缓存...")
        save_cache(cache)
        print("缓存保存成功。")

    # 5. 更新 XML 结构
    print("正在将翻译应用到 XML 结构中...")
    translated_applied = 0
    copied_applied = 0
    
    for translation, source_text in messages_to_process:
        if not should_translate(source_text):
            # 不需要翻译的文本，直接复制原文，并移除 unfinished 属性
            translation.text = source_text
            translation.attrib.pop('type', None)
            copied_applied += 1
        elif source_text in cache:
            # 应用缓存中的翻译
            translation.text = cache[source_text]
            translation.attrib.pop('type', None)
            translated_applied += 1
        else:
            # 理论上应该都在 cache 里，如果没有（例如翻译失败），则保持现状
            pass

    print(f"应用完成:")
    print(f" - 成功汉化并填充的条目数: {translated_applied}")
    print(f" - 直接复制原文的条目数 (URL/数字/符号): {copied_applied}")

    # 6. 保存新的 XML 文件并恢复 DOCTYPE 头部
    print(f"正在保存最终的汉化文件到 '{OUTPUT_FILE}'...")
    try:
        import io
        # 写入内存 buffer
        buffer = io.BytesIO()
        tree.write(buffer, encoding='utf-8', xml_declaration=True)
        xml_bytes = buffer.getvalue()
        xml_str = xml_bytes.decode('utf-8')
        
        # 构造标准的 Qt TS XML 头部 (带有双引号的 xml 声明和 DOCTYPE TS)
        standard_header = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
        
        # 替换单引号声明
        if xml_str.startswith("<?xml version='1.0' encoding='utf-8'?>"):
            xml_str = xml_str.replace("<?xml version='1.0' encoding='utf-8'?>", standard_header, 1)
        else:
            # 如果没有找到对应的单引号声明，直接加在最前面
            xml_str = standard_header + xml_str
            
        # 写入文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(xml_str)
            
        print(f"汉化完成！生成文件: '{OUTPUT_FILE}'")
        
    except Exception as e:
        print(f"错误: 写入输出文件失败 ({e})")


if __name__ == '__main__':
    # 解决 Windows 下 asyncio 可能会抛出 Event Loop Closed 异常的问题
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
