# Cisco Packet Tracer 汉化翻译工具

使用 DeepSeek AI 将 Cisco Packet Tracer 的英文 UI 字符串（Qt TS 格式）批量翻译为简体中文。

## 项目结构

```
├── src/                  # 源代码
│   └── translate.py      # 主翻译脚本
├── data/                 # 数据文件
│   ├── template_en.ts    # 英文源模板（Qt TS 格式）
│   └── translation_cache.json  # 翻译缓存
├── output/               # 翻译产出
│   └── zh_CN.ts          # 汉化后的 TS 文件
├── archive/              # 历史文件归档
├── .env.example          # 环境变量配置模板
├── requirements.txt      # Python 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 3. 运行翻译

```bash
cd src
python translate.py
```

### 4. 获取结果

翻译完成后，汉化文件输出在 `output/zh_CN.ts`。

## 工作原理

1. 读取 `data/template_en.ts`（Qt TS XML 格式的英文 UI 字符串）
2. 通过 `translation_cache.json` 跳过已翻译的内容
3. 批量调用 DeepSeek API 翻译未完成的字符串
4. 合并缓存和应用翻译结果
5. 输出完整的 `output/zh_CN.ts`

## 翻译规则

- 技术术语和协议名保持原文（如 OSPF、BGP、VLAN、TCP/IP、CLI）
- 格式说明符（%1、%n 等）原样保留
- XML/HTML 标签原样保留
- URL 和文件路径不翻译
