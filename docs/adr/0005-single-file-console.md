# ADR-0005 控制台单文件零构建，不上 React + Vite

- 状态：采纳
- 日期：2026-09-23
- 作者：晨星

## 背景

需要一个能对话、能看引用、能看链路瀑布、能跑评测的界面。
两种做法：现代前端工程（React + Vite + 组件库），或单文件 HTML。

## 备选方案

**A. React + Vite + TypeScript。**
组件化、类型安全、生态成熟。但代价是：`node_modules` 数百 MB、
构建步骤进入复现链路、产物需要托管、CI 需要 node 环境与 `npm ci`、
以及「clone 下来还要 build 才能看到界面」。

**B. 服务端渲染 HTML 片段。** 交互（瀑布图、实时状态、评测展开）难以做顺。

**C. 单文件 `web/index.html`：内联全部 CSS / JS / SVG。**（本决策）

**D. 引入 CDN 上的轻量库（如 Alpine / htmx）。**
省代码，但**引入外部网络依赖**，与「离线可用」直接冲突。否决。

## 决策

采用 C。要求：

- 一个 HTML 文件，内联 `<style>` 与 `<script>`，**零外部资源**。
- 图标用内联 SVG，不用 emoji、不用图标字体。
- 由服务端 `FileResponse` 直接吐出，`/web` 同时挂静态目录便于单独引用。
- 前端测试用 **jsdom 真执行内联脚本**，而不是正则匹配 HTML 字符串。

## 理由

1. **复现链路最短**。`pip install -e . && stellarnx serve` 之后浏览器打开就是界面，
   没有 `npm install`、没有 build、没有 dev server。
2. **零外部依赖是可测的硬约束**。E2E 里有一条断言：

   ```python
   check("控制台零外部依赖", "cdn." not in page and "unpkg" not in page
         and 'src="http' not in page and "@import" not in page)
   ```

   这条断言存在的意义是：单文件方案最大的风险是「某天顺手加了个 CDN 链接」，
   而那种改动在开发机上永远看不出问题（有网），只在离线部署时炸。
3. **功能规模匹配**。六个面板、无路由、无状态管理、无多用户。
   引入组件框架解决的是不存在的问题。
4. **性能更好**。无框架运行时、无打包产物、首屏一个请求。

## 代价与补救

| 代价 | 补救 |
|---|---|
| 无类型检查 | 用 jsdom 冒烟测试覆盖交互（30 项断言）；DOM 契约写在 `SPEC.md` 第 8 章 |
| 无组件复用 | 用三个纯函数承担渲染：`renderAnswer` / `renderTraces` / `renderWaterfall`，都在同一个 IIFE 内且无副作用依赖 |
| 状态靠手写 DOM 操作 | 状态集中为 4 个变量（`history` / `traces` / `API` / `HEALTHY`），渲染函数从状态重绘 |
| 文件会变长（约 1000 行） | 分区注释明确划分「对话 / 检索 / 知识库 / 追踪 / 评测 / 设置」六段 |
| 无主题系统 | CSS 变量 + `data-theme` 属性，默认跟随 `prefers-color-scheme`，可手动切换并持久化 |

## 前端测试为什么不用正则

最初想过「读 HTML，断言里面有没有 `id="chat"`」。问题是：
**字符串存在 ≠ 功能可用**。它证明不了标签切换生效、证明不了 `fetch` 发得出去、
证明不了答案渲染得出来、更证明不了内联脚本没有语法错误。

改用 jsdom：真解析、真执行内联脚本、把 `fetch` 换成桩，逐项断言 DOM 结果。
成本是多一个 `devDependency`（39 个包，17 秒装完），收益是控制台真正进入了测试覆盖。

> 补一个真实收益：写这套测试时立刻发现两处问题 —— 内联脚本里
> `var history = []` 会遮蔽 `window.history`（在 IIFE 内是合法的，但如果哪天
> 把脚本提到顶层就会静默改变行为），以及文档列表需要的 `chunks` 字段
> 后端接口根本没返回（`list_documents` 只查了 documents 表）。
> 第二个问题是纯后端缺陷，靠肉眼 review HTML 永远发现不了。

## 反悔路径

前端如果真要长成工程，`web/index.html` 可以直接作为 Vite 项目的入口文件，
现有的渲染函数逐个提取成组件。后端一行不用改 —— 因为界面只消费 REST 接口，
不依赖任何服务端渲染。

## 相关

- 文件：`web/index.html`
- 测试：`scripts/web_smoke.mjs`（30 项断言）、`scripts/e2e.py`（零外部依赖断言）
- 契约：`SPEC.md` 第 8 章
