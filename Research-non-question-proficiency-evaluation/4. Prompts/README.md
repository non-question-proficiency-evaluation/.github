<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Prompts Directory</div>

This directory contains prompt files and templates used for research, AI interactions, and project documentation.

## 1. Files in This Directory

### 1.1. `research_prompt.md`
Initial research prompt used to guide AI assistants in conducting deep research on the non-question proficiency evaluation framework. Contains project context, research objectives, and key research questions.

### 1.2. `research_scope_clarification.md`
Clarification document addressing scope questions about the research, including age groups, subject focus, and other parameters that guide the research direction.

### 1.3. `notebook_llm_synthesis_prompt.md`
Comprehensive prompt designed for notebook LLMs (with RAG capabilities) to synthesize and summarize multiple research reports. This prompt instructs the LLM to:
- Analyze three separate research reports (from ChatGPT, Gemini, and Opus)
- Extract key findings, methodologies, and recommendations from each
- Synthesize the information into a unified, comprehensive summary
- Identify areas of agreement, disagreement, and complementary insights
- Create a structured output with unified frameworks and implementation roadmaps

## 2. Usage

These prompts are designed to be used with various AI systems:
- **Research prompts** (`research_prompt.md`): For initial research tasks with AI assistants
- **Synthesis prompt** (`notebook_llm_synthesis_prompt.md`): For notebook LLMs that have access to the research documents via RAG

## 3. Project Context

All prompts in this directory relate to the **Non-Question Proficiency Evaluation Framework** project (NEXS-399), which aims to develop methods for estimating student proficiency gains from non-assessment learning activities (videos, PDFs, interactive content).

